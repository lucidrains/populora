from __future__ import annotations

import math
import torch
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init
from torch.linalg import qr, svd

from einops import einsum, rearrange
from torch_einops_utils import tree_map_tensor

from contextlib import contextmanager

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def extract_dict(v, k):
    return v[k] if isinstance(v, dict) else v

# tensor helpers

def _efficient_svd_of_lora(weight_down, weight_up):
    Q_A, R_A = qr(weight_down)
    Q_B, R_B = qr(weight_up)

    C = einsum(R_A, R_B, 'i j, k j -> i k')
    U_C, S, V_C_T = svd(C)

    U = einsum(Q_A, U_C, 'd r, r s -> d s')
    V = einsum(Q_B, V_C_T, 'e r, s r -> e s')
    return U, S, V

def skew_symmetrize(t):
    return (t - rearrange(t, 'i j -> j i')) / 2

# mutations

MUTATION_REGISTRY = {}

def register_mutation(name: str, fn: callable):
    MUTATION_REGISTRY[name] = fn

# M1
def mutation_svd_structured(
    population: Population,
    idx: int,
    epsilon: float = 0.1,
    **kwargs
):
    device = population.device

    for key in population.weight_down.keys():
        weight_down = population.weight_down[key][idx]
        weight_up = population.weight_up[key][idx]

        U, S, V = _efficient_svd_of_lora(weight_down, weight_up)
        r = S.shape[-1]

        z = torch.randn_like(S)
        S_new = S * torch.exp(epsilon * z)

        M_U = torch.randn((r, r), device = device, dtype = U.dtype)
        R_U = torch.eye(r, device = device, dtype = U.dtype) + epsilon * skew_symmetrize(M_U)

        M_V = torch.randn((r, r), device = device, dtype = V.dtype)
        R_V = torch.eye(r, device = device, dtype = V.dtype) + epsilon * skew_symmetrize(M_V)

        U_new = einsum(U, R_U, 'd r, r s -> d s')
        V_new = einsum(V, R_V, 'e r, r s -> e s')

        S_sqrt = torch.sqrt(S_new)
        weight_down_new = einsum(U_new, S_sqrt, 'd r, r -> d r')
        weight_up_new = einsum(V_new, S_sqrt, 'e r, r -> e r')

        population.weight_down[key][idx].copy_(weight_down_new)
        population.weight_up[key][idx].copy_(weight_up_new)

# M2
def mutation_layer_selective_gaussian(
    population: Population,
    idx: int,
    epsilon: float = 0.1,
    f: float = 0.33,
    **kwargs
):
    device = population.device
    keys = list(population.weight_down.keys())
    num_mutate = max(1, int(f * len(keys)))

    rand_indices = torch.randperm(len(keys), device = device)[:num_mutate].tolist()
    mutate_keys = [keys[i] for i in rand_indices]

    for key in mutate_keys:
        weight_down = population.weight_down[key][idx]
        weight_up = population.weight_up[key][idx]

        weight_down_noise = torch.randn_like(weight_down) * (epsilon * weight_down.std())
        weight_up_noise = torch.randn_like(weight_up) * (epsilon * weight_up.std())

        population.weight_down[key][idx].add_(weight_down_noise)
        population.weight_up[key][idx].add_(weight_up_noise)

# M3
def mutation_component_masking(
    population: Population,
    idx: int,
    rho: float = 0.3,
    **kwargs
):
    device = population.device

    for key in population.weight_down.keys():
        weight_down = population.weight_down[key][idx]
        weight_up = population.weight_up[key][idx]

        U, S, V = _efficient_svd_of_lora(weight_down, weight_up)
        r = S.shape[-1]

        num_drop = int(math.ceil(rho * r))
        drop_indices = torch.randperm(r, device = device)[:num_drop]

        S_new = S.clone()
        S_new[drop_indices] = 0.0

        S_sqrt = torch.sqrt(S_new)
        weight_down_new = einsum(U, S_sqrt, 'd r, r -> d r')
        weight_up_new = einsum(V, S_sqrt, 'e r, r -> e r')

        population.weight_down[key][idx].copy_(weight_down_new)
        population.weight_up[key][idx].copy_(weight_up_new)

# M4
def mutation_full_gaussian(
    population: Population,
    idx: int,
    epsilon: float = 0.15,
    **kwargs
):
    for key in population.weight_down.keys():
        weight_down = population.weight_down[key][idx]
        weight_up = population.weight_up[key][idx]

        weight_down_noise = torch.randn_like(weight_down) * (epsilon * weight_down.std())
        weight_up_noise = torch.randn_like(weight_up) * (epsilon * weight_up.std())

        population.weight_down[key][idx].add_(weight_down_noise)
        population.weight_up[key][idx].add_(weight_up_noise)

# M5
def mutation_neftune_style(
    population: Population,
    idx: int,
    alpha: float = 10.0,
    **kwargs
):
    for key in population.weight_down.keys():
        weight_down = population.weight_down[key][idx]

        numel = weight_down.numel()
        bound = alpha / math.sqrt(numel)
        eta = (torch.rand_like(weight_down) * 2 - 1) * bound

        population.weight_down[key][idx].add_(eta)

register_mutation('svd_structured', mutation_svd_structured)
register_mutation('layer_selective_gaussian', mutation_layer_selective_gaussian)
register_mutation('component_masking', mutation_component_masking)
register_mutation('full_gaussian', mutation_full_gaussian)
register_mutation('neftune_style', mutation_neftune_style)

# main class

class Population(Module):
    def __init__(
        self,
        model: Module,
        *,
        pop_size: int,
        low_rank: int,
        lora_targets: tuple[str, ...] | list[str],
        requires_grad: bool = False,
        mutation_registry: dict | None = None
    ):
        super().__init__()
        self.model = model
        self.pop_size = pop_size
        self.mutation_registry = mutation_registry

        self.weight_down = ParameterDict()
        self.weight_up = ParameterDict()
        self._hooks = []

        for path in lora_targets:
            linear = model.get_submodule(path)
            assert isinstance(linear, Linear), f'{path} must point to a Linear module'

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            self.weight_down[key] = Parameter(torch.empty(pop_size, dim, low_rank), requires_grad = requires_grad)
            self.weight_up[key] = Parameter(torch.empty(pop_size, dim_inner, low_rank), requires_grad = requires_grad)

            init.normal_(self.weight_down[key], std = dim ** -0.5)
            init.normal_(self.weight_up[key], std = low_rank ** -0.5)

            self._hooks.append(linear.register_forward_hook(self._create_hook(key)))

        self._individual = None

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def mutate(
        self,
        mutation_type: str,
        individual: int | None = None,
        individuals: tuple[int, ...] | list[int] | None = None,
        all_individuals: bool = False,
        **kwargs
    ):
        assert sum((exists(individual), exists(individuals), all_individuals)) == 1

        mutation_registry = default(self.mutation_registry, MUTATION_REGISTRY)

        if mutation_type not in mutation_registry:
            raise ValueError(f'Unknown mutation type: {mutation_type}')

        mutation_fn = mutation_registry[mutation_type]

        if all_individuals:
            indices = list(range(self.pop_size))
        elif exists(individuals):
            indices = list(individuals)
        else:
            indices = [individual]

        for idx in indices:
            mutation_fn(self, idx, **kwargs)

    @contextmanager
    def _route(self, individual, individuals, all_individuals):
        assert sum((exists(individual), exists(individuals), all_individuals)) <= 1

        if all_individuals:
            individual = ...
        elif exists(individuals):
            individual = list(individuals)

        self._individual = individual

        try:
            yield
        finally:
            self._individual = None

    @contextmanager
    def _eval_and_no_grad(self, enabled):
        if not enabled:
            yield
            return

        is_training = self.training
        self.eval()

        with torch.no_grad():
            try:
                yield
            finally:
                self.train(is_training)

    def _create_hook(self, lora_key: str):
        def hook(_, args, output):
            if not exists(self._individual) and self._individual is not ...:
                return output

            weight_down, weight_up = self.weight_down[lora_key], self.weight_up[lora_key]
            x, = args

            if isinstance(self._individual, list) or self._individual is ...:
                weight_down_i, weight_up_i = weight_down[self._individual], weight_up[self._individual]
                p = weight_down_i.shape[0]

                x = rearrange(x, '(p b) ... -> p b ...', p = p)
                lora_out = einsum(x, weight_down_i, weight_up_i, 'p b ... d, p d r, p e r -> p b ... e')
                lora_out = rearrange(lora_out, 'p b ... -> (p b) ...')
            else:
                lora_out = einsum(x, weight_down[self._individual], weight_up[self._individual], '... d, d r, e r -> ... e')

            return output + lora_out

        return hook

    def forward(
        self,
        *args,
        individual: int | None = None,
        individuals: tuple[int, ...] | list[int] | None = None,
        all_individuals: bool = False,
        ignore_args_kwargs: tuple[int | str, ...] = tuple(),
        eval_and_no_grad: bool = False,
        **kwargs
    ):
        if all_individuals or exists(individuals):
            ignore = set(ignore_args_kwargs)
            p = self.pop_size if all_individuals else len(individuals)

            def maybe_expand_batch(t):
                assert t.shape[0] in (1, p), f'batch dimension {t.shape[0]} must be equal to 1 or number of individuals {p}'
                return t.expand(p, *t.shape[1:])

            args = tuple(
                tree_map_tensor(maybe_expand_batch, a) if i not in ignore else a
                for i, a in enumerate(args)
            )

            kwargs = {
                k: tree_map_tensor(maybe_expand_batch, v) if k not in ignore else v
                for k, v in kwargs.items()
            }

        with self._route(individual, individuals, all_individuals), self._eval_and_no_grad(eval_and_no_grad):
            return self.model(*args, **kwargs)

class Populations(Module):
    def __init__(
        self,
        *,
        pop_sizes: dict[str, int],
        low_ranks: int | dict[str, int],
        lora_targets: tuple[str, ...] | list[str] | dict[str, tuple[str, ...] | list[str]],
        model: Module | None = None,
        models: dict[str, Module] | None = None,
        requires_grad: bool = False
    ):
        super().__init__()

        models = default(models, {})

        self.populations = ModuleDict({})

        for pop_name, pop_size in pop_sizes.items():
            role_model = models.get(pop_name, model)
            assert exists(role_model), f"no model provided for population {pop_name}"

            self.populations[pop_name] = Population(
                model = role_model,
                pop_size = pop_size,
                low_rank = extract_dict(low_ranks, pop_name),
                lora_targets = extract_dict(lora_targets, pop_name),
                requires_grad = requires_grad
            )

    def forward(self, *args, pop_name: str, **kwargs):
        assert pop_name in self.populations, f"unknown population {pop_name}"
        return self.populations[pop_name](*args, **kwargs)

class PopuLoRA(Module):
    def __init__(
        self,
        *,
        num_teachers: int,
        num_students: int,
        low_rank: int | dict[str, int],
        lora_targets: tuple[str, ...] | list[str] | dict[str, tuple[str, ...] | list[str]],
        model: Module | None = None,
        teacher_model: Module | None = None,
        student_model: Module | None = None,
        requires_grad: bool = False
    ):
        super().__init__()

        models = dict()
        if exists(teacher_model):
            models['teacher'] = teacher_model

        if exists(student_model):
            models['student'] = student_model

        self.populations = Populations(
            pop_sizes = dict(teacher = num_teachers, student = num_students),
            low_ranks = low_rank,
            lora_targets = lora_targets,
            model = model,
            models = models,
            requires_grad = requires_grad
        )

    def forward(self, *args, **kwargs):
        return self.populations(*args, **kwargs)
