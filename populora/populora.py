from __future__ import annotations

import math
import random
from functools import wraps
from contextlib import contextmanager
from collections import namedtuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init
from torch.linalg import qr, svd

from einops import einsum, rearrange
from torch_einops_utils import tree_map_tensor

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
        R_U = torch.eye(r, device = device, dtype = U.dtype).add_(skew_symmetrize(M_U), alpha = epsilon)

        M_V = torch.randn((r, r), device = device, dtype = V.dtype)
        R_V = torch.eye(r, device = device, dtype = V.dtype).add_(skew_symmetrize(M_V), alpha = epsilon)

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
    keys = list(population.weight_down.keys())
    num_mutate = max(1, int(f * len(keys)))

    mutate_keys = random.sample(keys, num_mutate)

    for key in mutate_keys:
        weight_down = population.weight_down[key][idx]
        weight_up = population.weight_up[key][idx]

        weight_down.add_(torch.randn_like(weight_down), alpha = epsilon * weight_down.std())
        weight_up.add_(torch.randn_like(weight_up), alpha = epsilon * weight_up.std())

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

        num_drop = math.ceil(rho * r)
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

        weight_down.add_(torch.randn_like(weight_down), alpha = epsilon * weight_down.std())
        weight_up.add_(torch.randn_like(weight_up), alpha = epsilon * weight_up.std())

# M5
def mutation_neftune_style(
    population: Population,
    idx: int,
    alpha: float = 10.0,
    **kwargs
):
    for key in population.weight_down.keys():
        weight_down = population.weight_down[key][idx]

        bound = alpha / math.sqrt(weight_down.numel())
        noise = torch.rand_like(weight_down).mul_(2).sub_(1)

        weight_down.add_(noise, alpha = bound)

register_mutation('svd_structured', mutation_svd_structured)
register_mutation('layer_selective_gaussian', mutation_layer_selective_gaussian)
register_mutation('component_masking', mutation_component_masking)
register_mutation('full_gaussian', mutation_full_gaussian)
register_mutation('neftune_style', mutation_neftune_style)

# survivor selection

SELECTION_REGISTRY = {}
SelectionResult = namedtuple('SelectionResult', ['survivors', 'culled'])

def register_selection(name: str, fn: callable):
    SELECTION_REGISTRY[name] = fn

def with_elites(select_fn, elite_frac = 0.25):
    if elite_frac == 0.:
        return select_fn

    @wraps(select_fn)
    def inner(fitnesses, num_select, **kwargs):
        device = fitnesses.device
        pop_size = fitnesses.shape[0]
        num_elites = max(1, int(pop_size * elite_frac))

        if num_elites >= num_select:
            return fitnesses.topk(num_select).indices

        elite_indices = fitnesses.topk(num_elites).indices

        mask = torch.ones(pop_size, dtype = torch.bool, device = device)
        mask[elite_indices] = False

        remaining = torch.arange(pop_size, device = device)[mask]
        selected = select_fn(fitnesses[mask], num_select - num_elites, **kwargs)

        return torch.cat((elite_indices, remaining[selected]))
    return inner

def to_centered_ranks(fitnesses):
    ranks = fitnesses.argsort(dim = -1).argsort(dim = -1)
    pop_size = fitnesses.shape[-1]
    # center and scale to [-1, 1]
    return (ranks.float() / max(1, pop_size - 1)) * 2 - 1

def select_deterministic(fitnesses, num_select, **kwargs):
    return fitnesses.topk(num_select).indices

def select_probabilistic(fitnesses, num_select, temperature = 1., **kwargs):
    probs = F.softmax(fitnesses / temperature, dim = -1)
    return torch.multinomial(probs, num_select, replacement = False)

def select_fuss(fitnesses, num_select, eps = 1e-5, **kwargs):
    # fitness uniform selection scheme - Marcus Hutter https://arxiv.org/abs/cs/0103015

    pop_size = fitnesses.shape[0]
    sorted_fitness, sort_indices = fitnesses.sort()

    all_equal = sorted_fitness[0] == sorted_fitness[-1]

    if pop_size == 1 or all_equal:
        return torch.randperm(pop_size, device = fitnesses.device)[:num_select]

    # voronoi cell sizes

    padded = torch.cat((sorted_fitness[:1], sorted_fitness, sorted_fitness[-1:]))
    voronoi_cell_sizes = (padded[2:] - padded[:-2]) / 2

    selected = torch.multinomial(voronoi_cell_sizes + eps, num_select, replacement = False)
    return sort_indices[selected]

register_selection('deterministic', select_deterministic)
register_selection('probabilistic', select_probabilistic)
register_selection('fuss', select_fuss)

# parent selection - todo

# crossover - todo

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
        selection_registry: dict | None = None,
        mutation_registry: dict | None = None
    ):
        super().__init__()
        self.model = model
        self.pop_size = pop_size
        self.selection_registry = selection_registry
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
        assert mutation_type in mutation_registry, f'unknown mutation type {mutation_type}'

        mutation_fn = mutation_registry[mutation_type]

        if all_individuals:
            indices = range(self.pop_size)
        elif exists(individuals):
            indices = individuals
        else:
            indices = (individual,)

        for idx in indices:
            mutation_fn(self, idx, **kwargs)

    @torch.no_grad()
    def select(
        self,
        selection_type: str,
        fitnesses: Tensor,
        survive_frac: float = 0.8,
        elite_frac: float = 0.25,
        use_centered_ranks: bool = False,
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size

        if use_centered_ranks:
            fitnesses = to_centered_ranks(fitnesses)

        selection_registry = default(self.selection_registry, SELECTION_REGISTRY)
        assert selection_type in selection_registry, f'unknown selection type {selection_type}'

        num_survivors = max(1, int(self.pop_size * survive_frac))
        all_indices = torch.arange(self.pop_size, device = self.device)

        if num_survivors >= self.pop_size:
            return SelectionResult(all_indices, all_indices[:0])

        select_fn = with_elites(selection_registry[selection_type], elite_frac)
        survivors = select_fn(fitnesses, num_survivors, **kwargs)

        mask = torch.ones(self.pop_size, dtype = torch.bool, device = self.device)
        mask[survivors] = False

        return SelectionResult(survivors, all_indices[mask])

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
            if self._individual is None:
                return output

            weight_down, weight_up = self.weight_down[lora_key], self.weight_up[lora_key]
            x, = args

            if isinstance(self._individual, (list, tuple)) or self._individual is ...:
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
