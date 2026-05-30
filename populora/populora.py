from __future__ import annotations

import torch
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init

from einops import einsum, repeat, rearrange
from torch_einops_utils import tree_map_tensor

from contextlib import contextmanager

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def extract_dict(v, k):
    return v[k] if isinstance(v, dict) else v

# evolution

# selection

def select(population):
    raise NotImplementedError

# mutation

def mutation(population):
    raise NotImplementedError

# crossover

def crossover(*parents):
    raise NotImplementedError

# main class

class Population(Module):
    def __init__(
        self,
        model: Module,
        *,
        pop_size: int,
        low_rank: int,
        lora_targets: tuple[str, ...] | list[str],
        requires_grad: bool = False
    ):
        super().__init__()
        self.model = model
        self.pop_size = pop_size

        self.w_down = ParameterDict()
        self.w_up = ParameterDict()
        self._hooks = []

        for path in lora_targets:
            linear = model.get_submodule(path)
            assert isinstance(linear, Linear), f'{path} must point to a Linear module'

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            self.w_down[key] = Parameter(torch.empty(pop_size, dim, low_rank), requires_grad = requires_grad)
            self.w_up[key] = Parameter(torch.empty(pop_size, dim_inner, low_rank), requires_grad = requires_grad)

            init.normal_(self.w_down[key], std = dim ** -0.5)
            init.normal_(self.w_up[key], std = low_rank ** -0.5)

            self._hooks.append(linear.register_forward_hook(self._create_hook(key)))

        self._individual = None

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

            w_down, w_up = self.w_down[lora_key], self.w_up[lora_key]
            x, = args

            if isinstance(self._individual, list) or self._individual is ...:
                w_down_i, w_up_i = w_down[self._individual], w_up[self._individual]
                p = w_down_i.shape[0]

                x = rearrange(x, '(p b) ... -> p b ...', p = p)
                lora_out = einsum(x, w_down_i, w_up_i, 'p b ... d, p d r, p e r -> p b ... e')
                lora_out = rearrange(lora_out, 'p b ... -> (p b) ...')
            else:
                lora_out = einsum(x, w_down[self._individual], w_up[self._individual], '... d, d r, e r -> ... e')

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
