from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence
from contextlib import contextmanager

import torch
from torch import Tensor, atleast_1d, cat, is_tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init

from einops import einsum, rearrange, repeat
from torch_einops_utils import batched_index_select, temp_eval, tree_map_tensor, z_score

from populora._utils import cast_tensor, default, divisible_by, exists, extract_dict, first, has_, maybe_cast_tuple, resolve_dtype
from populora.distributed import distributed_device, evaluate_population_distributed, is_distributed
from populora.operators import (
    CROSSOVER_REGISTRY,
    ISLAND_REINIT_REGISTRY,
    MIGRATION_REGISTRY,
    MUTATION_REGISTRY,
    PARENT_SELECTION_REGISTRY,
    SELECTION_REGISTRY,
    SelectionResult,
    with_elites,
)

# helpers

def init_lora_weights(pop_size, dim, dim_inner, rank, device = None, dtype = None):
    # weights are drawn in float32 and cast down - quantization happens once at
    # the storage boundary instead of corrupting the init noise itself

    w_down = torch.empty(pop_size, dim, rank, device = device)
    w_up = torch.empty(pop_size, dim_inner, rank, device = device)

    init.normal_(w_down, std = dim ** -0.5)
    init.normal_(w_up, std = rank ** -0.5)

    if exists(dtype):
        w_down = w_down.to(dtype)
        w_up = w_up.to(dtype)

    return w_down, w_up

def _resolve_fn(type_or_fn, registry, kind):
    # an operator may be a callable, or the name of a registered one

    if callable(type_or_fn):
        return type_or_fn

    assert type_or_fn in registry, f'unknown {kind} type {type_or_fn}'
    return registry[type_or_fn]

def _lora_delta(w_down, w_up):
    return einsum(w_up.float(), w_down.float(), 'e r, d r -> e d')

def _concat_chunked_outputs(outputs, batch_size):
    first_output = outputs[0]

    if is_tensor(first_output):
        return cat(outputs, dim = 0) if first_output.ndim > 0 and first_output.shape[0] == batch_size else first_output

    if isinstance(first_output, (tuple, list)):
        return type(first_output)((_concat_chunked_outputs([output[i] for output in outputs], batch_size) for i in range(len(first_output))))

    return outputs[-1]

def _slice_batch(t, start, end, batch_size):
    return t[start:end] if is_tensor(t) and t.ndim > 0 and t.shape[0] == batch_size else t

def _chunked_forward(model, args, kwargs, ignore, chunk_size, batch_size):
    # run the model over the batch in chunks, concatenating the outputs -
    # caps the activation blowup on large models

    outputs = []

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)

        chunk_args = tuple(
            _slice_batch(a, start, end, batch_size) if i not in ignore else a
            for i, a in enumerate(args)
        )

        chunk_kwargs = {
            key: _slice_batch(value, start, end, batch_size) if key not in ignore else value
            for key, value in kwargs.items()
        }

        outputs.append(model(*chunk_args, **chunk_kwargs))

    return _concat_chunked_outputs(outputs, batch_size)

# shared lora hook machinery

class _LoRAMixin(Module):
    def __init__(self):
        super().__init__()
        self._hooks = []
        self._hooks_registered = False

    @property
    def device(self):
        return next(self.parameters()).device

    def register_hooks(self):
        if self._hooks_registered:
            return

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            self._hooks.append(linear.register_forward_hook(self._create_hook(key)))

        self._hooks_registered = True

    register_hooks_ = register_hooks

    def remove_hooks(self):
        if not self._hooks_registered:
            return

        for hook in self._hooks:
            hook.remove()

        self._hooks.clear()
        self._hooks_registered = False

    remove_hooks_ = remove_hooks

# main class

class Population(_LoRAMixin):
    def __init__(
        self,
        model: Module,
        *,
        pop_size: int,
        low_rank: int,
        lora_targets: Sequence[str],
        requires_grad: bool = False,
        eval_seed: int | None = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
        seed: int | None = None,
        selection_registry: dict | None = None,
        parent_selection_registry: dict | None = None,
        crossover_registry: dict | None = None,
        mutation_registry: dict | None = None,
        migration_registry: dict | None = None,
        island_reinit_registry: dict | None = None
    ):
        super().__init__()
        self.model = model
        self.pop_size = pop_size
        self.selection_registry = selection_registry
        self.parent_selection_registry = parent_selection_registry
        self.crossover_registry = crossover_registry
        self.mutation_registry = mutation_registry
        self.migration_registry = migration_registry
        self.island_reinit_registry = island_reinit_registry

        self._dtype = resolve_dtype(dtype) if exists(dtype) else torch.get_default_dtype()

        self.weight_down = ParameterDict()
        self.weight_up = ParameterDict()

        self.lora_targets = tuple(lora_targets)

        if exists(seed):
            torch.manual_seed(seed)
            random.seed(seed)

        if exists(device):
            device = torch.device(device)
        elif is_distributed():
            device = distributed_device()

        for path in lora_targets:
            linear = model.get_submodule(path)
            assert isinstance(linear, Linear), f'{path} must point to a Linear module'

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            w_down, w_up = init_lora_weights(pop_size, dim, dim_inner, low_rank, device = device, dtype = self._dtype)

            self.weight_down[key] = Parameter(w_down, requires_grad = requires_grad)
            self.weight_up[key] = Parameter(w_up, requires_grad = requires_grad)

        self.register_hooks()
        self._individual = None
        self._eval_seed = eval_seed

        if exists(device):
            self.to(device)

    # save and load

    @torch.no_grad()
    def state_dict_pkg(self, save_base_model = True):
        pkg = dict(
            weight_down = self.weight_down.state_dict(),
            weight_up = self.weight_up.state_dict(),
            pop_size = self.pop_size,
            low_rank = self.low_rank,
            lora_targets = list(self.lora_targets),
            dtype = self._dtype
        )

        if save_base_model:
            pkg['model'] = self.model.state_dict()

        return pkg

    @torch.no_grad()
    def save(self, path: str | Path, save_base_model: bool = True):
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(self.state_dict_pkg(save_base_model = save_base_model), path)
        return self

    @torch.no_grad()
    def load(self, path: str | Path | dict, strict: bool = True):
        pkg = torch.load(path, map_location = self.device, weights_only = False) if not isinstance(path, dict) else path

        if 'model' in pkg:
            self.model.load_state_dict(pkg['model'], strict = strict)

        self.weight_down.load_state_dict(pkg['weight_down'], strict = strict)
        self.weight_up.load_state_dict(pkg['weight_up'], strict = strict)
        return self

    @classmethod
    def from_checkpoint(cls, path: str | Path | dict, model: Module, **kwargs):
        pkg = torch.load(path, weights_only = False) if not isinstance(path, dict) else path

        if 'dtype' not in kwargs and 'dtype' in pkg:
            kwargs = dict(kwargs, dtype = pkg['dtype'])

        pop = cls(
            model = model,
            pop_size = pkg['pop_size'],
            low_rank = pkg['low_rank'],
            lora_targets = pkg['lora_targets'],
            **kwargs
        )
        return pop.load(pkg)

    @torch.no_grad()
    def individual_weights(self, individual = 0):
        # resolved index along with the individual's weight dicts

        if is_tensor(individual):
            individual = individual.item()

        weight_down = {key: self.weight_down[key][individual] for key in self.weight_down.keys()}
        weight_up = {key: self.weight_up[key][individual] for key in self.weight_up.keys()}

        return individual, (weight_down, weight_up)

    @torch.no_grad()
    def save_individual(self, path: str | Path, individual = 0):
        _, (weight_down, weight_up) = self.individual_weights(individual)

        pkg = dict(
            low_rank = self.low_rank,
            lora_targets = list(self.lora_targets),
            weight_down = {key: weight.clone() for key, weight in weight_down.items()},
            weight_up = {key: weight.clone() for key, weight in weight_up.items()}
        )

        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(pkg, path)
        return self

    @torch.no_grad()
    def load_individual(self, path: str | Path | dict, individual = 0, strict: bool = True):
        pkg = torch.load(path, map_location = self.device, weights_only = False) if not isinstance(path, dict) else path

        individual, (weight_down, weight_up) = self.individual_weights(individual)

        for key, w_down in pkg['weight_down'].items():
            if strict:
                assert key in self.weight_down, f'unknown lora target {key}'
                assert w_down.shape == weight_down[key].shape, f'shape mismatch for target {key}'

            weight_down[key].copy_(w_down.to(self.device))
            weight_up[key].copy_(pkg['weight_up'][key].to(self.device))

        return self

    def to_lora(self, individual = 0, requires_grad: bool = True):
        # extract an individual as a standalone trainable adapter, removing this population's hooks
        # so the delta is applied exactly once

        self.remove_hooks()

        _, (weight_down, weight_up) = self.individual_weights(individual)

        return LoRA(
            model = self.model,
            low_rank = self.low_rank,
            lora_targets = self.lora_targets,
            weight_down = weight_down,
            weight_up = weight_up,
            requires_grad = requires_grad
        )

    @property
    def low_rank(self):
        return next(iter(self.weight_down.values())).shape[-1]

    @property
    def eval_seed(self):
        # shared eval seed, auto-synced across ranks - None disables
        return self._eval_seed

    @torch.no_grad()
    def mutate_(
        self,
        mutation_type: str | callable,
        individual: int | None = None,
        individuals: Sequence[int] | Tensor | None = None,
        all_individuals: bool = False,
        ignore_individuals: Sequence[int] | Tensor | None = None,
        **kwargs
    ):
        assert sum((exists(individual), exists(individuals), all_individuals)) == 1

        mutation_registry = default(self.mutation_registry, MUTATION_REGISTRY)
        mutation_fn = _resolve_fn(mutation_type, mutation_registry, 'mutation')

        if all_individuals:
            indices = torch.arange(self.pop_size, device = self.device)
        elif exists(individuals):
            indices = cast_tensor(individuals, device = self.device)
        else:
            indices = cast_tensor((individual,), device = self.device)

        if exists(ignore_individuals):
            ignore = cast_tensor(ignore_individuals, device = self.device)
            indices = indices[~torch.isin(indices, ignore)]

        if len(indices) == 0:
            return self

        # batched mutations get the whole cohort at once - one op per layer
        # instead of one per individual; custom registry fns keep the scalar
        # contract, since they may rely on `weight_down[key][idx]` being a view

        if getattr(mutation_fn, 'batch', False):
            mutation_fn(self, indices, **kwargs)
        else:
            for idx in indices.tolist():
                mutation_fn(self, idx, **kwargs)

        return self

    @torch.no_grad()
    def select(
        self,
        selection_type: str | callable,
        fitnesses: Tensor,
        survive_frac: float = 0.8,
        elite_frac: float = 0.25,
        num_groups: int = 1,
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size
        assert divisible_by(self.pop_size, num_groups)

        selection_registry = default(self.selection_registry, SELECTION_REGISTRY)
        select_fn = _resolve_fn(selection_type, selection_registry, 'selection')

        group_size = self.pop_size // num_groups
        num_survivors = max(1, int(group_size * survive_frac))
        num_elites = max(1, int(group_size * elite_frac)) if elite_frac > 0. else 0
        all_indices = torch.arange(group_size, device = self.device)

        select_fn = with_elites(select_fn, elite_frac)

        # single-group selection fns keep the 1-d fitnesses contract; multi-group
        # selection works on the grouped fitnesses

        fitnesses_grouped = rearrange(fitnesses, '(g p) -> g p', g = num_groups) if num_groups > 1 else fitnesses

        if num_survivors >= group_size:
            survivors = repeat(all_indices, 'p -> g p', g = num_groups)
            culled = repeat(all_indices[:0], 'p -> g p', g = num_groups)

            if num_elites > 0:
                elites = fitnesses_grouped.topk(num_elites, dim = -1).indices
                if num_groups == 1:
                    elites = rearrange(elites, 'e -> 1 e')
            else:
                elites = repeat(all_indices[:0], 'p -> g p', g = num_groups)
        else:
            survivors = select_fn(fitnesses_grouped, num_survivors, **kwargs)

            if num_groups == 1:
                survivors = rearrange(survivors, 's -> 1 s')

            mask = torch.ones(num_groups, group_size, dtype = torch.bool, device = self.device)
            mask.scatter_(-1, survivors, False)

            culled = mask.long().topk(group_size - num_survivors, dim = -1, sorted = False).indices
            elites = survivors[..., :num_elites]

        offset = torch.arange(num_groups, device = fitnesses.device) * group_size

        survivors = survivors + rearrange(offset, 'g -> g 1')
        culled = culled + rearrange(offset, 'g -> g 1')

        if num_elites > 0:
            elites = elites + rearrange(offset, 'g -> g 1')

        return SelectionResult(
            rearrange(survivors, 'g s -> (g s)'),
            rearrange(culled, 'g c -> (g c)'),
            rearrange(elites, 'g e -> (g e)')
        )

    @torch.no_grad()
    def select_parents(
        self,
        selection_type: str | callable,
        fitnesses: Tensor,
        num_children: int,
        num_parents_per_child: int = 2,
        num_groups: int = 1,
        culled: Tensor | Sequence[int] | None = None,
        survivors: Tensor | Sequence[int] | None = None,
        ignore_indices: Tensor | Sequence[int] | None = None,
        **kwargs
    ):
        assert fitnesses.ndim == 1
        assert divisible_by(self.pop_size, num_groups)
        assert divisible_by(num_children, num_groups)

        # unwrap SelectionResult if passed

        if isinstance(culled, SelectionResult):
            culled = culled.culled
        if isinstance(survivors, SelectionResult):
            survivors = survivors.survivors

        # derive eligible parent indices

        eligible_indices = None

        if exists(survivors):
            eligible_indices = cast_tensor(survivors, self.device).flatten()
        elif exists(culled) or exists(ignore_indices):
            to_ignore = []
            if exists(culled):
                to_ignore.append(cast_tensor(culled, self.device).flatten())
            if exists(ignore_indices):
                to_ignore.append(cast_tensor(ignore_indices, self.device).flatten())

            ignored_tensor = cat(to_ignore)
            mask = torch.ones(self.pop_size, dtype = torch.bool, device = self.device)
            mask[ignored_tensor] = False
            eligible_indices = torch.arange(self.pop_size, device = self.device)[mask]

        parent_selection_registry = default(self.parent_selection_registry, PARENT_SELECTION_REGISTRY)
        select_fn = _resolve_fn(selection_type, parent_selection_registry, 'parent selection')

        # single group selection

        if num_groups == 1:
            if exists(eligible_indices):
                fitnesses = fitnesses[eligible_indices] if fitnesses.shape[-1] == self.pop_size else fitnesses
                selected = select_fn(fitnesses, num_children, num_parents_per_child = num_parents_per_child, **kwargs)
                return eligible_indices[selected]

            return select_fn(fitnesses, num_children, num_parents_per_child = num_parents_per_child, **kwargs)

        # island / multi-group selection

        group_size = self.pop_size // num_groups
        fitnesses_grouped = rearrange(fitnesses, '(g p) -> g p', g = num_groups)
        children_per_group = num_children // num_groups

        if exists(eligible_indices):
            mask = torch.zeros(self.pop_size, dtype = torch.bool, device = self.device)
            mask[eligible_indices] = True
            mask_grouped = rearrange(mask, '(g p) -> g p', g = num_groups)

            global_indices_grouped = rearrange(torch.arange(self.pop_size, device = self.device), '(g p) -> g p', g = num_groups)

            eligible_global_indices = rearrange(global_indices_grouped[mask_grouped], '(g p) -> g p', g = num_groups)
            eligible_fitnesses = rearrange(fitnesses_grouped[mask_grouped], '(g p) -> g p', g = num_groups)

            selected = select_fn(eligible_fitnesses, children_per_group, num_parents_per_child = num_parents_per_child, **kwargs)
            parents = batched_index_select(eligible_global_indices, selected, dim = 1)
            return rearrange(parents, 'g c p -> (g c) p')

        parents = select_fn(fitnesses_grouped, children_per_group, num_parents_per_child = num_parents_per_child, **kwargs)

        offset = torch.arange(num_groups, device = fitnesses.device) * group_size
        parents = parents + rearrange(offset, 'g -> g 1 1')

        return rearrange(parents, 'g c p -> (g c) p')

    @torch.no_grad()
    def crossover_(
        self,
        crossover_type: str | callable,
        parent_indices: Tensor,
        child_indices: Tensor,
        fitnesses: Tensor | None = None,
        **kwargs
    ):
        crossover_registry = default(self.crossover_registry, CROSSOVER_REGISTRY)
        crossover_fn = _resolve_fn(crossover_type, crossover_registry, 'crossover')

        if exists(fitnesses):
            kwargs = dict(kwargs, fitnesses = fitnesses)

        crossover_fn(self, parent_indices, child_indices, **kwargs)
        return self

    @torch.no_grad()
    def migrate_(
        self,
        migration_type_or_fn: str | callable,
        fitnesses: Tensor,
        num_islands: int,
        **kwargs
    ):
        assert num_islands > 1, 'migration requires more than one island'
        assert divisible_by(self.pop_size, num_islands), 'pop_size must be divisible by num_islands'

        if isinstance(migration_type_or_fn, str):
            migration_registry = default(self.migration_registry, MIGRATION_REGISTRY)
            assert migration_type_or_fn in migration_registry, f'unknown migration type {migration_type_or_fn}'
            migration_fn = migration_registry[migration_type_or_fn]
        else:
            migration_fn = migration_type_or_fn

        new_arrangement = migration_fn(fitnesses, num_islands, **kwargs)

        for w_down, w_up in zip(self.weight_down.values(), self.weight_up.values()):
            w_down.data.copy_(w_down.data[new_arrangement].clone())
            w_up.data.copy_(w_up.data[new_arrangement].clone())

        return self

    @torch.no_grad()
    def reinit_islands_(
        self,
        reinit_type_or_fn: str | callable,
        islands: int | Sequence[int] | Tensor,
        num_islands: int,
        fitnesses: Tensor | None = None,
        **kwargs
    ):
        assert num_islands > 1, 'num_islands must be > 1'
        assert divisible_by(self.pop_size, num_islands), 'pop_size must be divisible by num_islands'

        if isinstance(reinit_type_or_fn, str):
            reinit_registry = default(self.island_reinit_registry, ISLAND_REINIT_REGISTRY)
            assert reinit_type_or_fn in reinit_registry, f'unknown island reinit type {reinit_type_or_fn}'
            reinit_fn = reinit_registry[reinit_type_or_fn]
        else:
            reinit_fn = reinit_type_or_fn

        if isinstance(islands, int):
            islands = [islands]
        elif isinstance(islands, Tensor):
            islands = islands.tolist()

        for island_idx in islands:
            reinit_fn(
                population = self,
                island_idx = island_idx,
                num_islands = num_islands,
                fitnesses = fitnesses,
                **kwargs
            )

        return self

    @contextmanager
    def _route(self, individual, individuals, all_individuals):
        assert sum((exists(individual), exists(individuals), all_individuals)) <= 1

        if all_individuals:
            individual = ...
        elif exists(individuals):
            individual = cast_tensor(individuals, device = self.device)

        self._individual = individual

        try:
            yield
        finally:
            self._individual = None

    def route(
        self,
        individual = None,
        individuals = None,
        all_individuals = False
    ):
        return self._route(individual, individuals, all_individuals)

    @torch.no_grad()
    def merge_(self, individual = 0):
        if is_tensor(individual):
            individual = individual.item()

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            delta = _lora_delta(self.weight_down[key][individual], self.weight_up[key][individual])
            linear.weight.add_(delta.to(linear.weight.dtype))

        self.remove_hooks()
        return self.model

    merge = merge_

    @torch.no_grad()
    def select_and_merge_best_(self, fitnesses: Tensor):
        best_idx = fitnesses.argmax()
        return self.merge_(best_idx)

    select_and_merge_best = select_and_merge_best_

    @torch.no_grad()
    def select_and_merge_(
        self,
        fitnesses: Tensor | None = None,
        topk: int | float | None = None,
        temperature: float = 1.0,
        use_z_score: bool = True,
        indices: Tensor | Sequence[int] | int | None = None,
        remove_hooks: bool = False
    ):
        indices = maybe_cast_tuple(indices)
        assert exists(fitnesses) or exists(indices), 'either fitnesses or indices must be passed to select_and_merge_'

        if exists(fitnesses):
            assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size

            topk = default(topk, max(1, self.pop_size // 4))

            if isinstance(topk, float):
                topk = max(1, int(self.pop_size * topk))

            topk = min(topk, self.pop_size)

            topk_indices = fitnesses.topk(topk, dim = -1).indices
            topk_fitnesses = fitnesses[topk_indices]
            if use_z_score:
                topk_fitnesses = z_score(topk_fitnesses)
        else:
            topk_indices = torch.tensor(indices, device = self.device) if not isinstance(indices, Tensor) else indices
            topk_indices = atleast_1d(topk_indices)
            topk_fitnesses = torch.ones_like(topk_indices, dtype = torch.float32)

        weights = F.softmax(topk_fitnesses / temperature, dim = -1)

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            w_down_topk = self.weight_down[key][topk_indices].float()
            w_up_topk = self.weight_up[key][topk_indices].float()

            delta = einsum(weights, w_up_topk, w_down_topk, 'k, k e r, k d r -> e d')
            linear.weight.add_(delta.to(linear.weight.dtype))

        if remove_hooks:
            self.remove_hooks()

        return self.model

    select_and_merge = select_and_merge_

    @torch.no_grad()
    def repopulate_(
        self,
        std_down: float | None = None,
        std_up: float | None = None
    ):
        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features
            low_rank = self.weight_down[key].shape[-1]

            std_d = default(std_down, dim ** -0.5)
            std_u = default(std_up, low_rank ** -0.5)

            w_down = torch.empty(self.pop_size, dim, low_rank, device = self.device)
            w_up = torch.empty(self.pop_size, dim_inner, low_rank, device = self.device)
            w_down.normal_(std = std_d)
            w_up.normal_(std = std_u)

            self.weight_down[key].copy_(w_down.to(self._dtype))
            self.weight_up[key].copy_(w_up.to(self._dtype))

    repopulate = repopulate_

    @torch.no_grad()
    def regularize_(
        self,
        weight_decay: float = 0.0,
        soft_threshold: float = 0.0,
        individuals: Sequence[int] | Tensor | None = None
    ):
        if not (has_(weight_decay) or has_(soft_threshold)):
            return self

        for weight in (*self.weight_down.values(), *self.weight_up.values()):
            w = (weight if not exists(individuals) else weight[individuals]).float()

            if has_(weight_decay):
                w.mul_(1. - weight_decay)

            if has_(soft_threshold):
                w.copy_(w.sign() * (w.abs() - soft_threshold).clamp(min = 0.))

            if exists(individuals):
                weight[individuals] = w.to(weight.dtype)
            else:
                weight.copy_(w.to(weight.dtype))

        return self

    regularize = regularize_

    @torch.no_grad()
    def evolve_(
        self,
        fitnesses: Tensor,
        *,
        survive_frac = 0.5,
        elite_frac = 0.10,
        selection_type = 'deterministic',
        parent_selection_type = 'tournament',
        crossover_type = 'average',
        mutation_type = 'full_gaussian',
        num_groups = 1,
        epsilon = 0.1,
        weight_decay = 0.0,
        soft_threshold = 0.0,
        **kwargs
    ):
        result = self.select(
            selection_type,
            fitnesses,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            num_groups = num_groups,
            **kwargs
        )

        parents = self.select_parents(
            parent_selection_type,
            fitnesses,
            num_children = len(result.culled),
            culled = result.culled,
            num_groups = num_groups,
            **kwargs
        )

        self.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses, **kwargs) \
            .mutate_(mutation_type, individuals = result.culled, epsilon = epsilon, **kwargs) \
            .regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

        return result

    evolve = evolve_

    def evaluate_distributed(
        self,
        eval_fn,
        batch_eval = False,
        device: torch.device | str | None = None,
        contiguous = False,
        preserve_rng_state = True,
        shared_seed = True,
        sync_base_model = False,
        **kwargs
    ):
        return evaluate_population_distributed(
            self,
            eval_fn,
            batch_eval = batch_eval,
            device = device,
            contiguous = contiguous,
            preserve_rng_state = preserve_rng_state,
            shared_seed = shared_seed,
            sync_base_model = sync_base_model,
            **kwargs
        )

    evaluate_distributed_ = evaluate_distributed

    @staticmethod
    def adapt_mutation_epsilon(
        epsilon: float,
        success_rate: float,
        target_success_rate: float = 0.20,
        factor: float = 1.15,
        min_epsilon: float = 1e-4,
        max_epsilon: float = 1.0
    ) -> float:

        mult = factor if success_rate > target_success_rate else (1. / factor)
        return float(max(min_epsilon, min(epsilon * mult, max_epsilon)))

    @contextmanager
    def _eval_and_no_grad(self, enabled):
        if not enabled:
            yield
            return

        with temp_eval(self), torch.no_grad():
            yield

    def _create_hook(self, lora_key: str):
        def hook(_, args, output):
            if not exists(self._individual):
                return output

            x = first(args)
            if not exists(x):
                return output

            weight_down, weight_up = self.weight_down[lora_key], self.weight_up[lora_key]
            individual = self._individual

            weight_down_i, weight_up_i = weight_down[individual], weight_up[individual]

            # per-sample routing when ids are vectorized (e.g. one individual per env)

            routed_batch = individual is ... or (is_tensor(individual) and individual.ndim > 0)

            if routed_batch:
                p = weight_down_i.shape[0]

                assert divisible_by(x.shape[0], p), f'batch {x.shape[0]} must be a multiple of routed individuals {p}'

                x = rearrange(x, '(p b) ... -> p b ...', p = p)
                lora_out = einsum(x, weight_down_i.to(x.dtype), weight_up_i.to(x.dtype), 'p b ... d, p d r, p e r -> p b ... e')
                lora_out = rearrange(lora_out, 'p b ... -> (p b) ...')
            else:
                lora_out = einsum(x, weight_down_i.to(x.dtype), weight_up_i.to(x.dtype), '... d, d r, e r -> ... e')

            return output + lora_out.to(output.dtype)

        return hook

    def forward(
        self,
        *args,
        individual: int | None = None,
        individuals: Sequence[int] | None = None,
        all_individuals: bool = False,
        ignore_args_kwargs: Sequence[int | str] = tuple(),
        eval_and_no_grad: bool = False,
        micro_batch: int | None = None,
        **kwargs
    ):
        if not self._hooks_registered:
            self.register_hooks()
        if all_individuals or exists(individuals):
            ignore = set(ignore_args_kwargs)
            p = self.pop_size if all_individuals else len(individuals)
            batch_size = 0

            def maybe_repeat_batch(t):
                # broadcast a singleton batch to each individual, otherwise expect a multiple of the batch

                nonlocal batch_size

                if t.shape[0] == 1:
                    t = repeat(t, '1 ... -> p ...', p = p)
                else:
                    assert divisible_by(t.shape[0], p), f'batch {t.shape[0]} must be a multiple of individuals {p}'

                batch_size = t.shape[0]
                return t

            args = tuple(
                tree_map_tensor(maybe_repeat_batch, a) if i not in ignore else a
                for i, a in enumerate(args)
            )

            kwargs = {
                k: tree_map_tensor(maybe_repeat_batch, v) if k not in ignore else v
                for k, v in kwargs.items()
            }

            # with `micro_batch`, run the expanded batch through the model in
            # chunks, each a multiple of the population size so the per-sample
            # routing stays intact

            if exists(micro_batch) and all_individuals and batch_size > 0:
                assert divisible_by(micro_batch, self.pop_size), f'micro_batch {micro_batch} must be a multiple of the population size {self.pop_size}'

                with self._route(individual, individuals, all_individuals), self._eval_and_no_grad(eval_and_no_grad):
                    return _chunked_forward(self.model, args, kwargs, ignore, micro_batch, batch_size)

        with self._route(individual, individuals, all_individuals), self._eval_and_no_grad(eval_and_no_grad):
            return self.model(*args, **kwargs)

class LoRA(_LoRAMixin):
    def __init__(
        self,
        model: Module,
        *,
        low_rank: int,
        lora_targets: Sequence[str],
        weight_down: dict | None = None,
        weight_up: dict | None = None,
        requires_grad: bool = True,
        device: torch.device | str | None = None
    ):
        super().__init__()
        self.model = model
        self.low_rank = low_rank
        self.lora_targets = tuple(lora_targets)

        self.weight_down = ParameterDict()
        self.weight_up = ParameterDict()

        for path in lora_targets:
            linear = model.get_submodule(path)
            assert isinstance(linear, Linear), f'{path} must point to a Linear module'

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            if exists(weight_down) and key in weight_down:
                w_down = cast_tensor(weight_down[key], device)
                w_up = cast_tensor(weight_up[key], device)

                assert w_down.shape == (dim, low_rank), f'weight_down for {path} must be {dim, low_rank}'
                assert w_up.shape == (dim_inner, low_rank), f'weight_up for {path} must be {dim_inner, low_rank}'
            else:
                w_down, w_up = init_lora_weights(1, dim, dim_inner, low_rank, device = device)
                w_down, w_up = w_down[0], w_up[0]

            self.weight_down[key] = Parameter(w_down.clone(), requires_grad = requires_grad)
            self.weight_up[key] = Parameter(w_up.clone(), requires_grad = requires_grad)

        if exists(device):
            self.to(device)

        self.register_hooks()

    def _create_hook(self, lora_key: str):
        def hook(_, args, output):
            x = first(args)
            if not exists(x):
                return output

            weight_down = self.weight_down[lora_key]
            weight_up = self.weight_up[lora_key]

            lora_out = einsum(x, weight_down.to(x.dtype), weight_up.to(x.dtype), '... d, d r, e r -> ... e')
            return output + lora_out.to(output.dtype)

        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # save and load

    def state_dict_pkg(self):
        return dict(
            low_rank = self.low_rank,
            lora_targets = list(self.lora_targets),
            weight_down = {key: weight.clone() for key, weight in self.weight_down.items()},
            weight_up = {key: weight.clone() for key, weight in self.weight_up.items()}
        )

    @torch.no_grad()
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(self.state_dict_pkg(), path)
        return self

    @torch.no_grad()
    def load(self, path: str | Path | dict, strict: bool = True):
        pkg = torch.load(path, map_location = self.device, weights_only = False) if not isinstance(path, dict) else path
        self.weight_down.load_state_dict(pkg['weight_down'], strict = strict)
        self.weight_up.load_state_dict(pkg['weight_up'], strict = strict)
        return self

    @classmethod
    def from_checkpoint(cls, path: str | Path | dict, model: Module, **kwargs):
        pkg = torch.load(path, weights_only = False) if not isinstance(path, dict) else path
        return cls(
            model = model,
            low_rank = pkg['low_rank'],
            lora_targets = pkg['lora_targets'],
            weight_down = pkg['weight_down'],
            weight_up = pkg['weight_up'],
            **kwargs
        )

    @torch.no_grad()
    def merge_(self, model: Module | None = None):
        model = default(model, self.model)

        for path in self.lora_targets:
            linear = model.get_submodule(path)
            key = path.replace('.', '_')
            delta = _lora_delta(self.weight_down[key], self.weight_up[key])
            linear.weight.add_(delta.to(linear.weight.dtype))

        self.remove_hooks()
        return model

    merge = merge_

class Populations(Module):
    def __init__(
        self,
        *,
        pop_sizes: dict[str, int],
        low_ranks: int | dict[str, int],
        lora_targets: Sequence[str] | dict[str, Sequence[str]],
        model: Module | None = None,
        models: dict[str, Module] | None = None,
        requires_grad: bool = False,
        device: torch.device | str | None = None
    ):
        super().__init__()

        models = default(models, dict())

        self.populations = ModuleDict()

        for pop_name, pop_size in pop_sizes.items():
            role_model = models.get(pop_name, model)
            assert exists(role_model), f"no model provided for population {pop_name}"

            self.populations[pop_name] = Population(
                model = role_model,
                pop_size = pop_size,
                low_rank = extract_dict(low_ranks, pop_name),
                lora_targets = extract_dict(lora_targets, pop_name),
                requires_grad = requires_grad,
                device = device
            )

    # save and load

    @torch.no_grad()
    def save(self, path: str | Path, save_base_model: bool = True):
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        pkg = {pop_name: pop.state_dict_pkg(save_base_model = save_base_model) for pop_name, pop in self.populations.items()}
        torch.save(pkg, path)
        return self

    @torch.no_grad()
    def load(self, path: str | Path | dict, strict: bool = True):
        pkg = torch.load(path, weights_only = False) if not isinstance(path, dict) else path
        for pop_name, pop in self.populations.items():
            if pop_name in pkg:
                pop.load(pkg[pop_name], strict = strict)
        return self

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
        lora_targets: Sequence[str] | dict[str, Sequence[str]],
        model: Module | None = None,
        teacher_model: Module | None = None,
        student_model: Module | None = None,
        requires_grad: bool = False
    ):
        super().__init__()

        models = dict(teacher = teacher_model, student = student_model)
        models = {k: v for k, v in models.items() if exists(v)}

        self.populations = Populations(
            pop_sizes = dict(teacher = num_teachers, student = num_students),
            low_ranks = low_rank,
            lora_targets = lora_targets,
            model = model,
            models = models,
            requires_grad = requires_grad
        )

    # save and load

    def save(self, path: str | Path, save_base_model: bool = True):
        return self.populations.save(path, save_base_model = save_base_model)

    def load(self, path: str | Path | dict, strict: bool = True):
        return self.populations.load(path, strict = strict)

    def forward(self, *args, **kwargs):
        return self.populations(*args, **kwargs)
