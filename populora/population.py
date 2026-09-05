from __future__ import annotations

import math
import random
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor, atleast_1d, cat, is_tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init

from einx import get_at, set_at
from einops import einsum, rearrange, repeat
from torch_einops_utils import batched_index_select, temp_eval, tree_map_tensor, z_score

from populora._utils import cast_tensor, default, divisible_by, exists, extract_dict, first, has_, maybe_cast_tuple, maybe_progress, resolve_dtype, torch_save
from populora.distributed import distributed_device, evaluate_population_distributed, is_distributed
from populora.operators import (
    CROSSOVER_REGISTRY,
    ISLAND_REINIT_REGISTRY,
    MIGRATION_REGISTRY,
    MUTATION_REGISTRY,
    PARENT_SELECTION_REGISTRY,
    SELECTION_REGISTRY,
    TIER_RULE_REGISTRY,
    TieredResult,
    SelectionResult,
    reinit_es,
    with_elites,
)

# helpers

def linear_layer_paths(model: Module) -> list[str]:
    """module paths of every Linear layer, used as `lora_targets` for the population"""
    return [
        path for path, module in model.named_modules()
        if isinstance(module, Linear)
    ]

def init_lora_weights(pop_size, dim, dim_inner, rank, device = None, dtype = None, std_down = None, std_up = None):
    # weights are drawn in float32 and cast down - quantization happens once at
    # the storage boundary instead of corrupting the init noise itself

    std_d = default(std_down, dim ** -0.5)
    std_u = default(std_up, rank ** -0.5)

    w_down = torch.empty(pop_size, dim, rank, device = device)
    w_up = torch.empty(pop_size, dim_inner, rank, device = device)

    init.normal_(w_down, std = std_d)
    init.normal_(w_up, std = std_u)

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

def _adapter_key(path):
    # the storage key for a dotted module path - suffixed with the module name,
    # as keys in weight_down / weight_up / the sigma buffers

    return path.replace('.', '_')

def _merge_adapter(linear, w_down, w_up, individual = None):
    # add one individual's (or an unbroadcast whole) adapter into a Linear's weight
    # `individual` is an integer index - the caller resolves tensors beforehand

    if exists(individual):
        w_down = w_down[individual]
        w_up = w_up[individual]

    delta = _lora_delta(w_down, w_up)
    linear.weight.add_(delta.to(linear.weight.dtype))

def _iter_adapters(population):
    # (path, storage key, w_down, w_up) per lora target, paths dotted as in `lora_targets`

    for path in population.lora_targets:
        key = _adapter_key(path)
        yield path, key, population.weight_down[key], population.weight_up[key]

def _sigma_param(tensor):
    # wrap a step-size buffer as a parameter once - shared buffers ('pop' /
    # 'lora' / 'rank') must keep their identity across the targets they back

    return tensor if isinstance(tensor, Parameter) else Parameter(tensor, requires_grad = False)

def _concat_chunked_outputs(outputs, batch_size):
    first_output = outputs[0]

    if is_tensor(first_output):
        return cat(outputs, dim = 0) if len(outputs) > 1 else first_output

    if isinstance(first_output, (tuple, list)):
        return type(first_output)((_concat_chunked_outputs([output[i] for output in outputs], batch_size) for i in range(len(first_output))))

    return outputs[-1]

def _slice_batch(t, start, end, batch_size):
    return t[start:end] if is_tensor(t) and t.ndim > 0 and t.shape[0] == batch_size else t

def _expand_batch(t, p):
    # broadcast a singleton batch to each individual, otherwise check it is a
    # multiple - non-tensors (lengths, masks, ...) pass through untouched

    if is_tensor(t) and t.ndim > 0:
        if t.shape[0] == 1:
            t = repeat(t, '1 ... -> p ...', p = p)
        assert divisible_by(t.shape[0], p), f'batch {t.shape[0]} must be a multiple of individuals {p}'

    return t

def _at(values, indices):
    return get_at('[p], k -> k', values, indices) if len(indices) > 0 else values[:0]

# per-target operator params - mutation_type / epsilon / crossover_type (or any
# kwarg via PerTarget) may be a dict keyed by lora target or glob; a 'default'
# or '*' entry catches the rest, and every explicit key must match a target

_DEFAULT_SPEC_KEYS = ('default', '*')

def _is_sigma_map(value):
    # a per-target epsilon map (the adaptive-epsilon output - tensors keyed by
    # lora target) vs a per-target spec dict (scalar operator params): only the
    # latter groups targets together, tensor maps pass through to the operator

    return isinstance(value, dict) and len(value) > 0 and all(
        is_tensor(v) or (isinstance(v, (tuple, list)) and all(is_tensor(x) for x in v))
        for v in value.values()
    )

class PerTarget(dict):
    # explicit wrapper letting any operator kwarg vary per lora target -
    # e.g. alpha = PerTarget({'*to_q': 5., 'default': 10.}). mutation_type /
    # epsilon / crossover_type also accept bare dicts

    pass

def _target_spec_assignments(spec, dotted_paths, name):
    # resolve one spec {pattern: value} into one value per target -
    # dotted_paths maps each storage key to its dotted module path

    assert isinstance(spec, dict), f'{name} must be a scalar or a dict keyed by lora target'
    assert len(spec) > 0, f'{name} spec cannot be empty'

    explicit = [(key, value) for key, value in spec.items() if key not in _DEFAULT_SPEC_KEYS]
    fallback = next((value for key, value in spec.items() if key in _DEFAULT_SPEC_KEYS), None)

    def match_one(key):
        dotted = dotted_paths[key]

        for pattern, value in explicit:
            if pattern == key or pattern == dotted:
                return pattern, value

        for pattern, value in explicit:
            if fnmatch(key, pattern) or fnmatch(dotted, pattern):
                return pattern, value

        return None, fallback

    matched = set()
    assignments = dict()

    for key in dotted_paths:
        pattern, value = match_one(key)

        assert exists(value), f'{name} spec leaves target {key!r} uncovered - add a "default" entry'

        if exists(pattern):
            matched.add(pattern)

        assignments[key] = value

    unmatched = [pattern for pattern, _ in explicit if pattern not in matched]
    assert len(unmatched) == 0, f'{name} spec keys {unmatched} match none of {list(dotted_paths)}'

    return assignments

class _TargetView:
    # stand-in population exposing a subset of lora targets - operators written
    # against population.weight_{down,up}.values() transparently run over just
    # their group; every other attribute delegates to the real population

    def __init__(self, population, keys):
        self._population = population
        self.weight_down = {key: population.weight_down[key] for key in keys}
        self.weight_up = {key: population.weight_up[key] for key in keys}

    def __getattr__(self, name):
        return getattr(self._population, name)

# tiered evolve constants

_TIERED_DEFAULT_TIERS = ((0.3, 'keep'), (0.4, 'mutate'), (0.3, 'replace'))  # clone-and-perturb spec
_TIER_STRATA = ('fitness', 'novelty', 'group')

# shared lora hook machinery

class _LoRAMixin(Module):
    def __init__(self):
        super().__init__()
        self._hooks = []
        self._hooks_registered = False
        self._merged = False

    @property
    def device(self):
        return next(self.parameters()).device

    def register_hooks(self):
        if self._hooks_registered:
            return

        # after a merge the delta lives in the base weights - routing through
        # the population again would apply the adapter a second time. any
        # re-anchor (repopulate_ / reinit_individuals_ / load) lifts the guard

        assert not self._merged, 'population was merged into its base model - the lora deltas are baked into the weights, and routing forwards through the population would apply them a second time. re-anchor first (repopulate_ / reinit_individuals_)'

        for path, key, _, _ in _iter_adapters(self):
            linear = self.model.get_submodule(path)
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
        lora_targets: Sequence[str] | None = None,
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
        island_reinit_registry: dict | None = None,
        adaptive_epsilon: bool = False,
        epsilon_init: float = 0.1,
        epsilon_tau: float | None = None,
        epsilon_floor: float = 1e-4,
        epsilon_cap: float = 1.0,
        sigma_granularity: str = 'pop'
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

        lora_targets = default(lora_targets, linear_layer_paths(model))
        assert len(lora_targets) > 0, 'model has no Linear layers to target - pass explicit lora_targets'

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

            key = _adapter_key(path)
            dim, dim_inner = linear.in_features, linear.out_features

            w_down, w_up = init_lora_weights(pop_size, dim, dim_inner, low_rank, device = device, dtype = self._dtype)

            self.weight_down[key] = Parameter(w_down, requires_grad = requires_grad)
            self.weight_up[key] = Parameter(w_up, requires_grad = requires_grad)

        self.register_hooks()
        self._individual = None
        self._eval_seed = eval_seed

        self.register_buffer('_ages', torch.zeros(self.pop_size, dtype = torch.long)) # per-individual age in generations, 0 at birth
        self.register_buffer('_generation', torch.tensor(0, dtype = torch.long))

        # per-individual log-normal step size (Schwefel/Beyer) - inherited by
        # geometric mean, perturbed in place, selected with the weights it
        # shaped; granularity 'pop'/'lora'/'rank'/'weight', tau clamped

        self.adaptive_epsilon = adaptive_epsilon
        self.sigma_granularity = sigma_granularity
        self.epsilon_init = epsilon_init
        self.epsilon_floor = math.log(epsilon_floor)
        self.epsilon_cap = math.log(epsilon_cap)

        if exists(epsilon_tau):
            self.epsilon_tau = epsilon_tau
        else:
            n_params = sum(w.numel() for w in (*self.weight_down.values(), *self.weight_up.values())) // pop_size
            self.epsilon_tau = min(0.3, max(0.05, 1. / (2 * n_params) ** 0.25))

        # 'pop' / 'lora' / 'rank' keep a single tensor per adapter shared between
        # the down and up factor (and, for 'pop', one tensor for every adapter),
        # so one perturb / recombine / reset covers the whole genome

        self._log_sigma_down = ParameterDict()
        self._log_sigma_up = ParameterDict()

        shared_sigma = None

        for key, w_down, w_up in zip(self.weight_down.keys(), self.weight_down.values(), self.weight_up.values()):
            if sigma_granularity == 'pop':
                shared_sigma = default(shared_sigma, _sigma_param(torch.full((self.pop_size, 1, 1), math.log(epsilon_init), device = self.device)))
                sigma_down = sigma_up = shared_sigma
            elif sigma_granularity == 'lora':
                sigma_down = sigma_up = _sigma_param(torch.full((self.pop_size, 1, 1), math.log(epsilon_init), device = self.device))
            elif sigma_granularity == 'rank':
                sigma_down = sigma_up = _sigma_param(torch.full((self.pop_size, 1, w_down.shape[-1]), math.log(epsilon_init), device = self.device))
            elif sigma_granularity == 'weight':
                sigma_down = _sigma_param(torch.full(w_down.shape, math.log(epsilon_init), device = self.device))
                sigma_up = _sigma_param(torch.full(w_up.shape, math.log(epsilon_init), device = self.device))
            else:
                raise ValueError(f'unknown sigma_granularity {sigma_granularity!r} - choose from "pop", "lora", "rank", "weight"')

            self._log_sigma_down[key] = sigma_down
            self._log_sigma_up[key] = sigma_up

        self._twin_pairs = None

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
            dtype = self._dtype,
            ages = self._ages.clone(),
            generation = self.generation,
        )

        if self.adaptive_epsilon:
            pkg['sigma'] = dict(
                down = {key: log_sigma.clone() for key, log_sigma in self._log_sigma_down.items()},
                up = {key: log_sigma.clone() for key, log_sigma in self._log_sigma_up.items()},
            )

        if save_base_model:
            pkg['model'] = self.model.state_dict()

        return pkg

    @torch.no_grad()
    def save(self, path: str | Path, save_base_model: bool = True):
        torch_save(self.state_dict_pkg(save_base_model = save_base_model), path)
        return self

    @torch.no_grad()
    def load(self, path: str | Path | dict, strict: bool = True):
        pkg = torch.load(path, map_location = self.device, weights_only = False) if not isinstance(path, dict) else path

        if 'model' in pkg:
            self.model.load_state_dict(pkg['model'], strict = strict)

        self.weight_down.load_state_dict(pkg['weight_down'], strict = strict)
        self.weight_up.load_state_dict(pkg['weight_up'], strict = strict)

        if self.adaptive_epsilon and 'sigma' in pkg:
            sigma = pkg['sigma']

            # a plain tensor (or flat per-key dict) is a legacy checkpoint of
            # the shared scalar step size - broadcast it into every buffer

            if isinstance(sigma, dict):
                down, up = (sigma['down'], sigma['up']) if 'down' in sigma else (sigma, sigma)
            else:
                down = up = {key: sigma for key in self._log_sigma_down.keys()}

            for key in self._log_sigma_down.keys():
                self._log_sigma_down[key].data.copy_(down[key].to(self.device).broadcast_to(self._log_sigma_down[key].shape))
                self._log_sigma_up[key].data.copy_(up[key].to(self.device).broadcast_to(self._log_sigma_up[key].shape))

        if 'ages' in pkg:
            self._ages.data.copy_(pkg['ages'].to(self.device))

        if 'generation' in pkg:
            self.generation = pkg['generation']

        # a fresh set of adapters re-anchors the population - a prior merge no
        # longer taints routed forwards

        self._merged = False
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

        torch_save(pkg, path)
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

    @torch.no_grad()
    def backup_individual(self, individual = 0):
        individual, (weight_down, weight_up) = self.individual_weights(individual)
        return {
            'down': {key: weight.clone() for key, weight in weight_down.items()},
            'up': {key: weight.clone() for key, weight in weight_up.items()}
        }

    @torch.no_grad()
    def restore_individual(self, individual, backup: dict):
        individual, (weight_down, weight_up) = self.individual_weights(individual)
        for key, w_down in backup['down'].items():
            weight_down[key].copy_(w_down)
        for key, w_up in backup['up'].items():
            weight_up[key].copy_(w_up)
        return self

    def to_lora(self, individual = 0, requires_grad: bool = True):
        # extract an individual as a standalone trainable adapter, removing this population's hooks
        # so the delta is applied exactly once

        self.remove_hooks()
        self._merged = True

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

    @property
    def ages(self):
        return self._ages.clone() # generations each individual has survived

    @property
    def generation(self) -> int:
        return int(self._generation.item())

    @generation.setter
    def generation(self, value: int):
        self._generation.copy_(torch.as_tensor(value, dtype = torch.long, device = self.device))

    def _target_groups(self, knobs, kwargs):
        # group targets by resolved per-target params - a bare dict or PerTarget
        # (non-tensor values) makes that param vary per target; None when nothing
        # varies (scalar fast path), else [(target keys, resolved kwargs)]

        eligible = dict(kwargs)
        eligible.update(knobs)

        specs = [
            (name, value) for name, value in eligible.items()
            if isinstance(value, (dict, PerTarget)) and (name in knobs or isinstance(value, PerTarget)) and not _is_sigma_map(value)
        ]

        if len(specs) == 0:
            return None

        dotted = {_adapter_key(path): path for path in self.lora_targets}
        assignments = [_target_spec_assignments(spec, dotted, name) for name, spec in specs]
        spec_names = [name for name, _ in specs]

        static = {name: value for name, value in kwargs.items() if name not in spec_names}

        signatures = dict()

        for key in self.weight_down.keys():
            signature = tuple(assignment[key] for assignment in assignments)
            signatures.setdefault(signature, []).append(key)

        groups = []

        for signature, keys in signatures.items():
            params = dict(static)
            params.update(zip(spec_names, signature))
            groups.append((keys, params))

        return groups

    @torch.no_grad()
    def mutate_(
        self,
        mutation_type: str | callable | dict,
        individual: int | None = None,
        individuals: Sequence[int] | Tensor | None = None,
        all_individuals: bool = False,
        ignore_individuals: Sequence[int] | Tensor | None = None,
        **kwargs
    ):
        # mutation_type and epsilon may be per-target dicts (see PerTarget) -
        # targets sharing the same resolved params are dispatched together
        # through a filtered view of this population

        assert sum((exists(individual), exists(individuals), all_individuals)) == 1

        mutation_registry = default(self.mutation_registry, MUTATION_REGISTRY)

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

        def run(population, fn, params):
            if getattr(fn, 'batch', False):
                fn(population, indices, **params)
            else:
                for idx in indices.tolist():
                    fn(population, idx, **params)

        eps = kwargs.get('epsilon', None)
        if callable(eps):
            try:
                kwargs['epsilon'] = float(eps(self.generation))
            except TypeError:
                kwargs['epsilon'] = float(eps())

        groups = self._target_groups(dict(mutation_type = mutation_type, epsilon = kwargs.get('epsilon')), kwargs)

        if groups is None:
            run(self, _resolve_fn(mutation_type, mutation_registry, 'mutation'), kwargs)
            return self

        for keys, params in groups:
            fn = _resolve_fn(params.pop('mutation_type', mutation_type), mutation_registry, 'mutation')
            run(_TargetView(self, keys), fn, params)

        return self

    @torch.no_grad()
    def select(
        self,
        selection_type: str | callable,
        fitnesses: Tensor,
        # defaults kept identical to evolve_, so the same truncation pressure
        # applies whichever entry point is used

        survive_frac: float = 0.5,
        elite_frac: float = 0.10,
        num_elites: int | None = None,
        num_groups: int = 1,
        max_age: int | None = None, # hard retirement past a max lifetime - https://arxiv.org/abs/2109.13744
        elite_max_age: int | None = None, # elite-only tenure cap - https://ieeexplore.ieee.org/document/573957
        aging_decay: float | None = None, # fitness discounted by decay ** age - https://doi.org/10.1145/1569901.1570012
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size
        assert divisible_by(self.pop_size, num_groups)

        selection_registry = default(self.selection_registry, SELECTION_REGISTRY)
        select_fn = _resolve_fn(selection_type, selection_registry, 'selection')

        group_size = self.pop_size // num_groups
        num_survivors = max(1, int(group_size * survive_frac))
        if num_elites is None:
            num_elites = max(1, int(group_size * elite_frac)) if elite_frac > 0. else 0
        else:
            num_elites = min(int(num_elites), num_survivors)
        all_indices = torch.arange(group_size, device = self.device)

        if exists(max_age) or exists(elite_max_age) or exists(aging_decay):
            return self._select_aging_(
                select_fn, fitnesses, num_survivors, num_elites, num_groups,
                group_size, max_age, elite_max_age, aging_decay, **kwargs
            )

        if selection_type == 'twin_duel':
            kwargs.setdefault('twin_pairs', getattr(self, '_twin_pairs', None))
        else:
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

            if num_elites > 0:
                top_elites = fitnesses_grouped.topk(num_elites, dim = -1).indices
                if num_groups == 1 and top_elites.ndim == 1:
                    top_elites = rearrange(top_elites, 'e -> 1 e')

                survivor_list = []
                for g in range(num_groups):
                    g_elites = top_elites[g]
                    g_surv = survivors[g]
                    keep = g_surv[~torch.isin(g_surv, g_elites)]
                    merged = torch.cat([g_elites, keep])[:num_survivors]
                    survivor_list.append(merged)
                survivors = torch.stack(survivor_list, dim = 0)

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
    def _select_aging_(
        self,
        select_fn,
        fitnesses: Tensor,
        num_survivors: int,
        num_elites: int,
        num_groups: int,
        group_size: int,
        max_age,
        elite_max_age,
        aging_decay,
        **kwargs
    ):
        # aging-aware selection - the age knobs filter the selection pool

        assert max_age is None or max_age > 0, 'max_age must be positive'
        assert elite_max_age is None or elite_max_age > 0, 'elite_max_age must be positive'
        assert aging_decay is None or 0. < aging_decay <= 1., 'aging_decay must be in (0, 1]'

        ages = self._ages
        device = fitnesses.device

        scores = fitnesses * aging_decay ** ages.float() if exists(aging_decay) else fitnesses

        survivor_lists, culled_lists, elite_lists = [], [], []

        for g in range(num_groups):
            start = g * group_size

            idx = torch.arange(group_size, device = device) + start

            # retirement - prune the old before selection
            if exists(max_age):
                keep = torch.nonzero(~_at(ages >= max_age, idx)).flatten()
                idx = _at(idx, keep)

            # sort best first; elites are the top slice
            order = _at(scores, idx).argsort(descending = True)
            idx = _at(idx, order)

            if exists(elite_max_age):
                ok = torch.nonzero(_at(ages < elite_max_age, idx)).flatten()[:num_elites]
                elite = _at(idx, ok)
            else:
                elite = idx[:num_elites]

            elite = elite[:num_survivors]

            other = torch.nonzero(~torch.isin(idx, elite)).flatten()
            idx_other = _at(idx, other) if len(other) > 0 else idx[:0]

            num_rest = min(num_survivors - len(elite), len(idx_other)) if len(idx_other) > 0 else 0

            if num_rest > 0:
                sel = select_fn(_at(scores, idx_other), num_rest, **kwargs)
                rest = _at(idx_other, sel)
            else:
                rest = idx[:0]

            survivors = cat((elite, rest))
            survivor_lists.append(survivors)
            elite_lists.append(elite)

            # culled - every non-survivor of the group
            all_idx = torch.arange(group_size, device = device) + start
            culled = all_idx[~torch.isin(all_idx, survivors)] if len(survivors) < group_size else all_idx[:0]
            culled_lists.append(culled)

        return SelectionResult(
            cat(survivor_lists),
            cat(culled_lists),
            cat(elite_lists)
        )

    @torch.no_grad()
    def _retire_refill_es_(
        self,
        indices: Tensor,
        fitnesses: Tensor,
        num_groups: int = 1,
        eta: float = 1.0,
        elite_frac: float = 0.25,
        noise_std_min: float = 1e-5
    ):
        # refill retired slots from the island-ES update - delegate per island
        # to the same operator used by island reinitialization, touching only
        # the retired subset

        assert divisible_by(self.pop_size, num_groups)

        if len(indices) == 0:
            return self

        group_size = self.pop_size // num_groups

        for g in range(num_groups):
            island = torch.arange(group_size, device = self.device) + g * group_size
            refill = indices[torch.isin(indices, island)]

            if len(refill) == 0:
                continue

            reinit_es(
                self,
                island_idx = g,
                num_islands = num_groups,
                fitnesses = fitnesses,
                refill = refill,
                eta = eta,
                elite_frac = elite_frac,
                noise_std_min = noise_std_min
            )

            self._sigma_reset_(refill)

        return self

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
        yin_yang: bool = False,
        **kwargs
    ):
        assert fitnesses.ndim == 1
        assert divisible_by(self.pop_size, num_groups)
        assert divisible_by(num_children, num_groups)

        if yin_yang and num_children >= 2:
            num_pairs = num_children // 2
            half_parents = self.select_parents(
                selection_type = selection_type,
                fitnesses = fitnesses,
                num_children = num_pairs,
                num_parents_per_child = num_parents_per_child,
                num_groups = num_groups,
                culled = culled,
                survivors = survivors,
                ignore_indices = ignore_indices,
                yin_yang = False,
                **kwargs
            )
            paired_parents = torch.repeat_interleave(half_parents, 2, dim = 0)
            if num_children > len(paired_parents):
                extra_parents = self.select_parents(
                    selection_type = selection_type,
                    fitnesses = fitnesses,
                    num_children = num_children - len(paired_parents),
                    num_parents_per_child = num_parents_per_child,
                    num_groups = num_groups,
                    culled = culled,
                    survivors = survivors,
                    ignore_indices = ignore_indices,
                    yin_yang = False,
                    **kwargs
                )
                paired_parents = torch.cat([paired_parents, extra_parents], dim = 0)
            return paired_parents

        # unwrap SelectionResult if passed

        if isinstance(culled, SelectionResult):
            culled = culled.culled
        if isinstance(survivors, SelectionResult):
            survivors = survivors.survivors

        # derive eligible parent indices

        eligible_indices = None

        if exists(survivors):
            eligible_indices = cast_tensor(survivors, device = self.device).flatten()
        elif exists(culled) or exists(ignore_indices):
            to_ignore = []
            if exists(culled):
                to_ignore.append(cast_tensor(culled, device = self.device).flatten())
            if exists(ignore_indices):
                to_ignore.append(cast_tensor(ignore_indices, device = self.device).flatten())

            ignored_tensor = cat(to_ignore)
            dd = dict(device = self.device, dtype = torch.bool)
            mask = torch.ones(self.pop_size, **dd)
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
        crossover_type: str | callable | dict,
        parent_indices: Tensor,
        child_indices: Tensor,
        fitnesses: Tensor | None = None,
        **kwargs
    ):
        # crossover_type may be a per-target dict (see PerTarget) - targets
        # sharing the same resolved params are dispatched together through a
        # filtered view of this population

        crossover_registry = default(self.crossover_registry, CROSSOVER_REGISTRY)

        if exists(fitnesses):
            kwargs = dict(kwargs, fitnesses = fitnesses)

        groups = self._target_groups(dict(crossover_type = crossover_type), kwargs)

        if groups is None:
            crossover_fn = _resolve_fn(crossover_type, crossover_registry, 'crossover')
            crossover_fn(self, parent_indices, child_indices, **kwargs)
            return self

        for keys, params in groups:
            crossover_fn = _resolve_fn(params.pop('crossover_type', crossover_type), crossover_registry, 'crossover')
            crossover_fn(_TargetView(self, keys), parent_indices, child_indices, **params)

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

        migration_registry = default(self.migration_registry, MIGRATION_REGISTRY)
        migration_fn = _resolve_fn(migration_type_or_fn, migration_registry, 'migration')

        new_arrangement = migration_fn(fitnesses, num_islands, **kwargs)

        for w_down, w_up in zip(self.weight_down.values(), self.weight_up.values()):
            # advanced indexing already materializes a fresh tensor, so copy_
            # never aliases its source

            w_down.data.copy_(w_down.data[new_arrangement])
            w_up.data.copy_(w_up.data[new_arrangement])

        if self.adaptive_epsilon:
            for log_sigma in self._sigma_tensors():
                log_sigma.data.copy_(log_sigma.data[new_arrangement])

        self._ages.data.copy_(_at(self._ages, new_arrangement))

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

        reinit_registry = default(self.island_reinit_registry, ISLAND_REINIT_REGISTRY)
        reinit_fn = _resolve_fn(reinit_type_or_fn, reinit_registry, 'island reinit')

        if isinstance(islands, int):
            islands = [islands]
        elif isinstance(islands, Tensor):
            islands = islands.tolist()

        island_size = self.pop_size // num_islands

        for island_idx in islands:
            reinit_fn(
                population = self,
                island_idx = island_idx,
                num_islands = num_islands,
                fitnesses = fitnesses,
                **kwargs
            )

            if self.adaptive_epsilon:
                island_indices = torch.arange(island_size, device = self.device) + island_idx * island_size
                self._sigma_reset_(island_indices)

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

        for path, key, w_down, w_up in _iter_adapters(self):
            linear = self.model.get_submodule(path)
            _merge_adapter(linear, w_down, w_up, individual)

        self.remove_hooks()
        self._merged = True
        return self.model

    merge = merge_

    @torch.no_grad()
    def select_and_merge_(
        self,
        fitnesses: Tensor | None = None,
        topk: int | float | None = None,
        temperature: float = 1.0,
        use_z_score: bool = True,
        indices: Tensor | Sequence[int] | int | None = None,
        remove_hooks: bool = False,
        scale: float = 1.0
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

        for path, key, w_down, w_up in _iter_adapters(self):
            linear = self.model.get_submodule(path)
            w_down_topk = w_down[topk_indices].float()
            w_up_topk = w_up[topk_indices].float()

            delta = einsum(weights, w_up_topk, w_down_topk, 'k, k e r, k d r -> e d')
            linear.weight.add_((scale * delta).to(linear.weight.dtype))

        if remove_hooks:
            self.remove_hooks()
            self._merged = True

        return self.model

    select_and_merge = select_and_merge_

    @torch.no_grad()
    def reinit_individuals_(
        self,
        individuals: int | Sequence[int] | Tensor,
        std_down: float | None = None,
        std_up: float | None = None
    ):
        individuals = cast_tensor(individuals, device = self.device)
        self._merged = False

        for path, key, w_down, w_up in _iter_adapters(self):
            linear = self.model.get_submodule(path)
            dim, dim_inner = linear.in_features, linear.out_features
            low_rank = w_down.shape[-1]

            w_down, w_up = init_lora_weights(len(individuals), dim, dim_inner, low_rank, device = self.device, dtype = self._dtype, std_down = std_down, std_up = std_up)

            self.weight_down[key].data[individuals] = w_down
            self.weight_up[key].data[individuals] = w_up

        self._sigma_reset_(individuals)

        self._ages.data.copy_(set_at('[p], k, k -> [p]', self._ages, individuals, torch.zeros_like(individuals)))

        return self

    reinit_individuals = reinit_individuals_

    @torch.no_grad()
    def repopulate_(
        self,
        std_down: float | None = None,
        std_up: float | None = None,
        save_champion: bool = False
    ):
        self.reinit_individuals_(torch.arange(self.pop_size, device = self.device), std_down = std_down, std_up = std_up)

        if save_champion:
            for key in self.weight_up:
                self.weight_up[key].data[0].zero_()

        return self

    repopulate = repopulate_

    @torch.no_grad()
    def merge_champion_(
        self,
        fitnesses: Tensor | None = None,
        individual: int | None = None,
        repopulate: bool = True,
        save_champion: bool = True,
        scale: float = 1.0
    ):
        assert exists(fitnesses) or exists(individual), 'either fitnesses or individual must be passed to merge_champion_'
        champion_idx = fitnesses.argmax().item() if exists(fitnesses) else individual

        self.select_and_merge_(indices = champion_idx, remove_hooks = False, scale = scale)

        if repopulate:
            self.repopulate_(save_champion = save_champion)

        return self

    merge_champion = merge_champion_

    @torch.no_grad()
    def _shared_fitnesses_(
        self,
        fitnesses: Tensor,
        quantile: float = 0.5,
        kernel_power: float = 1.0,
    ):
        """Goldberg & Richardson (1987) fitness sharing over genotype distance.

        Each individual's fitness is divided by its niche count - the number of
        other individuals within a sharing radius of its LoRA genome (scaled by
        a triangular kernel). Dense basins (e.g. every offspring in the same
        stable-shuffle niche) are discounted, so underrepresented niches (e.g.
        a rare but promising flight gait) survive selection and are refined
        instead of being crowded out. `quantile` picks the sharing radius as a
        quantile of all pairwise genome distances, so it adapts to the scale of
        the population without manual tuning."""
        assert 0.0 < quantile <= 1.0, 'sharing quantile must be in (0, 1]'

        n = self.pop_size
        device, dtype = self.device, self._dtype

        # squared LoRA genome distance matrix, summed over all targets

        sq = torch.zeros(n, n, device = device, dtype = torch.float32)

        for key in self.weight_down:
            wd = self.weight_down[key].float()
            wu = self.weight_up[key].float()
            wd_flat = wd.reshape(n, -1)
            wu_flat = wu.reshape(n, -1)

            for w in (wd_flat, wu_flat):
                sq += (w * w).sum(-1, keepdim = True)
                sq -= 2 * w @ w.T
                sq += (w * w).sum(-1, keepdim = True).T

        dist = sq.clamp(min = 0.0).sqrt()

        # sharing radius from pairwise distances, then shared fitness

        triu = dist.masked_select(torch.triu(torch.ones(n, n, dtype = torch.bool, device = device), diagonal = 1))
        radius = triu.quantile(quantile).item() if n > 1 else 1.0
        radius = max(radius, 1e-8)

        overlap = (1.0 - (dist / radius).clamp(max = 1.0)) ** kernel_power
        niche_count = overlap.sum(dim = -1).clamp(min = 1e-6)

        shared = fitnesses.to(device).float() / niche_count
        return shared

    @torch.no_grad()
    def mirror_antithetic_(self):
        """Rearrange the adapter population into antithetic pairs in *weight*
        space: individual 2k and 2k+1 share the same `down` factor while the
        `up` factor is negated, so their LoRA deltas exactly cancel
        (ΔW = U · Dᵀ negates when only one factor flips). This is the
        variance-reduced mirror sampling of Salimans et al. (2017), made
        correct for the bilinear LoRA reparameterization."""
        n = self.pop_size
        assert divisible_by(n, 2), 'pop_size must be even for mirror_antithetic_'

        for key in self.weight_down:
            wd = self.weight_down[key]
            wu = self.weight_up[key]
            wd.data[1::2] = wd[0::2].clone()
            wu.data[1::2] = (-wu[0::2]).clone()

        return self

    @torch.no_grad()
    def evolve_es_(
        self,
        fitnesses: Tensor,
        *,
        mirrored: bool = True,
        lr: float = 1.0,
        weight_decay: float = 0.0,
        soft_threshold: float = 0.0,
        temperature: float = 1.0,
    ):
        """OpenAI-ES style rank-weighted update (Salimans et al. 2017) on the
        LoRA population, merged directly into the base network. Individuals are
        the perturbation directions (fresh noise drawn around the base each
        generation); fitnesses become centered ranks in [-0.5, 0.5] and the
        base takes a step along the fitness-weighted mean delta. When
        `mirrored` is True the population must be arranged in antithetic pairs
        (see mirror_antithetic_) so each pair contributes its fitness
        difference - a low-variance directional estimate.

        The gradient is normalized by the realized population noise scale,
        making the step size scale-free (only `lr` matters)."""
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size

        n = self.pop_size
        f = fitnesses.to(self.device).float()

        # centered ranks in [-0.5, 0.5]
        ranks = f.argsort().argsort().float()
        ranks = ranks / max(n - 1, 1) - 0.5

        if mirrored:
            assert divisible_by(n, 2), 'pop_size must be even when mirrored = True'
            ranks = ranks.view(-1, 2)
            weights = ranks[:, 0] - ranks[:, 1]   # (pairs,) - zero-mean per pair
            selected = torch.arange(0, n, 2, device = self.device)
        else:
            weights = ranks
            selected = torch.arange(n, device = self.device)

        for path, key, w_down, w_up in _iter_adapters(self):
            linear = self.model.get_submodule(path)
            wd = w_down[selected].float()
            wu = w_up[selected].float()

            # realized LoRA deltas of the selected individuals
            delta_all = einsum(wu, wd, 'k e r, k d r -> k e d')   # (k, out, in)
            sigma = delta_all.std() + 1e-8

            grad = einsum(weights, delta_all, 'k, k e d -> e d') / (sigma * n * 0.5)
            linear.weight.add_((lr * grad).to(linear.weight.dtype))

            if has_(weight_decay):
                linear.weight.mul_(1. - weight_decay)

            if has_(soft_threshold):
                w = linear.weight
                w.copy_(w.sign() * (w.abs() - soft_threshold).clamp(min = 0.))

        return self

    evolve_es = evolve_es_

    def rl_finetune_elites_(self, fitnesses, env, **kwargs):
        from populora.rl_finetune import rl_finetune_elites_
        return rl_finetune_elites_(self, fitnesses, env, **kwargs)

    rl_finetune_elites = rl_finetune_elites_

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

    # per-individual mutation step size - `epsilon` becomes a per-individual
    # tensor, or a per-target (down, up) map at granularity finer than 'pop',
    # driven by the log-sigma carried in the genome

    def _sigma_tensors(self):
        # the unique step-size buffers - each logical sigma is perturbed /
        # recombined / reset once, never double-drawing its noise

        seen = set()

        for tensors in (self._log_sigma_down, self._log_sigma_up):
            for log_sigma in tensors.values():
                if id(log_sigma) not in seen:
                    seen.add(id(log_sigma))
                    yield log_sigma

    def _sigma_perturb_(self, indices: Tensor):
        if not self.adaptive_epsilon:
            return self

        for log_sigma in self._sigma_tensors():
            log_sigma.data[indices] = (log_sigma[indices] + self.epsilon_tau * torch.randn_like(log_sigma[indices])).clamp(self.epsilon_floor, self.epsilon_cap)
        return self

    def _sigma_recombine_(self, children: Tensor, parents: Tensor):
        if not self.adaptive_epsilon:
            return self

        # geometric mean of the parents' step sizes, perturbed log-normally -
        # `parents` is (C, P); a single parent (clone crossover) passes its
        # step size straight through. every granularity broadcasts the same way

        for log_sigma in self._sigma_tensors():
            parent_log_sigma = log_sigma[parents].mean(dim = 1)     # (C, ...)
            log_sigma.data[children] = (parent_log_sigma + self.epsilon_tau * torch.randn_like(parent_log_sigma)).clamp(self.epsilon_floor, self.epsilon_cap)
        return self

    def _sigma_epsilon_(self, indices: Tensor) -> dict:
        # per-target (down, up) step-size pair, ready for the mutation operators

        return {
            key: (torch.exp(self._log_sigma_down[key][indices]), torch.exp(self._log_sigma_up[key][indices]))
            for key in self.weight_down.keys()
        }

    def _sigma_reset_(self, indices: Tensor):
        if self.adaptive_epsilon:
            for log_sigma in self._sigma_tensors():
                log_sigma.data[indices] = math.log(self.epsilon_init)
        return self

    @torch.no_grad()
    def evolve_(
        self,
        fitnesses: Tensor,
        *,
        survive_frac = 0.5,
        elite_frac = 0.10,
        num_elites: int | None = None,
        selection_type = 'deterministic',
        parent_selection_type = 'tournament',
        crossover_type = 'average',
        mutation_type = 'full_gaussian',
        yin_yang = False,
        twin_duel: bool | None = None,
        num_groups = 1,
        sharing_quantile: float = 0.0, # > 0: Goldberg-Richardson fitness sharing over LoRA genome distance
        epsilon = 0.1,
        weight_decay = 0.0,
        soft_threshold = 0.0,
        tiered = False,
        tiers = None,
        strata = 'fitness',
        novelty = None,
        burn_in = 0,
        gen = None,
        max_age: int | None = None, # hard retirement past a max lifetime - https://arxiv.org/abs/2109.13744
        elite_max_age: int | None = None, # elite-only tenure cap - https://ieeexplore.ieee.org/document/573957
        aging_decay: float | None = None, # fitness discounted by decay ** age - https://doi.org/10.1145/1569901.1570012
        retire_refill: str | None = None, # refill retired slots: 'crossover' (default), 'reinit', 'es'
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size
        assert retire_refill in (None, 'crossover', 'reinit', 'es'), f'unknown retire_refill {retire_refill!r} - choose from "crossover", "reinit", "es"'
        assert not (exists(retire_refill) and not exists(max_age)), 'retire_refill requires max_age'

        cur_gen = self.generation if gen is None else int(gen)
        if exists(gen):
            self.generation = gen

        if callable(epsilon):
            try:
                epsilon = float(epsilon(cur_gen))
            except TypeError:
                epsilon = float(epsilon())

        tiered = tiered or exists(tiers)

        if tiered:
            res = self._evolve_tiered_(
                fitnesses,
                tiers = tiers,
                strata = strata,
                novelty = novelty,
                parent_selection_type = parent_selection_type,
                crossover_type = crossover_type,
                mutation_type = mutation_type,
                num_groups = num_groups,
                epsilon = epsilon,
                burn_in = burn_in,
                gen = cur_gen,
                max_age = max_age,
                elite_max_age = elite_max_age,
                aging_decay = aging_decay,
                **kwargs
            )
            self._generation.add_(1)
            return res

        twin_duel = default(twin_duel, yin_yang)
        selection_type = 'twin_duel' if twin_duel else selection_type

        # Goldberg-Richardson fitness sharing: selection and parent choice see
        # fitness discounted by niche density, so crowded basins cannot crowd
        # out underrepresented ones. The raw champion is always retained.

        if sharing_quantile > 0.:
            shared_fitnesses = self._shared_fitnesses_(fitnesses, quantile = sharing_quantile)
        else:
            shared_fitnesses = fitnesses

        raw_fitnesses = fitnesses

        result = self.select(
            selection_type,
            shared_fitnesses,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            num_elites = num_elites,
            num_groups = num_groups,
            max_age = max_age,
            elite_max_age = elite_max_age,
            aging_decay = aging_decay,
            **kwargs
        )

        if sharing_quantile > 0.:
            # strict elitism on the raw fitness: the true champion always survives
            best_raw = int(raw_fitnesses.argmax().item())
            survivors = result.survivors
            culled = result.culled

            if best_raw not in survivors.tolist():
                drop = shared_fitnesses[survivors].argmin().item()
                dropped = int(survivors[drop].item())
                survivors = torch.cat((survivors[:drop], survivors[drop + 1:], torch.tensor([best_raw], device = self.device)))
                culled = culled[culled != best_raw]
                culled = torch.cat((culled, torch.tensor([dropped], device = self.device)))
                result = SelectionResult(survivors, culled, result.elites)

        # age-forced culls, captured before bookkeeping for later refill

        retired = None
        if exists(max_age):
            retired = torch.nonzero(self._ages >= max_age).flatten()

        # survivors age a generation, the culled are reborn

        ages = set_at('[p], k, k -> [p]', self._ages, result.culled, torch.zeros_like(result.culled))
        ages = set_at('[p], k, k -> [p]', ages, result.survivors, _at(ages, result.survivors) + 1)
        self._ages.data.copy_(ages)

        if yin_yang:
            mutation_type = 'yin_yang'

        parents = self.select_parents(
            parent_selection_type,
            fitnesses,
            num_children = len(result.culled),
            culled = result.culled,
            num_groups = num_groups,
            yin_yang = yin_yang,
            **kwargs
        )

        if self.adaptive_epsilon:
            self._sigma_recombine_(result.culled, parents)
            epsilon = self._sigma_epsilon_(result.culled)

        self.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses, **kwargs)

        if yin_yang and len(result.culled) >= 2:
            num_pairs = len(result.culled) // 2
            yang_culled = result.culled[:num_pairs * 2:2]
            yin_culled = result.culled[1:num_pairs * 2:2]
            self._twin_pairs = (yang_culled.clone(), yin_culled.clone())
            for key in self.weight_down:
                self.weight_down[key].data[yin_culled] = self.weight_down[key].data[yang_culled]
                self.weight_up[key].data[yin_culled] = self.weight_up[key].data[yang_culled]
            if self.adaptive_epsilon:
                for log_sigma in self._sigma_tensors():
                    log_sigma.data[yin_culled] = log_sigma.data[yang_culled]
                epsilon = self._sigma_epsilon_(result.culled)
        else:
            self._twin_pairs = None

        self.mutate_(mutation_type, individuals = result.culled, epsilon = epsilon, **kwargs) \
            .regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

        if retire_refill is not None and exists(retired) and len(retired) > 0:
            # 'crossover' is the default path - retired slots were already culled
            # by select above, so no further handling happens here. only the
            # alternate refill schemes act on the retired indices

            if retire_refill == 'reinit':
                self.reinit_individuals_(retired)
            elif retire_refill == 'es':
                self._retire_refill_es_(retired, fitnesses, num_groups = num_groups)

        self._generation.add_(1)
        return result

    evolve = evolve_

    @torch.no_grad()
    def _tier_bins(self, axis, tiers, num_groups):
        # quantile strata of the axis, per group, best first - the last tier
        # absorbs the rounding remainder

        group_size = self.pop_size // num_groups
        axis_grouped = rearrange(axis, '(g p) -> g p', g = num_groups)

        bins = [list() for _ in tiers]

        for g in range(num_groups):
            order = axis_grouped[g].argsort(descending = True)
            offset = g * group_size
            cum = 0

            for i, (frac, _) in enumerate(tiers):
                count = group_size - cum if i == len(tiers) - 1 else min(max(1, int(frac * group_size)), group_size - cum)
                bins[i].append(order[:count] + offset)
                order = order[count:]
                cum += count

        return [cat(b) for b in bins]

    @torch.no_grad()
    def _evolve_tiered_(
        self,
        fitnesses: Tensor,
        *,
        tiers = None,
        strata = 'fitness',
        novelty = None,
        parent_selection_type = 'tournament',
        crossover_type = 'average',
        mutation_type = 'full_gaussian',
        num_groups = 1,
        epsilon = 0.1,
        burn_in: int = 0,
        gen: int | None = None,
        max_age: int | None = None,
        elite_max_age: int | None = None,
        aging_decay: float | None = None,
        **kwargs
    ):
        # tiered evolve - quantile strata of an axis (fitness / novelty / group),
        # each tier with a rule: keep / mutate / replace / crossover / reinit /
        # archive. `burn_in` skips recently-touched individuals (needs `gen`)

        tiers = default(tiers, _TIERED_DEFAULT_TIERS)

        assert all(frac >= 0. and exists(rule) for frac, rule in tiers)
        assert sum(frac for frac, _ in tiers) <= 1. + 1e-6, 'tier fractions must sum to at most 1'
        assert all(rule in TIER_RULE_REGISTRY for _, rule in tiers), f'unknown tier rule - choose from {tuple(TIER_RULE_REGISTRY)}'
        assert strata in _TIER_STRATA, f'unknown strata {strata} - choose from {_TIER_STRATA}'
        assert not (strata == 'novelty' and not exists(novelty)), 'strata = "novelty" requires a novelty tensor'
        assert divisible_by(self.pop_size, num_groups)
        assert not (burn_in > 0 and not exists(gen)), 'gen must be passed when burn_in > 0'

        # stratum axis - 'group' is per-island fitness, which the binning does anyway

        axis = novelty if strata == 'novelty' else fitnesses

        if exists(aging_decay) and strata == 'fitness':
            assert 0. < aging_decay <= 1., 'aging_decay must be in (0, 1]'
            axis = axis * aging_decay ** self._ages.float()

        tier_bins = self._tier_bins(axis, tiers, num_groups)
        rules = [TIER_RULE_REGISTRY[rule] for _, rule in tiers]

        if burn_in > 0:
            # pause - individuals touched too recently to have adapted are exempt
            # until gen + burn_in (the top tier is never touched)

            last_mutated = getattr(self, '_tiered_last_mutated', None)

            if not exists(last_mutated) or last_mutated.shape[0] != self.pop_size:
                self._tiered_last_mutated = torch.full((self.pop_size,), -burn_in, dtype = torch.long, device = self.device)
                last_mutated = self._tiered_last_mutated

            eligible = (gen - last_mutated) >= burn_in

            for i in range(1, len(tier_bins)):
                tier_bins[i] = tier_bins[i][eligible[tier_bins[i]]]

        # aging: the elite cap demotes the aged, retirement rotates them into the replace tier

        if exists(elite_max_age):
            keep = tier_bins[0]
            aged = get_at('[p], k -> k', self._ages, keep) >= elite_max_age
            tier_bins[0] = keep[~aged]
            tier_bins[1] = cat((tier_bins[1], keep[aged]))

        if exists(max_age):
            aged = self._ages >= max_age
            retire = cat([tier_bins[i][aged[tier_bins[i]]] for i in range(len(tier_bins) - 1)])
            tier_bins = [tier_bins[i][~aged[tier_bins[i]]] for i in range(len(tier_bins) - 1)] + [cat((tier_bins[-1], retire))]

        top = tier_bins[0] # replace tier draws from the highest remaining tier
        sources = torch.empty(0, dtype = torch.long, device = self.device)

        rule_kwargs = dict(
            mutation_type = mutation_type,
            epsilon = epsilon,
            parent_selection_type = parent_selection_type,
            crossover_type = crossover_type,
            num_groups = num_groups,
            **kwargs
        )

        for i, (indices, rule) in enumerate(zip(tier_bins, rules)):
            if i > 0:
                sources = cat((sources, tier_bins[i - 1]))

            rule(self, indices, sources = sources, top = top, fitnesses = fitnesses, **rule_kwargs)

            if burn_in > 0 and i > 0:
                self._tiered_last_mutated[indices] = gen

        surviving = cat(tier_bins[:-1])
        ages = set_at('[p], k, k -> [p]', self._ages, tier_bins[-1], torch.zeros_like(tier_bins[-1]))
        ages = set_at('[p], k, k -> [p]', ages, surviving, _at(ages, surviving) + 1)
        self._ages.data.copy_(ages)

        return TieredResult(tier_bins)
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

        # toggling train/eval walks the whole module tree with per-module
        # setattrs - skipped when the model tree is already fully in eval
        # mode, which is the rollout / decode steady state

        if self.model.training:
            with temp_eval(self), torch.no_grad():
                yield
            return

        with torch.no_grad():
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

            if individual is ... or (is_tensor(individual) and individual.ndim > 0):
                p = weight_down_i.shape[0]

                # a 1-d activation is one unbatched feature vector, which cannot be
                # split across routed individuals - anything else is a guess

                assert x.ndim >= 2, f'routed forwards need batched inputs - got a {x.ndim}-d tensor of shape {tuple(x.shape)}. pass observations with their batch / feature axes'

                assert divisible_by(x.shape[0], p), f'batch {x.shape[0]} must be a multiple of routed individuals {p}'

                wd_i = weight_down_i.to(x.dtype)
                wu_i = weight_up_i.to(x.dtype)

                if x.shape[0] == p and x.ndim == 2:
                    # one sample per individual - the delta is two chained
                    # matmuls on a leading batch axis, no reshaping at all
                    lora_out = (x.unsqueeze(1) @ wd_i @ wu_i.transpose(-1, -2)).squeeze(1)
                else:
                    x = rearrange(x, '(p b) ... -> p b ...', p = p)
                    lora_out = einsum(x, wd_i, wu_i, 'p b ... d, p d r, p e r -> p b ... e')
                    lora_out = rearrange(lora_out, 'p b ... -> (p b) ...')
            else:
                lora_out = x @ weight_down_i.to(x.dtype) @ weight_up_i.to(x.dtype).transpose(-1, -2)

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

            if not ignore and not any(isinstance(t, (tuple, list, dict)) for t in (*args, *kwargs.values())):
                # fast path - plain tensor args, no per-call structure walk
                args = tuple(_expand_batch(a, p) for a in args)
                kwargs = {k: _expand_batch(v, p) for k, v in kwargs.items()}
                batch_size = next((t.shape[0] for t in (*args, *kwargs.values()) if is_tensor(t) and t.ndim > 0), 0)
            else:
                def maybe_repeat_batch(t):
                    # broadcast singleton batches to each individual, otherwise expect a
                    # multiple of the batch - `_expand_batch` skips non-tensor leaves

                    nonlocal batch_size

                    t = _expand_batch(t, p)

                    if is_tensor(t) and t.ndim > 0:
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

            # with `micro_batch`, chunk the expanded batch - each chunk is routed
            # by its explicit individual ids, so any chunk boundary is safe

            if exists(micro_batch) and all_individuals and batch_size > 0:
                per_indiv = batch_size // self.pop_size

                route_rows = repeat(
                    torch.arange(self.pop_size, device = self.device),
                    'p -> (p b)',
                    b = per_indiv
                )

                outputs = []

                with self._eval_and_no_grad(eval_and_no_grad):
                    for start in range(0, batch_size, micro_batch):
                        end = min(start + micro_batch, batch_size)

                        chunk_args = tuple(
                            _slice_batch(a, start, end, batch_size) if i not in ignore else a
                            for i, a in enumerate(args)
                        )

                        chunk_kwargs = {
                            key: _slice_batch(value, start, end, batch_size) if key not in ignore else value
                            for key, value in kwargs.items()
                        }

                        with self._route(None, route_rows[start:end], False):
                            outputs.append(self.model(*chunk_args, **chunk_kwargs))

                return _concat_chunked_outputs(outputs, batch_size)

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

            key = _adapter_key(path)
            dim, dim_inner = linear.in_features, linear.out_features

            # a supplied checkpoint must cover every target - silently random
            # initializing the missing ones would yield a quietly wrong adapter

            if exists(weight_down) or exists(weight_up):
                assert exists(weight_down) and exists(weight_up), 'weight_down and weight_up must be provided together'
                assert key in weight_down and key in weight_up, f'missing lora weights for target {path} in the provided checkpoint'

                w_down = cast_tensor(weight_down[key], device = device)
                w_up = cast_tensor(weight_up[key], device = device)

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

            lora_out = x @ weight_down.to(x.dtype) @ weight_up.to(x.dtype).transpose(-1, -2)
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
        torch_save(self.state_dict_pkg(), path)
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

        for path, key, w_down, w_up in _iter_adapters(self):
            linear = model.get_submodule(path)
            _merge_adapter(linear, w_down, w_up)

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
        device: torch.device | str | None = None,
        seed: int | None = None
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
                device = device,
                seed = seed
            )

    # save and load

    @torch.no_grad()
    def save(self, path: str | Path, save_base_model: bool = True):
        pkg = {pop_name: pop.state_dict_pkg(save_base_model = save_base_model) for pop_name, pop in self.populations.items()}
        torch_save(pkg, path)
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
        requires_grad: bool = False,
        seed: int | None = None
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
            requires_grad = requires_grad,
            seed = seed
        )

    # save and load

    def save(self, path: str | Path, save_base_model: bool = True):
        return self.populations.save(path, save_base_model = save_base_model)

    def load(self, path: str | Path | dict, strict: bool = True):
        return self.populations.load(path, strict = strict)

    def forward(self, *args, **kwargs):
        return self.populations(*args, **kwargs)

# module-level evolve - the generation loop for any task that scores the
# population with a fitness function, the non-env sibling of `evolve_with_env`

def _generation_loop(
    population: Population,
    evaluate: Callable,
    *,
    num_generations: int,
    target_fitness: float | None = None,
    patience: int = 1,
    progress: bool = False,
    start_generation: int = 0,
    initial_state: tuple | None = None,
    on_generation: Callable | None = None,
    adaptive_epsilon: bool = False,
    target_success_rate: float = 0.20,
    epsilon_factor: float = 1.15,
    **evolve_kwargs
):
    """the shared generation driver of `evolve` and `EnvInteractor.evolve` -
    evaluate the population, record best / mean, evolve it, and stop once
    `target_fitness` has been reached for `patience` consecutive generations.

    `evaluate` is a zero-arg callable returning one fitness per individual;
    history entries and best tracking stay in the same schema across both
    entry points. `start_generation` / `initial_state` resume an interrupted
    run (generation counter plus (best_fitness, best_index, history)), and
    `on_generation(generation, best_fitness, best_index, history, is_best)`
    fires after each evolve_, e.g. for checkpointing. returns
    (best_fitness, best_index, history)."""

    assert num_generations >= 1, 'num_generations must be at least 1'
    assert patience >= 1, 'patience must be at least 1'

    best_fitness, best_index, history = default(initial_state, (float('-inf'), 0, []))
    streak = 0
    epsilon = evolve_kwargs.get('epsilon', 0.2)
    smoothed_success = None
    prev_median = None

    for generation in maybe_progress(range(start_generation, num_generations), progress, 'evolving'):
        fitnesses = evaluate().detach()

        gen_best_fitness = float(fitnesses.max())
        is_best = gen_best_fitness > best_fitness

        if is_best:
            best_fitness = gen_best_fitness
            best_index = int(fitnesses.argmax())

        history.append(dict(
            best_fitness = gen_best_fitness,
            mean_fitness = float(fitnesses.mean()),
        ))

        # Rechenberg 1/5 rule with exponential smoothing - adapts epsilon
        # based on the fraction of individuals beating the prior median
        if adaptive_epsilon and prev_median is not None:
            success = float((fitnesses >= prev_median).float().mean())
            smoothed_success = success if smoothed_success is None else 0.30 * success + 0.70 * smoothed_success

            epsilon = Population.adapt_mutation_epsilon(
                epsilon, smoothed_success, target_success_rate = target_success_rate, factor = epsilon_factor
            )
            evolve_kwargs['epsilon'] = epsilon

        if exists(target_fitness):
            streak = streak + 1 if gen_best_fitness >= target_fitness else 0

            if streak >= patience:
                break

        population.evolve_(fitnesses, **evolve_kwargs)
        prev_median = float(fitnesses.median())

        if exists(on_generation):
            on_generation(generation, best_fitness, best_index, history, is_best)

    return best_fitness, best_index, history

def evolve(
    population: Population,
    fitness_fn: Callable,
    *,
    num_generations: int,
    target_fitness: float | None = None,
    patience: int = 1,
    progress: bool = False,
    return_history: bool = False,
    **evolve_kwargs
):
    """evolve a population against any fitness function, in one call.

    `fitness_fn` receives the population and returns one fitness per individual
    (a tensor, or anything tensor-able). each generation it is evaluated, best /
    mean are recorded, and `population.evolve_` is run with `evolve_kwargs`.
    evolution stops once `target_fitness` has been reached for `patience`
    consecutive generations, and the best individual is merged back into the
    base model. returns the merged policy, plus the history (per-generation
    best / mean) with `return_history = True`."""

    assert callable(fitness_fn), 'fitness_fn must be a callable taking the population and returning one fitness per individual'

    def evaluate_gen():
        dd = dict(device = population.device, dtype = torch.float32)
        fitnesses = cast_tensor(fitness_fn(population), **dd)
        assert fitnesses.shape == (population.pop_size,), f'fitness_fn must return one fitness per individual, got {tuple(fitnesses.shape)}'
        return fitnesses

    _, best_index, history = _generation_loop(
        population,
        evaluate_gen,
        num_generations = num_generations,
        target_fitness = target_fitness,
        patience = patience,
        progress = progress,
        **evolve_kwargs
    )

    policy = population.merge_(best_index)

    if return_history:
        return policy, history

    return policy
