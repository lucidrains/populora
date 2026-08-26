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

from einops import einsum, rearrange, repeat
from torch_einops_utils import batched_index_select, temp_eval, tree_map_tensor, z_score

from populora._utils import cast_tensor, default, divisible_by, exists, extract_dict, first, has_, maybe_cast_tuple, maybe_progress, resolve_dtype
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
    with_elites,
)

# helpers

def linear_layer_paths(model: Module) -> list[str]:
    """module paths of every Linear layer, used as `lora_targets` for the population"""
    return [
        path for path, module in model.named_modules()
        if isinstance(module, Linear)
    ]

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

# per-target operator params - mutation_type, epsilon, and crossover_type (and,
# via PerTarget, any operator kwarg) may be given as a dict keyed by lora
# target instead of a scalar. keys match a target's dotted module path or its
# storage key exactly, else by glob pattern ('*' wildcards, first match wins in
# insertion order); a 'default' or '*' entry catches everything left. every
# explicit key must match at least one target, and coverage must be total

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

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            w_down, w_up = init_lora_weights(pop_size, dim, dim_inner, low_rank, device = device, dtype = self._dtype)

            self.weight_down[key] = Parameter(w_down, requires_grad = requires_grad)
            self.weight_up[key] = Parameter(w_up, requires_grad = requires_grad)

        self.register_hooks()
        self._individual = None
        self._eval_seed = eval_seed

        # per-individual mutation step size (log-normal self-adaptation,
        # Schwefel 1981 / Beyer 2001): each individual carries its own mutation
        # strength in log space, inherited from its parents (geometric mean,
        # `_sigma_recombine_`) or perturbed in place (`_sigma_perturb_`), and the
        # perturbed value is what mutates its offspring's weights - so selection
        # tunes the mutation rate itself, no hand-set schedule. the log-sigma
        # lives in one buffer per lora target (down and up factor), each shaped
        # so it broadcasts against that target's weights; `sigma_granularity`
        # picks the finest structure the step size is tracked at:
        #
        #   'pop'    one per individual, shared across the whole genome
        #   'lora'   one per individual per LoRA adapter (down/up pair share it)
        #   'rank'   one per individual per singular-value direction
        #   'weight' one per individual per parameter (every element of every
        #            LoRA matrix) - roughly doubles the population's storage
        #
        # default tau is the textbook 1 / sqrt(2 * sqrt(n_params)), clamped to
        # stay responsive on small networks

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

    def _target_groups(self, knobs, kwargs):
        # group lora targets by resolved per-target params - among `knobs`
        # (the canonical params accepting bare dicts, layered over kwargs) and
        # kwargs itself, any bare dict or PerTarget-wrapped value makes that
        # param vary per lora target. returns None when nothing varies (scalar
        # fast path), else [(target keys, resolved kwargs)] per group

        eligible = dict(kwargs)
        eligible.update(knobs)

        specs = [
            (name, value) for name, value in eligible.items()
            if isinstance(value, (dict, PerTarget)) and (name in knobs or isinstance(value, PerTarget)) and not _is_sigma_map(value)
        ]

        if len(specs) == 0:
            return None

        dotted = {path.replace('.', '_'): path for path in self.lora_targets}
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

        if isinstance(migration_type_or_fn, str):
            migration_registry = default(self.migration_registry, MIGRATION_REGISTRY)
            assert migration_type_or_fn in migration_registry, f'unknown migration type {migration_type_or_fn}'
            migration_fn = migration_registry[migration_type_or_fn]
        else:
            migration_fn = migration_type_or_fn

        new_arrangement = migration_fn(fitnesses, num_islands, **kwargs)

        for w_down, w_up in zip(self.weight_down.values(), self.weight_up.values()):
            # advanced indexing already materializes a fresh tensor, so copy_
            # never aliases its source

            w_down.data.copy_(w_down.data[new_arrangement])
            w_up.data.copy_(w_up.data[new_arrangement])

        if self.adaptive_epsilon:
            for log_sigma in self._sigma_tensors():
                log_sigma.data.copy_(log_sigma.data[new_arrangement])

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

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            delta = _lora_delta(self.weight_down[key][individual], self.weight_up[key][individual])
            linear.weight.add_(delta.to(linear.weight.dtype))

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

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features
            low_rank = self.weight_down[key].shape[-1]

            std_d = default(std_down, dim ** -0.5)
            std_u = default(std_up, low_rank ** -0.5)

            w_down = torch.empty(len(individuals), dim, low_rank, device = self.device)
            w_up = torch.empty(len(individuals), dim_inner, low_rank, device = self.device)
            w_down.normal_(std = std_d)
            w_up.normal_(std = std_u)

            self.weight_down[key].data[individuals] = w_down.to(self._dtype)
            self.weight_up[key].data[individuals] = w_up.to(self._dtype)

        self._sigma_reset_(individuals)

        return self

    reinit_individuals = reinit_individuals_

    @torch.no_grad()
    def repopulate_(
        self,
        std_down: float | None = None,
        std_up: float | None = None
    ):
        return self.reinit_individuals_(torch.arange(self.pop_size, device = self.device), std_down = std_down, std_up = std_up)

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

    # per-individual mutation step size (log-normal self-adaptation) - `epsilon`
    # for the mutation operators becomes a per-individual tensor, or a per-target
    # map of (down, up) pairs at `sigma_granularity` finer than 'pop', drawn from
    # a log-sigma that lives inside the genome: perturbed when an individual
    # mutates in place, recombined from its parents (geometric mean) when it is
    # (re)born through crossover, selected along with the weights it shaped

    def _sigma_tensors(self):
        # the unique step-size buffers - 'pop' / 'lora' / 'rank' share a single
        # tensor between the down and up factor (and, for 'pop', across every
        # adapter), so a perturb / recombine / reset touches each logical sigma
        # exactly once instead of double-drawing its noise

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
        selection_type = 'deterministic',
        parent_selection_type = 'tournament',
        crossover_type = 'average',
        mutation_type = 'full_gaussian',
        num_groups = 1,
        epsilon = 0.1,
        weight_decay = 0.0,
        soft_threshold = 0.0,
        tiered = False,
        tiers = None,
        strata = 'fitness',
        novelty = None,
        burn_in = 0,
        gen = None,
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size

        tiered = tiered or exists(tiers)

        if tiered:
            return self._evolve_tiered_(
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
                gen = gen,
                **kwargs
            )

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

        if self.adaptive_epsilon:
            self._sigma_recombine_(result.culled, parents)
            epsilon = self._sigma_epsilon_(result.culled)

        self.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses, **kwargs) \
            .mutate_(mutation_type, individuals = result.culled, epsilon = epsilon, **kwargs) \
            .regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

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
        **kwargs
    ):
        # tiered evolve - bin the population into quantile strata of an axis
        # (fitness, novelty, or per-island fitness) and process each tier by its
        # rule - keep / mutate / replace (top-tier clones + re-mutate) / crossover
        # (children of higher tiers) / reinit / archive (hof replay). the default
        # spec is the clone-and-perturb scheme of bench_tiered.py
        #
        # `burn_in` (an N_adapt-style pause) exempts individuals processed within
        # the last `burn_in` generations from further processing, giving them time
        # to adapt - requires `gen` (the caller's generation counter)

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

        top = tier_bins[0]
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

        if any(module.training for module in self.model.modules()):
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
            # chunks. the routes are sliced in lockstep with each chunk - a
            # chunk's rows are routed by their explicit individual ids, so
            # any chunk boundary is safe, not only tile-aligned ones

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

            key = path.replace('.', '_')
            dim, dim_inner = linear.in_features, linear.out_features

            # a supplied checkpoint must cover every target - silently random
            # initializing the missing ones would yield a quietly wrong adapter

            if exists(weight_down) or exists(weight_up):
                assert exists(weight_down) and exists(weight_up), 'weight_down and weight_up must be provided together'
                assert key in weight_down and key in weight_up, f'missing lora weights for target {path} in the provided checkpoint'

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
    assert num_generations >= 1, 'num_generations must be at least 1'
    assert patience >= 1, 'patience must be at least 1'

    best_fitness = float('-inf')
    best_index = 0
    streak = 0
    history = []

    for _ in maybe_progress(range(num_generations), progress, 'evolving'):
        fitnesses = cast_tensor(fitness_fn(population)).to(population.device).float()
        assert fitnesses.shape == (population.pop_size,), f'fitness_fn must return one fitness per individual, got {tuple(fitnesses.shape)}'

        gen_best_fitness = float(fitnesses.max())
        is_best = gen_best_fitness > best_fitness

        if is_best:
            best_fitness = gen_best_fitness
            best_index = int(fitnesses.argmax())

        history.append(dict(
            best_fitness = gen_best_fitness,
            mean_fitness = float(fitnesses.mean()),
        ))

        if exists(target_fitness):
            streak = streak + 1 if gen_best_fitness >= target_fitness else 0

            if streak >= patience:
                break

        population.evolve_(fitnesses, **evolve_kwargs)

    policy = population.merge_(best_index)

    if return_history:
        return policy, history

    return policy
