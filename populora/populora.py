from __future__ import annotations

import math
import random
from functools import wraps
from contextlib import contextmanager
from collections import namedtuple

import torch
import torch.distributed as dist
from torch import Tensor, cat, is_tensor, stack
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleDict, Parameter, ParameterDict, init
from torch.linalg import qr, svd

from einops import einsum, rearrange, repeat
from torch_einops_utils import pad_right_at_dim_to, temp_eval, tree_map_tensor, z_score

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0] if len(arr) > 0 else None

def extract_dict(v, k):
    return v[k] if isinstance(v, dict) else v

# tensor helpers

def _efficient_svd_of_lora(weight_down, weight_up):
    r = weight_down.shape[-1]

    Q_A, R_A = qr(weight_down)
    Q_B, R_B = qr(weight_up)

    C = einsum(R_A, R_B, 'i j, k j -> i k')
    U_C, S, V_C_T = svd(C, full_matrices = False)

    U = einsum(Q_A, U_C, 'd i, i s -> d s')
    V = einsum(Q_B, V_C_T, 'e j, s j -> e s')

    U = pad_right_at_dim_to(U, r, dim = -1)
    S = pad_right_at_dim_to(S, r, dim = -1)
    V = pad_right_at_dim_to(V, r, dim = -1)

    return U, S, V

def skew_symmetrize(t):
    return (t - rearrange(t, 'i j -> j i')) / 2

# mutations

MUTATION_REGISTRY = dict()

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

    for weight_down, weight_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down = weight_down[idx]
        w_up = weight_up[idx]

        U, S, V = _efficient_svd_of_lora(w_down, w_up)
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
        weight_down[idx].copy_(einsum(U_new, S_sqrt, 'd r, r -> d r'))
        weight_up[idx].copy_(einsum(V_new, S_sqrt, 'e r, r -> e r'))

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

    for weight_down, weight_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down = weight_down[idx]
        w_up = weight_up[idx]

        U, S, V = _efficient_svd_of_lora(w_down, w_up)
        r = S.shape[-1]

        num_drop = math.ceil(rho * r)
        drop_indices = torch.randperm(r, device = device)[:num_drop]

        S_new = S.clone()
        S_new[drop_indices] = 0.0

        S_sqrt = torch.sqrt(S_new)
        weight_down[idx].copy_(einsum(U, S_sqrt, 'd r, r -> d r'))
        weight_up[idx].copy_(einsum(V, S_sqrt, 'e r, r -> e r'))

# M4
def mutation_full_gaussian(
    population: Population,
    idx: int,
    epsilon: float = 0.15,
    **kwargs
):
    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_i = w_down[idx]
        w_up_i = w_up[idx]

        w_down_i.add_(torch.randn_like(w_down_i), alpha = epsilon * w_down_i.std())
        w_up_i.add_(torch.randn_like(w_up_i), alpha = epsilon * w_up_i.std())

# M5
def mutation_neftune_style(
    population: Population,
    idx: int,
    alpha: float = 10.0,
    **kwargs
):
    for weight_down in population.weight_down.values():
        w_down = weight_down[idx]

        bound = alpha / math.sqrt(w_down.numel())
        noise = torch.empty_like(w_down).uniform_(-bound, bound)

        w_down.add_(noise)

register_mutation('svd_structured', mutation_svd_structured)
register_mutation('layer_selective_gaussian', mutation_layer_selective_gaussian)
register_mutation('component_masking', mutation_component_masking)
register_mutation('full_gaussian', mutation_full_gaussian)
register_mutation('neftune_style', mutation_neftune_style)

# survivor selection

SELECTION_REGISTRY = dict()

class SelectionResult(namedtuple('_SelectionResult', ['survivors', 'culled', 'elites'])):
    @property
    def selected_out_indices(self):
        return self.culled

    @property
    def culled_indices(self):
        return self.culled

    @property
    def survivor_indices(self):
        return self.survivors

    @property
    def elite_indices(self):
        return self.elites

def register_selection(name: str, fn: callable):
    SELECTION_REGISTRY[name] = fn

def with_elites(select_fn, elite_frac = 0.25):
    if elite_frac == 0.:
        return select_fn

    @wraps(select_fn)
    def inner(fitnesses, num_select, **kwargs):
        pop_size = fitnesses.shape[-1]
        num_elites = max(1, int(pop_size * elite_frac))

        if num_elites >= num_select:
            return fitnesses.topk(num_select, dim = -1).indices

        elite_indices = fitnesses.topk(num_elites, dim = -1).indices

        mask = torch.ones_like(fitnesses, dtype = torch.bool)
        mask.scatter_(-1, elite_indices, False)

        sorted_mask_indices = mask.long().argsort(dim = -1, descending = True)
        remaining_indices = sorted_mask_indices[..., :pop_size - num_elites]

        remaining_fitnesses = fitnesses.gather(-1, remaining_indices)
        selected = select_fn(remaining_fitnesses, num_select - num_elites, **kwargs)

        mapped_selected = remaining_indices.gather(-1, selected)

        return cat((elite_indices, mapped_selected), dim = -1)
    return inner

def select_deterministic(fitnesses, num_select, **kwargs):
    return fitnesses.topk(num_select, dim = -1).indices

def select_probabilistic(fitnesses, num_select, temperature = 1., **kwargs):
    probs = F.softmax(fitnesses / temperature, dim = -1)
    return torch.multinomial(probs, num_select, replacement = False)

def select_fuss(fitnesses, num_select, eps = 1e-5, **kwargs):
    # fitness uniform selection scheme - Marcus Hutter https://arxiv.org/abs/cs/0103015

    pop_size = fitnesses.shape[-1]
    sorted_fitness, sort_indices = fitnesses.sort(dim = -1)

    if pop_size == 1:
        return torch.rand_like(fitnesses).argsort(dim = -1)[..., :num_select]

    # voronoi cell sizes

    padded = cat((sorted_fitness[..., :1], sorted_fitness, sorted_fitness[..., -1:]), dim = -1)
    voronoi_cell_sizes = (padded[..., 2:] - padded[..., :-2]) / 2

    # when all equal, voronoi cell sizes are 0, plus eps falls back to uniform
    selected = torch.multinomial(voronoi_cell_sizes + eps, num_select, replacement = False)
    return sort_indices.gather(-1, selected)

register_selection('deterministic', select_deterministic)
register_selection('probabilistic', select_probabilistic)
register_selection('fuss', select_fuss)

# parent selection

PARENT_SELECTION_REGISTRY = dict()

def register_parent_selection(name: str, fn: callable):
    PARENT_SELECTION_REGISTRY[name] = fn

def parent_select_tournament(fitnesses, num_children, num_parents_per_child = 2, tournament_size = 3, **kwargs):
    pop_size = fitnesses.shape[-1]
    device = fitnesses.device

    batch_shape = fitnesses.shape[:-1]
    rand_shape = (*batch_shape, num_children, pop_size)
    contender_ids = torch.randn(rand_shape, device = device).argsort(dim = -1)[..., :tournament_size]

    if fitnesses.ndim == 1:
        tournaments = fitnesses[contender_ids]
    else:
        expanded_fitnesses = repeat(fitnesses, '... p -> ... c p', c = num_children)
        tournaments = expanded_fitnesses.gather(-1, contender_ids)

    if num_parents_per_child == 1:
        winners = tournaments.argmax(dim = -1)
        return contender_ids.gather(-1, rearrange(winners, '... -> ... 1'))

    top_winners = tournaments.topk(num_parents_per_child, dim = -1, largest = True, sorted = False).indices
    return contender_ids.gather(-1, top_winners)

def parent_select_fuss(fitnesses, num_children, num_parents_per_child = 2, eps = 1e-5, **kwargs):
    # fitness uniform selection scheme - Marcus Hutter https://arxiv.org/abs/cs/0103015

    pop_size = fitnesses.shape[-1]
    sorted_fitness, sort_indices = fitnesses.sort(dim = -1)
    batch_shape = fitnesses.shape[:-1]

    if pop_size == 1:
        return torch.randint(0, pop_size, (*batch_shape, num_children, num_parents_per_child), device = fitnesses.device)

    # voronoi cell sizes

    padded = cat((sorted_fitness[..., :1], sorted_fitness, sorted_fitness[..., -1:]), dim = -1)
    voronoi_cell_sizes = (padded[..., 2:] - padded[..., :-2]) / 2

    num_samples = num_children * num_parents_per_child
    selected = torch.multinomial(voronoi_cell_sizes + eps, num_samples, replacement = True)
    selected = rearrange(selected, '... (c p) -> ... c p', c = num_children)

    expanded_sort_indices = repeat(sort_indices, '... p -> ... c p', c = num_children)
    return expanded_sort_indices.gather(-1, selected)

def parent_select_roulette(fitnesses, num_children, num_parents_per_child = 2, temperature = 1., **kwargs):
    probs = F.softmax(fitnesses / temperature, dim = -1)
    num_samples = num_children * num_parents_per_child
    selected = torch.multinomial(probs, num_samples, replacement = True)
    return rearrange(selected, '... (c p) -> ... c p', c = num_children)

def parent_select_queen_bee(fitnesses, num_children, num_parents_per_child = 2, tournament_size = 3, num_elites = 1, **kwargs):
    # queen-bee mutant-bee evolution - Jung 2007 https://www.researchgate.net/publication/290131255

    device = fitnesses.device
    batch_shape = fitnesses.shape[:-1]

    elites = fitnesses.topk(num_elites, dim = -1).indices
    queen_indices = torch.randint(0, elites.shape[-1], (*batch_shape, num_children, 1), device = device)

    if elites.ndim == 1:
        queens = elites[queen_indices]
    else:
        queens = repeat(elites, '... e -> ... c e', c = num_children).gather(-1, queen_indices)

    drones = parent_select_tournament(fitnesses, num_children, num_parents_per_child = num_parents_per_child - 1, tournament_size = tournament_size, **kwargs)

    return cat((queens, drones), dim = -1)

register_parent_selection('tournament', parent_select_tournament)
register_parent_selection('fuss', parent_select_fuss)
register_parent_selection('roulette', parent_select_roulette)
register_parent_selection('queen_bee', parent_select_queen_bee)

# crossover

CROSSOVER_REGISTRY = dict()

def register_crossover(name: str, fn: callable):
    CROSSOVER_REGISTRY[name] = fn

def crossover_average(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down.data[child_indices] = w_down.data[parent_indices].mean(dim = 1)
        w_up.data[child_indices] = w_up.data[parent_indices].mean(dim = 1)

# X1
def crossover_dare(population, parent_indices, child_indices, p = 0.7, fitnesses = None, **kwargs):
    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[parent_indices]
        w_up_parents = w_up.data[parent_indices]

        w_down_dropped = F.dropout(w_down_parents, p = p, training = True)
        w_up_dropped = F.dropout(w_up_parents, p = p, training = True)

        w_down.data[child_indices] = w_down_dropped.mean(dim = 1)
        w_up.data[child_indices] = w_up_dropped.mean(dim = 1)

# X2
def crossover_layer_wise(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    device = population.device
    batch_indices = torch.arange(num_children, device = device)

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        parent_choice = torch.randint(0, num_parents, (num_children,), device = device)
        w_down.data[child_indices] = w_down.data[parent_indices][batch_indices, parent_choice]
        w_up.data[child_indices] = w_up.data[parent_indices][batch_indices, parent_choice]

# X3
def crossover_svd_subspace(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'svd subspace crossover requires exactly 2 parents'

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[parent_indices]
        w_up_parents = w_up.data[parent_indices]

        r = w_down_parents.shape[-1]

        for i in range(num_children):
            U1, S1, V1 = _efficient_svd_of_lora(w_down_parents[i, 0], w_up_parents[i, 0])
            U2, S2, V2 = _efficient_svd_of_lora(w_down_parents[i, 1], w_up_parents[i, 1])

            k = torch.randint(1, r, (1,)).item() if r > 1 else 1

            U_child = cat((U1[:, :k], U2[:, k:]), dim = 1)
            S_child = cat((S1[:k], S2[k:]), dim = 0).clamp(min = 0.)
            V_child = cat((V1[:, :k], V2[:, k:]), dim = 1)

            S_sqrt = torch.sqrt(S_child)

            w_down.data[child_indices[i]] = U_child * S_sqrt
            w_up.data[child_indices[i]] = V_child * S_sqrt

# X4
def crossover_extrapolative(population, parent_indices, child_indices, eta_min = 1.0, eta_max = 1.5, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'extrapolative crossover requires exactly 2 parents'
    device = population.device

    eta = torch.empty((num_children, 1, 1), device = device).uniform_(eta_min, eta_max)

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[parent_indices]
        w_up_parents = w_up.data[parent_indices]

        w_down.data[child_indices] = w_down_parents[:, 0].lerp(w_down_parents[:, 1], eta)
        w_up.data[child_indices] = w_up_parents[:, 0].lerp(w_up_parents[:, 1], eta)

register_crossover('average', crossover_average)
register_crossover('dare', crossover_dare)
register_crossover('layer_wise', crossover_layer_wise)
register_crossover('svd_subspace', crossover_svd_subspace)
register_crossover('extrapolative', crossover_extrapolative)

# X5
def crossover_xes(population, parent_indices, child_indices, fitnesses = None, num_bad_parents = None, eta = 1.0, **kwargs):
    assert exists(fitnesses), 'XES crossover requires fitnesses'

    device = population.device
    num_children, num_good_parents = parent_indices.shape
    num_bad_parents = default(num_bad_parents, num_good_parents)
    pop_size = fitnesses.shape[-1]

    tournament_size = min(max(kwargs.get('tournament_size', 3), num_bad_parents), pop_size - num_good_parents)

    # 1. select bad parents via tournament, excluding good parents without replacement

    rand = torch.randn((num_children, pop_size), device = device)
    rand.scatter_(-1, parent_indices, -float('inf'))

    contender_ids = rand.argsort(dim = -1, descending = True)[..., :tournament_size]

    neg_fitnesses = repeat(-fitnesses, 'p -> c p', c = num_children).gather(-1, contender_ids)

    worst_ids = neg_fitnesses.topk(num_bad_parents, dim = -1, largest = True, sorted = False).indices
    bad_parent_indices = contender_ids.gather(-1, worst_ids)

    all_parent_indices = cat((parent_indices, bad_parent_indices), dim = -1)

    # 2. compute z-scored weights

    selected_fitnesses = fitnesses[all_parent_indices]
    weights = z_score(selected_fitnesses, dim = -1) / selected_fitnesses.shape[-1]

    # 3. apply update - mean + eta * weighted direction

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[all_parent_indices]
        w_up_parents = w_up.data[all_parent_indices]

        w_down.data[child_indices] = w_down_parents.mean(dim = 1) + eta * einsum(weights, w_down_parents, 'c p, c p ... -> c ...')
        w_up.data[child_indices] = w_up_parents.mean(dim = 1) + eta * einsum(weights, w_up_parents, 'c p, c p ... -> c ...')

register_crossover('xes', crossover_xes)

# migration

MIGRATION_REGISTRY = dict()

def register_migration(name: str, fn: callable):
    MIGRATION_REGISTRY[name] = fn

def migrate_fuss_roll(
    fitnesses: Tensor,
    num_islands: int,
    migrate_frac: float = 0.1,
    elite_frac: float = 0.25,
    eps: float = 1e-5,
    **kwargs
):
    device = fitnesses.device
    pop_size = fitnesses.shape[-1]
    island_size = pop_size // num_islands
    num_elites = int(island_size * elite_frac)
    num_migrate = max(1, int(island_size * migrate_frac))

    assert num_elites + num_migrate <= island_size, 'elites + migrants cannot exceed island size'

    fitnesses_grouped = rearrange(fitnesses, '(i p) -> i p', i = num_islands)

    # exclude elites from migration candidates

    if num_elites > 0:
        elite_indices = fitnesses_grouped.topk(num_elites, dim = -1).indices
    else:
        elite_indices = torch.empty((num_islands, 0), dtype = torch.long, device = device)

    mask = torch.ones_like(fitnesses_grouped, dtype = torch.bool)
    mask.scatter_(-1, elite_indices, False)

    remaining_indices = mask.long().argsort(dim = -1, descending = True)[..., :island_size - num_elites]
    remaining_fitnesses = fitnesses_grouped.gather(-1, remaining_indices)

    # fuss to select migrants, roll to shift to neighboring island

    selected = select_fuss(remaining_fitnesses, num_migrate, eps = eps)
    migrate_local = remaining_indices.gather(-1, selected)

    offset = torch.arange(num_islands, device = device) * island_size
    migrate_abs = migrate_local + rearrange(offset, 'i -> i 1')

    new_arrangement = torch.arange(pop_size, device = device)
    sources = torch.roll(migrate_abs, shifts = 1, dims = 0)
    new_arrangement.scatter_(0, migrate_abs.flatten(), sources.flatten())

    return new_arrangement

register_migration('fuss_roll', migrate_fuss_roll)

# island reinitialization

ISLAND_REINIT_REGISTRY = dict()

def register_island_reinit(name: str, fn: callable):
    ISLAND_REINIT_REGISTRY[name] = fn

def reinit_es(
    population: Population,
    island_idx: int,
    num_islands: int,
    fitnesses: Tensor,
    elite_frac: float = 0.25,
    eta: float = 1.0,
    noise_std_min: float = 1e-5,
    **kwargs
):
    assert exists(fitnesses), 'ES reinit requires fitnesses'
    pop_size = population.pop_size
    island_size = pop_size // num_islands

    offset = island_idx * island_size
    island_indices = torch.arange(island_size, device = fitnesses.device) + offset

    island_fitnesses = fitnesses[island_indices]
    weights = z_score(island_fitnesses, dim = -1) / island_size

    num_elites = max(1, int(island_size * elite_frac))
    elite_local_indices = island_fitnesses.topk(num_elites, dim = -1).indices

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_island = w_down.data[island_indices]
        w_up_island = w_up.data[island_indices]

        w_down_mean = w_down_island.mean(dim = 0) + eta * einsum(weights, w_down_island, 'p, p ... -> ...')
        w_up_mean = w_up_island.mean(dim = 0) + eta * einsum(weights, w_up_island, 'p, p ... -> ...')

        w_down_std = w_down_island[elite_local_indices].std(dim = 0, unbiased = False).clamp(min = noise_std_min)
        w_up_std = w_up_island[elite_local_indices].std(dim = 0, unbiased = False).clamp(min = noise_std_min)

        w_down.data[island_indices] = w_down_mean + torch.randn_like(w_down_island) * w_down_std
        w_up.data[island_indices] = w_up_mean + torch.randn_like(w_up_island) * w_up_std

def reinit_pool_and_breed(
    population: Population,
    island_idx: int,
    num_islands: int,
    fitnesses: Tensor,
    parent_islands: list[int] | tuple[int, ...] | Tensor,
    parent_selection_type: str = 'tournament',
    crossover_type: str = 'average',
    mutation_type: str = 'full_gaussian',
    num_parents_per_child: int = 2,
    **kwargs
):
    assert exists(fitnesses), 'pool_and_breed reinit requires fitnesses'
    device = fitnesses.device
    pop_size = population.pop_size
    island_size = pop_size // num_islands

    offset = island_idx * island_size
    child_indices = torch.arange(island_size, device = device) + offset

    parent_islands_tensor = torch.tensor(parent_islands, device = device) if not isinstance(parent_islands, Tensor) else parent_islands

    parent_offsets = parent_islands_tensor * island_size
    parent_local_indices = torch.arange(island_size, device = device)

    parent_pool_indices = (rearrange(parent_offsets, 'i -> i 1') + parent_local_indices).flatten()
    pool_fitnesses = fitnesses[parent_pool_indices]

    parent_selection_registry = default(population.parent_selection_registry, PARENT_SELECTION_REGISTRY)
    select_fn = parent_selection_registry[parent_selection_type]

    selected_in_pool = select_fn(
        pool_fitnesses,
        num_children = island_size,
        num_parents_per_child = num_parents_per_child,
        **kwargs
    )

    parent_indices = parent_pool_indices[selected_in_pool]

    population.crossover_(crossover_type, parent_indices, child_indices, fitnesses = fitnesses, **kwargs)
    population.mutate_(mutation_type, individuals = child_indices, **kwargs)

register_island_reinit('es', reinit_es)
register_island_reinit('pool_and_breed', reinit_pool_and_breed)

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

        self.weight_down = ParameterDict()
        self.weight_up = ParameterDict()
        self._hooks = []

        self.lora_targets = tuple(lora_targets)

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
    def mutate_(
        self,
        mutation_type: str,
        individual: int | None = None,
        individuals: tuple[int, ...] | list[int] | Tensor | None = None,
        all_individuals: bool = False,
        ignore_individuals: tuple[int, ...] | list[int] | Tensor | None = None,
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

        if exists(ignore_individuals):
            ignore_set = set(ignore_individuals.tolist() if isinstance(ignore_individuals, Tensor) else ignore_individuals)
            indices = [i for i in indices if i not in ignore_set]

        for idx in indices:
            mutation_fn(self, idx, **kwargs)

    @torch.no_grad()
    def select(
        self,
        selection_type: str,
        fitnesses: Tensor,
        survive_frac: float = 0.8,
        elite_frac: float = 0.25,
        num_groups: int = 1,
        **kwargs
    ):
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size
        assert divisible_by(self.pop_size, num_groups)

        selection_registry = default(self.selection_registry, SELECTION_REGISTRY)
        assert selection_type in selection_registry, f'unknown selection type {selection_type}'

        group_size = self.pop_size // num_groups
        num_survivors = max(1, int(group_size * survive_frac))
        num_elites = max(1, int(group_size * elite_frac)) if elite_frac > 0. else 0
        all_indices = torch.arange(group_size, device = self.device)

        select_fn = with_elites(selection_registry[selection_type], elite_frac)

        if num_groups == 1:
            if num_survivors >= group_size:
                elites = fitnesses.topk(num_elites, dim = -1).indices if num_elites > 0 else all_indices[:0]
                return SelectionResult(all_indices, all_indices[:0], elites)

            survivors = select_fn(fitnesses, num_survivors, **kwargs)
            mask = torch.ones(group_size, dtype = torch.bool, device = self.device)
            mask.scatter_(-1, survivors, False)

            sorted_mask_indices = mask.long().argsort(dim = -1, descending = True)
            culled = sorted_mask_indices[..., :group_size - num_survivors]

            elites = survivors[..., :num_elites]
            return SelectionResult(survivors, culled, elites)

        fitnesses_grouped = rearrange(fitnesses, '(g p) -> g p', g = num_groups)

        if num_survivors >= group_size:
            if num_elites > 0:
                elites = fitnesses_grouped.topk(num_elites, dim = -1).indices
            else:
                elites = repeat(all_indices[:0], 'p -> g p', g = num_groups)
            survivors = repeat(all_indices, 'p -> g p', g = num_groups)
            culled = repeat(all_indices[:0], 'p -> g p', g = num_groups)
        else:
            survivors = select_fn(fitnesses_grouped, num_survivors, **kwargs)
            mask = torch.ones(num_groups, group_size, dtype = torch.bool, device = self.device)
            mask.scatter_(-1, survivors, False)

            sorted_mask_indices = mask.long().argsort(dim = -1, descending = True)
            culled = sorted_mask_indices[..., :group_size - num_survivors]

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
        selection_type: str,
        fitnesses: Tensor,
        num_children: int,
        num_parents_per_child: int = 2,
        num_groups: int = 1,
        **kwargs
    ):
        assert fitnesses.ndim == 1
        assert divisible_by(self.pop_size, num_groups)
        assert divisible_by(num_children, num_groups)

        parent_selection_registry = default(self.parent_selection_registry, PARENT_SELECTION_REGISTRY)
        assert selection_type in parent_selection_registry, f'unknown parent selection type {selection_type}'

        select_fn = parent_selection_registry[selection_type]

        if num_groups == 1:
            return select_fn(fitnesses, num_children, num_parents_per_child = num_parents_per_child, **kwargs)

        fitnesses_grouped = rearrange(fitnesses, '(g p) -> g p', g = num_groups)
        children_per_group = num_children // num_groups

        parents = select_fn(fitnesses_grouped, children_per_group, num_parents_per_child = num_parents_per_child, **kwargs)

        offset = torch.arange(num_groups, device = fitnesses.device) * (self.pop_size // num_groups)
        parents = parents + rearrange(offset, 'g -> g 1 1')

        return rearrange(parents, 'g c p -> (g c) p')

    @torch.no_grad()
    def crossover_(
        self,
        crossover_type: str,
        parent_indices: Tensor,
        child_indices: Tensor,
        fitnesses: Tensor | None = None,
        **kwargs
    ):
        crossover_registry = default(self.crossover_registry, CROSSOVER_REGISTRY)
        assert crossover_type in crossover_registry, f'unknown crossover type {crossover_type}'

        if exists(fitnesses):
            kwargs = dict(kwargs, fitnesses = fitnesses)

        crossover_fn = crossover_registry[crossover_type]
        crossover_fn(self, parent_indices, child_indices, **kwargs)

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

    @torch.no_grad()
    def reinit_islands_(
        self,
        reinit_type_or_fn: str | callable,
        islands: int | list[int] | tuple[int, ...] | Tensor,
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

    def route(
        self,
        individual = None,
        individuals = None,
        all_individuals = False
    ):
        return self._route(individual, individuals, all_individuals)

    @torch.no_grad()
    def merge_(self, individual = 0):
        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            w_down = self.weight_down[key][individual]
            w_up = self.weight_up[key][individual]
            linear.weight.add_(einsum(w_up, w_down, 'e r, d r -> e d'))

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def select_and_merge_(
        self,
        fitnesses: Tensor | None = None,
        topk: int | float | None = None,
        temperature: float = 1.0,
        indices: Tensor | tuple[int, ...] | list[int] | None = None,
        remove_hooks: bool = False
    ):
        assert exists(fitnesses) or exists(indices), 'either fitnesses or indices must be passed to select_and_merge_'

        if exists(fitnesses):
            assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.pop_size

            topk = default(topk, max(1, self.pop_size // 4))
            if isinstance(topk, float) and topk < 1.0:
                topk = max(1, int(self.pop_size * topk))

            topk_indices = fitnesses.topk(topk, dim = -1).indices
            topk_fitnesses = fitnesses[topk_indices]
        else:
            topk_indices = torch.tensor(indices, device = self.device) if not isinstance(indices, Tensor) else indices
            topk_fitnesses = torch.ones_like(topk_indices, dtype = torch.float32)

        weights = F.softmax(topk_fitnesses / temperature, dim = -1)

        for path in self.lora_targets:
            linear = self.model.get_submodule(path)
            key = path.replace('.', '_')
            w_down_topk = self.weight_down[key][topk_indices]
            w_up_topk = self.weight_up[key][topk_indices]

            delta = einsum(weights, w_up_topk, w_down_topk, 'k, k e r, k d r -> e d')
            linear.weight.add_(delta.to(linear.weight.dtype))

        if remove_hooks:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

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

            init.normal_(self.weight_down[key], std = std_d)
            init.normal_(self.weight_up[key], std = std_u)

    repopulate = repopulate_

    @contextmanager
    def _eval_and_no_grad(self, enabled):
        if not enabled:
            yield
            return

        with temp_eval(self), torch.no_grad():
            yield

    def _create_hook(self, lora_key: str):
        def hook(_, args, output):
            if self._individual is None:
                return output

            x = first(args)
            if not exists(x):
                return output

            weight_down, weight_up = self.weight_down[lora_key], self.weight_up[lora_key]

            if isinstance(self._individual, (list, tuple)) or self._individual is ...:
                weight_down_i, weight_up_i = weight_down[self._individual], weight_up[self._individual]
                p = weight_down_i.shape[0]

                x = rearrange(x, '(p b) ... -> p b ...', p = p)
                lora_out = einsum(x, weight_down_i.to(x.dtype), weight_up_i.to(x.dtype), 'p b ... d, p d r, p e r -> p b ... e')
                lora_out = rearrange(lora_out, 'p b ... -> (p b) ...')
            else:
                lora_out = einsum(x, weight_down[self._individual].to(x.dtype), weight_up[self._individual].to(x.dtype), '... d, d r, e r -> ... e')

            return output + lora_out.to(output.dtype)

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

            def maybe_repeat_batch(t):
                assert t.shape[0] in (1, p), f'batch dimension {t.shape[0]} must be equal to 1 or number of individuals {p}'
                return repeat(t, '1 ... -> p ...', p = p) if t.shape[0] == 1 else t

            args = tuple(
                tree_map_tensor(maybe_repeat_batch, a) if i not in ignore else a
                for i, a in enumerate(args)
            )

            kwargs = {
                k: tree_map_tensor(maybe_repeat_batch, v) if k not in ignore else v
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

    def forward(self, *args, **kwargs):
        return self.populations(*args, **kwargs)

# distributed population evaluation

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def evaluate_population_distributed(
    population: Population,
    eval_fn: callable,
    batch_eval: bool = False,
    device: torch.device | str | None = None
) -> Tensor:

    pop_size = population.pop_size
    device = default(device, population.device)

    world_size, rank = (dist.get_world_size(), dist.get_rank()) if is_distributed() else (1, 0)
    assigned_indices = list(range(rank, pop_size, world_size))

    fitnesses = torch.zeros(pop_size, device = device, dtype = torch.float32)

    if batch_eval:
        if len(assigned_indices) > 0:
            res = eval_fn(population, assigned_indices)
            fitnesses[assigned_indices] = res if is_tensor(res) else torch.tensor(res, device = device, dtype = torch.float32)
    else:
        for idx in assigned_indices:
            fitnesses[idx] = eval_fn(population, idx)

    if not is_distributed():
        return fitnesses

    dist.all_reduce(fitnesses, op = dist.ReduceOp.SUM)
    return fitnesses
