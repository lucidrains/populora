from __future__ import annotations

import math
from collections import namedtuple
from functools import wraps
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import Tensor, cat
import torch.nn.functional as F
from torch.linalg import qr, svd

from einops import einsum, rearrange, repeat
from torch_einops_utils import pad_right_at_dim_to, z_score

from populora._utils import default, exists

if TYPE_CHECKING:
    from populora.population import Population

# tensor helpers

def _efficient_svd_of_lora(weight_down, weight_up):
    # batched over any leading dims - one individual or many at once. computed
    # in float32 regardless of the storage dtype - low-rank singular values are
    # tiny and bf16 / fp8 mantissas are not enough to resolve them

    weight_down = weight_down.float()
    weight_up = weight_up.float()

    r = weight_down.shape[-1]

    Q_A, R_A = qr(weight_down)
    Q_B, R_B = qr(weight_up)

    C = einsum(R_A, R_B, '... i j, ... k j -> ... i k')
    U_C, S, V_C_T = svd(C, full_matrices = False)

    U = einsum(Q_A, U_C, '... d i, ... i s -> ... d s')
    V = einsum(Q_B, V_C_T, '... e j, ... s j -> ... e s')

    U = pad_right_at_dim_to(U, r, dim = -1)
    S = pad_right_at_dim_to(S, r, dim = -1)
    V = pad_right_at_dim_to(V, r, dim = -1)

    return U, S, V

def skew_symmetrize(t):
    return (t - rearrange(t, '... i j -> ... j i')) / 2

# noise helpers - numpy's ziggurat generator is ~2x faster than torch's
# box-muller for the big elementwise draws. a numpy generator is seeded from
# the torch generator's live state each call (advanced by one draw, so every
# call gets a distinct stream): seeded runs stay reproducible, a fresh
# torch.manual_seed restarts the noise exactly, and cuda keeps torch's own
# generator, which is device-fast

def _cpu_rng():
    torch.rand(1)
    state = torch.random.get_rng_state().view(torch.int64)
    seed = int(state.sum().item()) & 0x7FFFFFFFFFFFFFFF  # numpy seeds are non-negative
    return np.random.default_rng(seed)

def _normal_noise(shape, device):
    if device.type == 'cpu':
        return torch.from_numpy(_cpu_rng().standard_normal(shape, dtype = np.float32))
    return torch.randn(shape, device = device)

def _uniform_noise(shape, device, low, high):
    if device.type == 'cpu':
        return torch.from_numpy(_cpu_rng().uniform(low, high, shape).astype(np.float32))
    return torch.empty(shape, device = device).uniform_(low, high)

def _noise_like(w, epsilon):
    # epsilon-scaled per-individual-std gaussian noise, in w's precision

    return epsilon * w.std(dim = (1, 2), keepdim = True) * _normal_noise(w.shape, w.device)

# mutations

MUTATION_REGISTRY = dict()

def register_mutation(name: str, fn: callable):
    MUTATION_REGISTRY[name] = fn

def batchable(fn):
    # a mutation that can mutate a cohort of individuals in one batched call -
    # `mutate_` passes it a 1-d tensor of indices instead of looping per index

    fn.batch = True
    return fn

# M1
@batchable
def mutation_svd_structured(
    population: Population,
    idx: Tensor,
    epsilon: float = 0.1,
    **kwargs
):
    device = population.device

    for weight_down, weight_up in zip(population.weight_down.values(), population.weight_up.values()):
        dtype = weight_down.dtype
        w_down = weight_down.data[idx].float()
        w_up = weight_up.data[idx].float()

        U, S, V = _efficient_svd_of_lora(w_down, w_up)
        r = S.shape[-1]

        z = _normal_noise(S.shape, device)
        S_new = S * torch.exp(epsilon * z)

        M_U = _normal_noise((*S.shape[:-1], r, r), device)
        R_U = torch.eye(r, device = device) + epsilon * skew_symmetrize(M_U)

        M_V = _normal_noise((*S.shape[:-1], r, r), device)
        R_V = torch.eye(r, device = device) + epsilon * skew_symmetrize(M_V)

        U_new = einsum(U, R_U, '... d r, ... r s -> ... d s')
        V_new = einsum(V, R_V, '... e r, ... r s -> ... e s')

        S_sqrt = torch.sqrt(S_new)
        weight_down.data[idx] = einsum(U_new, S_sqrt, '... d r, ... r -> ... d r').to(dtype)
        weight_up.data[idx] = einsum(V_new, S_sqrt, '... e r, ... r -> ... e r').to(dtype)

# M2
@batchable
def mutation_layer_selective_gaussian(
    population: Population,
    idx: Tensor,
    epsilon: float = 0.1,
    f: float = 0.33,
    **kwargs
):
    keys = list(population.weight_down.keys())
    num_layers = len(keys)
    num_mutate = max(1, int(f * num_layers))

    # per-individual random subset of layers - `rand(...).topk` draws the same
    # permutation distribution as a per-row `randperm`; layer i is mutated when
    # it lands within the first `num_mutate` positions

    layer_choice = torch.rand(len(idx), num_layers, device = population.device).topk(num_mutate, dim = -1, sorted = False).indices

    mutate_mask = torch.zeros(len(idx), num_layers, dtype = torch.bool, device = population.device)
    mutate_mask.scatter_(1, layer_choice, True)

    for i, key in enumerate(keys):
        rows = idx[mutate_mask[:, i]]

        if len(rows) == 0:
            continue

        w_down = population.weight_down[key]
        w_up = population.weight_up[key]

        w_down_rows = w_down.data[rows].float()
        w_up_rows = w_up.data[rows].float()

        w_down_rows.add_(_noise_like(w_down_rows, epsilon))
        w_up_rows.add_(_noise_like(w_up_rows, epsilon))

        w_down.data[rows] = w_down_rows.to(w_down.dtype)
        w_up.data[rows] = w_up_rows.to(w_up.dtype)

# M3
@batchable
def mutation_component_masking(
    population: Population,
    idx: Tensor,
    rho: float = 0.3,
    **kwargs
):
    device = population.device

    for weight_down, weight_up in zip(population.weight_down.values(), population.weight_up.values()):
        dtype = weight_down.dtype
        w_down = weight_down.data[idx].float()
        w_up = weight_up.data[idx].float()

        U, S, V = _efficient_svd_of_lora(w_down, w_up)
        r = S.shape[-1]

        num_drop = math.ceil(rho * r)
        drop_indices = torch.rand(len(idx), r, device = device).topk(num_drop, dim = -1, sorted = False).indices

        S_new = S.clone()
        S_new.scatter_(-1, drop_indices, 0.)

        S_sqrt = torch.sqrt(S_new)
        weight_down.data[idx] = einsum(U, S_sqrt, '... d r, ... r -> ... d r').to(dtype)
        weight_up.data[idx] = einsum(V, S_sqrt, '... e r, ... r -> ... e r').to(dtype)

# M4
@batchable
def mutation_full_gaussian(
    population: Population,
    idx: Tensor,
    epsilon: float = 0.15,
    **kwargs
):
    for weight_down, weight_up in zip(population.weight_down.values(), population.weight_up.values()):
        dtype = weight_down.dtype
        w_down = weight_down.data[idx].float()
        w_up = weight_up.data[idx].float()

        w_down.add_(_noise_like(w_down, epsilon))
        w_up.add_(_noise_like(w_up, epsilon))

        weight_down.data[idx] = w_down.to(dtype)
        weight_up.data[idx] = w_up.to(dtype)

# M5
@batchable
def mutation_neftune_style(
    population: Population,
    idx: Tensor,
    alpha: float = 10.0,
    **kwargs
):
    for weight_down in population.weight_down.values():
        dtype = weight_down.dtype
        w_down = weight_down.data[idx].float()

        bound = alpha / math.sqrt(w_down.shape[-2] * w_down.shape[-1])
        w_down.add_(_uniform_noise(w_down.shape, w_down.device, -bound, bound))

        weight_down.data[idx] = w_down.to(dtype)

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

class TieredResult(namedtuple('_TieredResult', ['tiers'])):
    # per-tier index tensors, best tier first - as processed, so burn_in-paused
    # individuals are excluded from their tier

    @property
    def survivors(self):
        return cat(self.tiers[:-1]) if len(self.tiers) > 1 else self.tiers[0]

    @property
    def culled(self):
        return self.tiers[-1]

    @property
    def elites(self):
        return self.tiers[0]

    @property
    def mid(self):
        return self.tiers[1] if len(self.tiers) > 2 else self.tiers[0][:0]

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

        remaining_indices = mask.long().topk(pop_size - num_elites, dim = -1, sorted = False).indices

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
        return sort_indices[..., :num_select]

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

    tournament_size = min(tournament_size, pop_size)
    num_parents_per_child = min(num_parents_per_child, tournament_size)

    batch_shape = fitnesses.shape[:-1]
    rand_shape = (*batch_shape, num_children, pop_size)
    contender_ids = torch.randn(rand_shape, device = device).topk(tournament_size, dim = -1, sorted = False).indices

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
    num_samples = num_children * num_parents_per_child
    batch_shape = fitnesses.shape[:-1]

    if math.isinf(temperature):
        # a uniform draw - softmax over +-inf fitnesses at infinite temperature
        # would come out all-nan

        if fitnesses.shape[-1] <= 1:
            selected = torch.zeros((*batch_shape, num_samples), dtype = torch.long, device = fitnesses.device)
        else:
            selected = torch.randint(0, fitnesses.shape[-1], (*batch_shape, num_samples), device = fitnesses.device)
    else:
        probs = F.softmax(fitnesses / temperature, dim = -1)
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
        w_down.data[child_indices] = w_down.data[parent_indices].float().mean(dim = 1).to(w_down.dtype)
        w_up.data[child_indices] = w_up.data[parent_indices].float().mean(dim = 1).to(w_up.dtype)

# X1
def crossover_dare(population, parent_indices, child_indices, p = 0.7, fitnesses = None, **kwargs):
    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[parent_indices].float()
        w_up_parents = w_up.data[parent_indices].float()

        w_down_dropped = F.dropout(w_down_parents, p = p, training = True)
        w_up_dropped = F.dropout(w_up_parents, p = p, training = True)

        w_down.data[child_indices] = w_down_dropped.mean(dim = 1).to(w_down.dtype)
        w_up.data[child_indices] = w_up_dropped.mean(dim = 1).to(w_up.dtype)

# X2
def crossover_layer_wise(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    device = population.device
    batch_indices = torch.arange(num_children, device = device)

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        parent_choice = torch.randint(0, num_parents, (num_children,), device = device)
        w_down.data[child_indices] = w_down.data[parent_indices][batch_indices, parent_choice].to(w_down.dtype)
        w_up.data[child_indices] = w_up.data[parent_indices][batch_indices, parent_choice].to(w_up.dtype)

# X3
def crossover_svd_subspace(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'svd subspace crossover requires exactly 2 parents'

    device = population.device

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        dtype = w_down.dtype
        w_down_parents = w_down.data[parent_indices].float()
        w_up_parents = w_up.data[parent_indices].float()

        r = w_down_parents.shape[-1]

        # batched svd over both parents of every child - `_efficient_svd_of_lora`
        # takes any leading dims, so the per-child loop folds into one call

        U, S, V = _efficient_svd_of_lora(w_down_parents, w_up_parents)

        U1, U2 = U[:, 0], U[:, 1]
        S1, S2 = S[:, 0], S[:, 1]
        V1, V2 = V[:, 0], V[:, 1]

        if r > 1:
            k = torch.randint(1, r, (num_children,), device = device)
        else:
            # rank 1 admits no split point - each child clones one parent wholesale

            k = torch.randint(0, 2, (num_children,), device = device)

        split_mask = torch.arange(r, device = device)[None, :] < k[:, None]

        U_child = torch.where(split_mask[:, None, :], U1, U2)
        S_child = torch.where(split_mask, S1, S2).clamp(min = 0.)
        V_child = torch.where(split_mask[:, None, :], V1, V2)

        S_sqrt = torch.sqrt(S_child)

        w_down.data[child_indices] = (U_child * S_sqrt[:, None, :]).to(dtype)
        w_up.data[child_indices] = (V_child * S_sqrt[:, None, :]).to(dtype)

# X4
def crossover_extrapolative(population, parent_indices, child_indices, eta_min = 1.0, eta_max = 1.5, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'extrapolative crossover requires exactly 2 parents'
    device = population.device

    eta = torch.empty((num_children, 1, 1), device = device).uniform_(eta_min, eta_max)

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[parent_indices].float()
        w_up_parents = w_up.data[parent_indices].float()

        w_down.data[child_indices] = w_down_parents[:, 0].lerp(w_down_parents[:, 1], eta).to(w_down.dtype)
        w_up.data[child_indices] = w_up_parents[:, 0].lerp(w_up_parents[:, 1], eta).to(w_up.dtype)

register_crossover('average', crossover_average)
register_crossover('dare', crossover_dare)
register_crossover('layer_wise', crossover_layer_wise)
register_crossover('svd_subspace', crossover_svd_subspace)
register_crossover('extrapolative', crossover_extrapolative)

# X6 - replacement: each child is an exact copy of its single parent, no mixing.
# used with num_parents_per_child = 1 by the tiered evolve mode, which then
# re-mutates the clone

def crossover_clone(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down.data[child_indices] = w_down.data[parent_indices][:, 0].to(w_down.dtype)
        w_up.data[child_indices] = w_up.data[parent_indices][:, 0].to(w_up.dtype)

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

    contender_ids = rand.topk(tournament_size, dim = -1, sorted = False).indices

    neg_fitnesses = repeat(-fitnesses, 'p -> c p', c = num_children).gather(-1, contender_ids)

    worst_ids = neg_fitnesses.topk(num_bad_parents, dim = -1, largest = True, sorted = False).indices
    bad_parent_indices = contender_ids.gather(-1, worst_ids)

    all_parent_indices = cat((parent_indices, bad_parent_indices), dim = -1)

    # 2. compute z-scored weights

    selected_fitnesses = fitnesses[all_parent_indices]
    weights = z_score(selected_fitnesses, dim = -1) / selected_fitnesses.shape[-1]

    # 3. apply update - mean + eta * weighted direction

    for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
        w_down_parents = w_down.data[all_parent_indices].float()
        w_up_parents = w_up.data[all_parent_indices].float()

        w_down.data[child_indices] = (w_down_parents.mean(dim = 1) + eta * einsum(weights, w_down_parents, 'c p, c p ... -> c ...')).to(w_down.dtype)
        w_up.data[child_indices] = (w_up_parents.mean(dim = 1) + eta * einsum(weights, w_up_parents, 'c p, c p ... -> c ...')).to(w_up.dtype)

register_crossover('xes', crossover_xes)
register_crossover('clone', crossover_clone)

# tier rules - each receives (population, indices, sources, top, fitnesses) plus
# the shared evolve params in kwargs; sources = higher tiers, top = best tier

TIER_RULE_REGISTRY = dict()

def register_tier_rule(name: str, fn: callable):
    TIER_RULE_REGISTRY[name] = fn

_UNIFORM_DRAW_TEMPERATURE = float('inf')  # roulette at this temperature is a uniform draw

def _tier_reproduce(population, indices, sources, fitnesses, *, parent_selection_type = 'tournament', crossover_type = 'average', num_parents_per_child = 2, mutation_type = 'full_gaussian', epsilon = 0.1, num_groups = 1, temperature = None, **kwargs):
    # children for `indices`, parents drawn from `sources` - select, crossover, mutate

    if len(indices) == 0 or len(sources) == 0:
        return

    parent_kwargs = dict(kwargs, temperature = temperature) if exists(temperature) else kwargs

    parents = population.select_parents(
        parent_selection_type, fitnesses,
        num_children = len(indices),
        survivors = sources,
        num_parents_per_child = num_parents_per_child,
        num_groups = num_groups,
        **parent_kwargs
    )

    population.crossover_(crossover_type, parents, indices, fitnesses = fitnesses, **kwargs)
    population.mutate_(mutation_type, individuals = indices, epsilon = epsilon, **kwargs)

def tier_rule_keep(population, indices, sources = None, top = None, fitnesses = None, **kwargs):
    pass

def tier_rule_mutate(population, indices, sources = None, top = None, fitnesses = None, mutation_type = 'full_gaussian', epsilon = 0.1, **kwargs):
    population.mutate_(mutation_type, individuals = indices, epsilon = epsilon, **kwargs)

def tier_rule_replace(population, indices, sources = None, top = None, fitnesses = None, **kwargs):
    # exact copies of a uniformly random top-tier agent, then re-mutated

    kwargs.pop('parent_selection_type', None)
    kwargs.pop('crossover_type', None)
    kwargs.pop('num_parents_per_child', None)

    _tier_reproduce(
        population, indices, top, fitnesses,
        parent_selection_type = 'roulette',
        crossover_type = 'clone',
        num_parents_per_child = 1,
        temperature = kwargs.pop('temperature', _UNIFORM_DRAW_TEMPERATURE),
        **kwargs
    )

def tier_rule_crossover(population, indices, sources = None, top = None, fitnesses = None, **kwargs):
    # standard pipeline, parents drawn from the higher tiers

    _tier_reproduce(population, indices, sources, fitnesses, **kwargs)

def tier_rule_reinit(population, indices, sources = None, top = None, fitnesses = None, **kwargs):
    population.reinit_individuals_(indices)

def tier_rule_archive(population, indices, sources = None, top = None, fitnesses = None, **kwargs):
    # replay archived individuals into the tier - requires hof = HallOfFame(...)

    hof = kwargs.get('hof')
    assert exists(hof), 'the "archive" tier rule requires hof = HallOfFame(...)'
    assert len(hof) > 0, 'the "archive" tier rule requires a non-empty hof'

    entry_indices = hof.sample(len(indices), mode = kwargs.get('archive_mode', 'uniform'))

    for slot, entry_idx in zip(indices.tolist(), entry_indices.tolist()):
        entry = hof.entries[entry_idx]
        population.load_individual(
            dict(weight_down = entry.weight_down, weight_up = entry.weight_up),
            individual = slot
        )

register_tier_rule('keep', tier_rule_keep)
register_tier_rule('mutate', tier_rule_mutate)
register_tier_rule('replace', tier_rule_replace)
register_tier_rule('crossover', tier_rule_crossover)
register_tier_rule('reinit', tier_rule_reinit)
register_tier_rule('archive', tier_rule_archive)

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

    remaining_indices = mask.long().topk(island_size - num_elites, dim = -1, sorted = False).indices
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
        w_down_island = w_down.data[island_indices].float()
        w_up_island = w_up.data[island_indices].float()

        w_down_mean = w_down_island.mean(dim = 0) + eta * einsum(weights, w_down_island, 'p, p ... -> ...')
        w_up_mean = w_up_island.mean(dim = 0) + eta * einsum(weights, w_up_island, 'p, p ... -> ...')

        w_down_std = w_down_island[elite_local_indices].std(dim = 0, unbiased = False).clamp(min = noise_std_min)
        w_up_std = w_up_island[elite_local_indices].std(dim = 0, unbiased = False).clamp(min = noise_std_min)

        w_down.data[island_indices] = (w_down_mean + torch.randn_like(w_down_island) * w_down_std).to(w_down.dtype)
        w_up.data[island_indices] = (w_up_mean + torch.randn_like(w_up_island) * w_up_std).to(w_up.dtype)

def reinit_pool_and_breed(
    population: Population,
    island_idx: int,
    num_islands: int,
    fitnesses: Tensor,
    parent_islands: Sequence[int] | Tensor,
    parent_selection_type: str | callable = 'tournament',
    crossover_type: str | callable = 'average',
    mutation_type: str | callable = 'full_gaussian',
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
    select_fn = parent_selection_type if callable(parent_selection_type) else parent_selection_registry[parent_selection_type]

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
