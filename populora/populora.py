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

from einops import einsum, rearrange, repeat
from torch_einops_utils import tree_map_tensor

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

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

def z_score(t, dim = -1, eps = 1e-5):
    mean = t.mean(dim = dim, keepdim = True)
    std = t.std(dim = dim, keepdim = True).clamp(min = eps)
    return (t - mean) / std

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
SelectionResult = namedtuple('SelectionResult', ['survivors', 'culled', 'elites'])

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
            return fitnesses.topk(num_select, dim=-1).indices

        elite_indices = fitnesses.topk(num_elites, dim=-1).indices

        mask = torch.ones_like(fitnesses, dtype = torch.bool)
        mask.scatter_(-1, elite_indices, False)

        sorted_mask_indices = mask.long().argsort(dim=-1, descending=True)
        remaining_indices = sorted_mask_indices[..., :pop_size - num_elites]

        remaining_fitnesses = fitnesses.gather(-1, remaining_indices)
        selected = select_fn(remaining_fitnesses, num_select - num_elites, **kwargs)

        mapped_selected = remaining_indices.gather(-1, selected)

        return torch.cat((elite_indices, mapped_selected), dim=-1)
    return inner

def select_deterministic(fitnesses, num_select, **kwargs):
    return fitnesses.topk(num_select, dim=-1).indices

def select_probabilistic(fitnesses, num_select, temperature = 1., **kwargs):
    probs = F.softmax(fitnesses / temperature, dim = -1)
    return torch.multinomial(probs, num_select, replacement = False)

def select_fuss(fitnesses, num_select, eps = 1e-5, **kwargs):
    # fitness uniform selection scheme - Marcus Hutter https://arxiv.org/abs/cs/0103015

    pop_size = fitnesses.shape[-1]
    sorted_fitness, sort_indices = fitnesses.sort(dim=-1)

    if pop_size == 1:
        return torch.rand_like(fitnesses).argsort(dim=-1)[..., :num_select]

    # voronoi cell sizes

    padded = torch.cat((sorted_fitness[..., :1], sorted_fitness, sorted_fitness[..., -1:]), dim=-1)
    voronoi_cell_sizes = (padded[..., 2:] - padded[..., :-2]) / 2

    # when all equal, voronoi cell sizes are 0, plus eps falls back to uniform
    selected = torch.multinomial(voronoi_cell_sizes + eps, num_select, replacement = False)
    return sort_indices.gather(-1, selected)

register_selection('deterministic', select_deterministic)
register_selection('probabilistic', select_probabilistic)
register_selection('fuss', select_fuss)

# parent selection

PARENT_SELECTION_REGISTRY = {}

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
    sorted_fitness, sort_indices = fitnesses.sort(dim=-1)
    batch_shape = fitnesses.shape[:-1]

    if pop_size == 1:
        return torch.randint(0, pop_size, (*batch_shape, num_children, num_parents_per_child), device = fitnesses.device)

    # voronoi cell sizes

    padded = torch.cat((sorted_fitness[..., :1], sorted_fitness, sorted_fitness[..., -1:]), dim=-1)
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

register_parent_selection('tournament', parent_select_tournament)
register_parent_selection('fuss', parent_select_fuss)
register_parent_selection('roulette', parent_select_roulette)

# crossover

CROSSOVER_REGISTRY = {}

def register_crossover(name: str, fn: callable):
    CROSSOVER_REGISTRY[name] = fn

def crossover_average(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    for key in population.weight_down:
        w_down = population.weight_down[key].data[parent_indices]
        w_up = population.weight_up[key].data[parent_indices]

        population.weight_down[key].data[child_indices] = w_down.mean(dim = 1)
        population.weight_up[key].data[child_indices] = w_up.mean(dim = 1)

# X1
def crossover_dare(population, parent_indices, child_indices, p = 0.7, fitnesses = None, **kwargs):
    for key in population.weight_down:
        w_down = population.weight_down[key].data[parent_indices]
        w_up = population.weight_up[key].data[parent_indices]

        w_down_dropped = F.dropout(w_down, p = p, training = True)
        w_up_dropped = F.dropout(w_up, p = p, training = True)

        population.weight_down[key].data[child_indices] = w_down_dropped.mean(dim = 1)
        population.weight_up[key].data[child_indices] = w_up_dropped.mean(dim = 1)

# X2
def crossover_layer_wise(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    device = population.device
    batch_indices = torch.arange(num_children, device = device)

    for key in population.weight_down:
        w_down = population.weight_down[key].data[parent_indices]
        w_up = population.weight_up[key].data[parent_indices]

        parent_choice = torch.randint(0, num_parents, (num_children,), device = device)

        population.weight_down[key].data[child_indices] = w_down[batch_indices, parent_choice]
        population.weight_up[key].data[child_indices] = w_up[batch_indices, parent_choice]

# X3
def crossover_svd_subspace(population, parent_indices, child_indices, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'svd subspace crossover requires exactly 2 parents'

    for key in population.weight_down:
        w_down = population.weight_down[key].data[parent_indices]
        w_up = population.weight_up[key].data[parent_indices]

        r = w_down.shape[-1]

        for i in range(num_children):
            U1, S1, V1 = _efficient_svd_of_lora(w_down[i, 0], w_up[i, 0])
            U2, S2, V2 = _efficient_svd_of_lora(w_down[i, 1], w_up[i, 1])

            k = torch.randint(1, r, (1,)).item() if r > 1 else 1

            U_child = torch.cat((U1[:, :k], U2[:, k:]), dim = 1)
            S_child = torch.cat((S1[:k], S2[k:]), dim = 0)
            V_child = torch.cat((V1[:, :k], V2[:, k:]), dim = 1)

            S_sqrt = torch.sqrt(S_child)

            population.weight_down[key].data[child_indices[i]] = U_child * S_sqrt
            population.weight_up[key].data[child_indices[i]] = V_child * S_sqrt

# X4
def crossover_extrapolative(population, parent_indices, child_indices, eta_min = 1.0, eta_max = 1.5, fitnesses = None, **kwargs):
    num_children, num_parents = parent_indices.shape
    assert num_parents == 2, 'extrapolative crossover requires exactly 2 parents'

    eta = random.uniform(eta_min, eta_max)

    for key in population.weight_down:
        w_down = population.weight_down[key].data[parent_indices]
        w_up = population.weight_up[key].data[parent_indices]

        population.weight_down[key].data[child_indices] = w_down[:, 0].lerp(w_down[:, 1], eta)
        population.weight_up[key].data[child_indices] = w_up[:, 0].lerp(w_up[:, 1], eta)

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

    tournament_size = min(max(kwargs.get('tournament_size', 3), num_bad_parents), pop_size)

    # 1. select bad parents via tournament on inverted fitnesses

    inverted_fitnesses = -fitnesses
    inv_fit_expanded = repeat(inverted_fitnesses, 'p -> c p', c = num_children).clone()

    inv_fit_expanded.scatter_(-1, parent_indices, -float('inf'))

    rand_shape = (num_children, pop_size)
    contender_ids = torch.randn(rand_shape, device = device).argsort(dim = -1)[..., :tournament_size]

    tournaments = inv_fit_expanded.gather(-1, contender_ids)

    if num_bad_parents == 1:
        bad_parent_indices = contender_ids.gather(-1, tournaments.argmax(dim = -1, keepdim = True))
    else:
        top_winners = tournaments.topk(num_bad_parents, dim = -1, largest = True, sorted = False).indices
        bad_parent_indices = contender_ids.gather(-1, top_winners)

    all_parent_indices = torch.cat((parent_indices, bad_parent_indices), dim = -1)

    # 2. compute z-scores

    selected_fitnesses = fitnesses[all_parent_indices]

    z_scores = z_score(selected_fitnesses, dim = -1)
    weights = z_scores / z_scores.shape[-1]

    # 3. apply update

    for key in population.weight_down:
        w_down = population.weight_down[key].data[all_parent_indices]
        w_up = population.weight_up[key].data[all_parent_indices]

        w_down_mean = w_down.mean(dim = 1)
        w_up_mean = w_up.mean(dim = 1)

        w_down_update = einsum(weights, w_down, 'c p, c p ... -> c ...')
        w_up_update = einsum(weights, w_up, 'c p, c p ... -> c ...')

        population.weight_down[key].data[child_indices] = w_down_mean + eta * w_down_update
        population.weight_up[key].data[child_indices] = w_up_mean + eta * w_up_update

register_crossover('xes', crossover_xes)

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
        mutation_registry: dict | None = None
    ):
        super().__init__()
        self.model = model
        self.pop_size = pop_size
        self.selection_registry = selection_registry
        self.parent_selection_registry = parent_selection_registry
        self.crossover_registry = crossover_registry
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
                elites = fitnesses.topk(num_elites, dim=-1).indices if num_elites > 0 else all_indices[:0]
                return SelectionResult(all_indices, all_indices[:0], elites)

            survivors = select_fn(fitnesses, num_survivors, **kwargs)
            mask = torch.ones(group_size, dtype = torch.bool, device = self.device)
            mask.scatter_(-1, survivors, False)

            sorted_mask_indices = mask.long().argsort(dim=-1, descending=True)
            culled = sorted_mask_indices[..., :group_size - num_survivors]

            elites = survivors[..., :num_elites]
            return SelectionResult(survivors, culled, elites)

        fitnesses_grouped = rearrange(fitnesses, '(g p) -> g p', g = num_groups)

        if num_survivors >= group_size:
            if num_elites > 0:
                elites = fitnesses_grouped.topk(num_elites, dim=-1).indices
            else:
                elites = repeat(all_indices[:0], 'p -> g p', g = num_groups)
            survivors = repeat(all_indices, 'p -> g p', g = num_groups)
            culled = repeat(all_indices[:0], 'p -> g p', g = num_groups)
        else:
            survivors = select_fn(fitnesses_grouped, num_survivors, **kwargs)
            mask = torch.ones(num_groups, group_size, dtype = torch.bool, device = self.device)
            mask.scatter_(-1, survivors, False)

            sorted_mask_indices = mask.long().argsort(dim=-1, descending=True)
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
            kwargs = {**kwargs, 'fitnesses': fitnesses}

        crossover_fn = crossover_registry[crossover_type]
        crossover_fn(self, parent_indices, child_indices, **kwargs)

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
