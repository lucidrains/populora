# clonal selection + hypermutation, after de castro & von zuben (2002);
# the stagnation burst is CHC's cataclysmic mutation (eshelman 1991)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from populora._utils import default, exists, maybe_progress
from populora.interact import EnvInteractor
from populora.operators import _project_lora_weighted_sum
from populora.population import Population

@dataclass
class ClonalState:
    sigma: float = 1.0
    sigma_min: float = 0.02
    sigma_max: float = 16.0
    gamma: float = 0.5
    target_success: float = 0.2
    stall_patience: int = 3
    burst: float = 4.0
    best: float = float('-inf')
    stall: int = 0

@torch.no_grad()
def clonal_reset_(population: Population, sigma: float = 1.0):
    for key, w_down in population.weight_down.items():
        w_up = population.weight_up[key]
        d, r = w_down.shape[1:]

        w_down.data.copy_(torch.randn_like(w_down) * d ** -0.5 * sigma)
        w_up.data.copy_(torch.randn_like(w_up) * r ** -0.5 * sigma)

    return population

@torch.no_grad()
def clonal_reproduce_(
    population: Population,
    champion: int,
    sigma: float,
    *,
    fitnesses: Tensor | None = None,
    elite_frac: float = 0.25,
    max_eff_sigma: float = 2.0,
):
    # slot 0 is an exact champion copy; when `fitnesses` is provided, slot 1 is
    # the Eckart-Young rank-r subspace centroid of the top-mu elites (CMA-ES
    # log-rank weights) and slots 2.. are antithetic hypermutated clones around
    # the subspace centroid

    p = population.pop_size
    assert p > 1, 'clonal reproduction needs at least two slots'
    device = population.device

    use_centroid = exists(fitnesses) and sigma > 0. and p >= 4
    if use_centroid:
        num_elites = max(2, min(p // 2, int(round(p * elite_frac))))
        elite_idx = fitnesses.topk(num_elites).indices
        ranks = torch.arange(1, num_elites + 1, device = device, dtype = torch.float32)
        raw_w = math.log(num_elites + 0.5) - torch.log(ranks)
        weights = raw_w / raw_w.sum()
        champion = int(elite_idx[0].item())

    eff_sigma = min(sigma, max_eff_sigma) if exists(fitnesses) else sigma

    for key, w_down in population.weight_down.items():
        w_up = population.weight_up[key]
        dtype = w_down.dtype
        d, r = w_down.shape[1:]
        e = w_up.shape[1]

        d_champ = w_down.data[champion].clone()
        u_champ = w_up.data[champion].clone()

        w_down.data[0] = d_champ
        w_up.data[0] = u_champ

        if eff_sigma == 0.:
            w_down.data[1:] = d_champ
            w_up.data[1:] = u_champ
            continue

        if use_centroid:
            d_elites = w_down.data[elite_idx].unsqueeze(0)
            u_elites = w_up.data[elite_idx].unsqueeze(0)
            d_center, u_center = _project_lora_weighted_sum(d_elites, u_elites, weights.unsqueeze(0))
            d_center = d_center.squeeze(0).to(dtype)
            u_center = u_center.squeeze(0).to(dtype)

            w_down.data[1] = d_center
            w_up.data[1] = u_center
            start_slot = 2
            anchor_d, anchor_u = d_center.float(), u_center.float()
        else:
            start_slot = 1
            anchor_d, anchor_u = d_champ.float(), u_champ.float()

        num_clones = p - start_slot
        if num_clones > 0:
            if use_centroid and num_clones >= 2:
                half = (num_clones + 1) // 2
                noise_d = eff_sigma * (d ** -0.5) * torch.randn((half, d, r), device = device, dtype = torch.float32)
                noise_u = eff_sigma * (r ** -0.5) * torch.randn((half, e, r), device = device, dtype = torch.float32)

                d_plus = anchor_d.unsqueeze(0) + noise_d
                u_plus = anchor_u.unsqueeze(0) + noise_u
                d_minus = anchor_d.unsqueeze(0) - noise_d
                u_minus = anchor_u.unsqueeze(0) - noise_u

                d_pairs = torch.stack([d_plus, d_minus], dim = 1).reshape(-1, d, r)[:num_clones]
                u_pairs = torch.stack([u_plus, u_minus], dim = 1).reshape(-1, e, r)[:num_clones]

                w_down.data[start_slot:] = d_pairs.to(dtype)
                w_up.data[start_slot:] = u_pairs.to(dtype)
            else:
                w_down.data[start_slot:] = (anchor_d + eff_sigma * d ** -0.5 * torch.randn_like(w_down[start_slot:].float())).to(dtype)
                w_up.data[start_slot:] = (anchor_u + eff_sigma * r ** -0.5 * torch.randn_like(w_up[start_slot:].float())).to(dtype)

    population._ages.data[0] = population._ages.data[champion] + 1
    population._ages.data[1:] = 0

    return population

def clonal_adapt_sigma_(state: ClonalState, fitnesses: Tensor, parent_fitness: float):
    # burst on stagnation, then the 1/5 rule; success is measured against the
    # parent the clones came from, not the best clone

    if state.stall >= state.stall_patience:
        state.sigma = min(state.sigma_max, state.sigma * state.burst)
        state.stall = 0

    success = float((fitnesses[1:] > parent_fitness).float().mean()) if len(fitnesses) > 1 else 0.
    state.sigma = state.sigma * math.exp(state.gamma * (success - state.target_success))
    state.sigma = min(max(state.sigma, state.sigma_min), state.sigma_max)

    return state

def evolve_clonal(
    interactor: EnvInteractor,
    population: Population | None = None,
    model: nn.Module | None = None,
    *,
    pop_size: int | None = None,
    low_rank: int | None = None,
    lora_targets: Sequence[str] | None = None,
    action: Callable | None = None,
    horizon: int = 500,
    num_episodes: int = 1,
    num_generations: int = 80,
    target_fitness: float | None = None,
    patience: int = 1,
    seed: int | None = None,
    sigma: float = 1.0,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    gamma: float = 0.35,
    target_success: float = 0.2,
    stall_patience: int = 4,
    burst: float = 1.6,
    elite_frac: float = 0.25,
    progress: bool = False,
    return_history: bool = False,
    init_state: ClonalState | None = None,
):
    """evolve by clonal selection with Eckart-Young subspace recombination and
    stagnation-triggered hypermutation. slot 0 preserves the champion, slot 1
    holds the rank-r subspace centroid of the top-mu elites, and slots 2.. are
    antithetic hypermutated clones around the centroid; step size follows the
    1/5 rule against the same-seed parent in slot 0."""

    if not exists(population):
        assert exists(model) and exists(pop_size) and exists(low_rank), 'pass a population, or model + pop_size + low_rank'
        population = interactor.population(model, pop_size = pop_size, low_rank = low_rank, lora_targets = lora_targets, seed = seed)

    state = default(init_state, ClonalState(
        sigma = sigma,
        sigma_min = sigma_min,
        sigma_max = sigma_max,
        gamma = gamma,
        target_success = target_success,
        stall_patience = stall_patience,
        burst = burst,
    ))

    clonal_reset_(population, sigma = state.sigma)

    history = []
    target_streak = 0
    fitnesses = interactor.evaluate(population, action = action, horizon = horizon, num_episodes = num_episodes, seed = population.eval_seed)

    for gen in maybe_progress(range(num_generations), progress, 'clonal'):
        gen_best, gen_mean = float(fitnesses.max()), float(fitnesses.mean())
        history.append(dict(best_fitness = gen_best, mean_fitness = gen_mean))

        # slot 0 is the exact parent evaluated on the current generation's CRN
        # seed, giving a zero-variance same-seed baseline for the 1/5 rule
        same_seed_parent = float(fitnesses[0]) if gen > 0 else gen_best

        if gen_best > state.best:
            state.best, state.stall = gen_best, 0
        elif gen > 0 and gen_best > same_seed_parent:
            state.stall = 0
        else:
            state.stall += 1

        hit_target = exists(target_fitness) and state.best >= target_fitness
        target_streak = target_streak + 1 if hit_target else 0

        if target_streak >= patience:
            break

        if gen == num_generations - 1:
            break

        champion = int(fitnesses.argmax())

        if gen > 0:
            clonal_adapt_sigma_(state, fitnesses, same_seed_parent)

        clonal_reproduce_(population, champion, state.sigma, fitnesses = fitnesses, elite_frac = elite_frac)

        fitnesses = interactor.evaluate(population, action = action, horizon = horizon, num_episodes = num_episodes, seed = population.eval_seed)

    policy = population.merge_(int(fitnesses.argmax()))

    if return_history:
        return policy, history

    return policy
