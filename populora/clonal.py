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
def clonal_reproduce_(population: Population, champion: int, sigma: float):
    # slot 0 is an exact champion copy, slots 1.. are its hypermutated clones

    p = population.pop_size
    assert p > 1, 'clonal reproduction needs at least two slots'

    for key, w_down in population.weight_down.items():
        w_up = population.weight_up[key]
        d, r = w_down.shape[1:]

        w_down.data[0] = w_down[champion]
        w_up.data[0] = w_up[champion]

        w_down.data[1:] = w_down[champion] + sigma * d ** -0.5 * torch.randn_like(w_down[1:])
        w_up.data[1:] = w_up[champion] + sigma * r ** -0.5 * torch.randn_like(w_up[1:])

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
    sigma_min: float = 0.02,
    sigma_max: float = 16.0,
    gamma: float = 0.5,
    target_success: float = 0.2,
    stall_patience: int = 3,
    burst: float = 4.0,
    progress: bool = False,
    return_history: bool = False,
    init_state: ClonalState | None = None,
):
    """evolve by clonal selection with stagnation-triggered hypermutation.
    the champion is cloned each generation, its children are hypermutated in
    LoRA weight space, and the step size follows the 1/5 rule."""

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
    parent_fitness = float(fitnesses.max())

    for gen in maybe_progress(range(num_generations), progress, 'clonal'):
        gen_best, gen_mean = float(fitnesses.max()), float(fitnesses.mean())
        history.append(dict(best_fitness = gen_best, mean_fitness = gen_mean))

        if gen_best > state.best:
            state.best, state.stall = gen_best, 0
        else:
            state.stall += 1

        hit_target = exists(target_fitness) and state.best >= target_fitness
        target_streak = target_streak + 1 if hit_target else 0

        if target_streak >= patience:
            break

        if gen == num_generations - 1:
            break

        champion = int(fitnesses.argmax())
        champion_fitness = float(fitnesses[champion])

        clonal_adapt_sigma_(state, fitnesses, parent_fitness)
        clonal_reproduce_(population, champion, state.sigma)

        fitnesses = interactor.evaluate(population, action = action, horizon = horizon, num_episodes = num_episodes, seed = population.eval_seed)
        parent_fitness = champion_fitness

    policy = population.merge_(int(fitnesses.argmax()))

    if return_history:
        return policy, history

    return policy
