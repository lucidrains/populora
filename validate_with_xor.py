# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "einx",
#   "tqdm",
#   "fire",
#   "torch-einops-utils",
#   "x-mlps-pytorch",
#   "populora",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

from __future__ import annotations

import torch
import einx
import fire
from tqdm import tqdm
from x_mlps_pytorch import MLP

from populora import (
    Population,
    CROSSOVER_REGISTRY,
    SELECTION_REGISTRY,
    PARENT_SELECTION_REGISTRY,
    MUTATION_REGISTRY
)

# constants

REGULARIZATION_ONLY_MUTATIONS = {'component_masking'}

# xor dataset - batch dim of 1 for population parallel routing

XOR_X = torch.tensor([[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]])
XOR_Y = torch.tensor([[[0.], [1.], [1.], [0.]]])

# helpers

def exists(v):
    return v is not None

def xor_mse(preds):
    return einx.mean('p [b] [d] -> p', (preds - XOR_Y) ** 2)

def train_xor(
    crossover_type = 'average',
    parent_selection_type = 'tournament',
    survivor_selection_type = 'deterministic',
    mutation_type = 'full_gaussian',
    auxiliary_mutation_type = None,
    mutation_kwargs = None,
    num_generations = 600,
    pop_size = 64,
    low_rank = 4,
    hidden_dim = 8,
    survive_frac = 0.5,
    elite_frac = 0.25,
    tournament_size = 3,
    auxiliary_epsilon = 0.05,
    seed = 42,
    desc = 'evaluating'
):
    torch.manual_seed(seed)

    pop = Population(
        MLP(2, hidden_dim, 1),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1']
    )

    for _ in tqdm(range(num_generations), desc = desc, leave = False):
        preds = pop(XOR_X, all_individuals = True)
        fitnesses = -xor_mse(preds)

        result = pop.select(
            selection_type = survivor_selection_type,
            fitnesses = fitnesses,
            survive_frac = survive_frac,
            elite_frac = elite_frac
        )

        parents = pop.select_parents(
            selection_type = parent_selection_type,
            fitnesses = fitnesses,
            num_children = len(result.culled),
            num_parents_per_child = 2,
            tournament_size = tournament_size
        )

        if exists(crossover_type):
            pop.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses)
        else:
            for w_down, w_up in zip(pop.weight_down.values(), pop.weight_up.values()):
                w_down.data[result.culled] = w_down.data[parents[:, 0]]
                w_up.data[result.culled] = w_up.data[parents[:, 0]]

        if exists(mutation_type):
            pop.mutate_(mutation_type, individuals = result.culled, **(mutation_kwargs or {}))

        if exists(auxiliary_mutation_type):
            pop.mutate_(auxiliary_mutation_type, individuals = result.culled, epsilon = auxiliary_epsilon)

    preds = pop(XOR_X, all_individuals = True)
    return xor_mse(preds).min().item()

# main

def validate_with_xor(
    threshold = 0.1,
    pop_size = 64,
    low_rank = 4,
    hidden_dim = 8,
    crossover_generations = 1000,
    parent_selection_generations = 1200,
    survivor_selection_generations = 1200,
    mutation_generations = 1200,
    crossover_epsilon = 0.02,
    auxiliary_epsilon = 0.05,
    survive_frac = 0.5,
    elite_frac = 0.25,
    tournament_size = 3,
    seed = 42
):
    results = []

    # crossovers

    for name in CROSSOVER_REGISTRY:
        loss = train_xor(
            crossover_type = name,
            mutation_kwargs = dict(epsilon = crossover_epsilon),
            num_generations = crossover_generations,
            pop_size = pop_size,
            low_rank = low_rank,
            hidden_dim = hidden_dim,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            tournament_size = tournament_size,
            seed = seed,
            desc = f'crossover: {name}'
        )
        results.append(('crossover', name, loss))

    # parent selections

    for name in PARENT_SELECTION_REGISTRY:
        loss = train_xor(
            parent_selection_type = name,
            mutation_kwargs = dict(epsilon = crossover_epsilon),
            num_generations = parent_selection_generations,
            pop_size = pop_size,
            low_rank = low_rank,
            hidden_dim = hidden_dim,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            tournament_size = tournament_size,
            seed = seed,
            desc = f'parent_sel: {name}'
        )
        results.append(('parent_sel', name, loss))

    # survivor selections

    for name in SELECTION_REGISTRY:
        loss = train_xor(
            survivor_selection_type = name,
            mutation_kwargs = dict(epsilon = crossover_epsilon),
            num_generations = survivor_selection_generations,
            pop_size = pop_size,
            low_rank = low_rank,
            hidden_dim = hidden_dim,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            tournament_size = tournament_size,
            seed = seed,
            desc = f'survivor_sel: {name}'
        )
        results.append(('survivor_sel', name, loss))

    # mutations

    for name in MUTATION_REGISTRY:
        auxiliary = 'full_gaussian' if name in REGULARIZATION_ONLY_MUTATIONS else None
        loss = train_xor(
            crossover_type = None,
            mutation_type = name,
            auxiliary_mutation_type = auxiliary,
            auxiliary_epsilon = auxiliary_epsilon,
            num_generations = mutation_generations,
            pop_size = pop_size,
            low_rank = low_rank,
            hidden_dim = hidden_dim,
            survive_frac = survive_frac,
            elite_frac = elite_frac,
            tournament_size = tournament_size,
            seed = seed,
            desc = f'mutation: {name}'
        )
        label = f'{name}+{auxiliary}' if exists(auxiliary) else name
        results.append(('mutation', label, loss))

    # print results

    print('\n' + '=' * 60)
    print(' PopuLoRA XOR Validation Results')
    print('=' * 60)

    for category, name, loss in results:
        passed = loss < threshold
        status = '✓' if passed else '✗'
        print(f'  {status} {category:<14} {name:<35} {loss:.6f}')
        assert passed, f'{category} {name} did not converge (loss {loss:.6f})'

    print('=' * 60)
    print(f' all {len(results)} configurations validated')
    print('=' * 60 + '\n')

if __name__ == '__main__':
    fire.Fire(validate_with_xor)
