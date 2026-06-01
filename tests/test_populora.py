import pytest
param = pytest.mark.parametrize

import torch
from torch import allclose

from x_transformers import TransformerWrapper, Decoder
from einops import rearrange
from populora import Population, Populations, PopuLoRA, register_mutation

# helper

def get_model():
    return TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 16,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

# tests

def test_population():
    model = get_model()

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
            'attn_layers.layers.0.1.to_k',
            'attn_layers.layers.0.1.to_v'
        ]
    )

    x = torch.randint(0, 1000, (1, 16))

    # forward passes

    out_orig = pop(x)
    out_0 = pop(x, individual = 0)
    out_1 = pop(x, individual = 1)

    assert out_orig.shape == (1, 16, 1000)
    assert out_0.shape == (1, 16, 1000)

    assert not allclose(out_orig, out_0)
    assert not allclose(out_0, out_1)

    out_all = pop(x, all_individuals = True)
    assert out_all.shape == (4, 16, 1000)

    out_subset = pop(x, individuals = [0, 1])
    assert out_subset.shape == (2, 16, 1000)

def test_populations():
    model = get_model()

    pops = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    out_solver = pops(x, individual = 0, pop_name = 'solver')
    out_conj = pops(x, individual = 0, pop_name = 'conjecturer')

    assert out_solver.shape == (1, 16, 1000)
    assert not allclose(out_solver, out_conj)

def test_populora():
    populora = PopuLoRA(
        model = get_model(),
        num_teachers = 2,
        num_students = 2,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    out_t = populora(x, individual = 0, pop_name = 'teacher')
    out_s = populora(x, individual = 0, pop_name = 'student')

    assert out_t.shape == (1, 16, 1000)
    assert not allclose(out_t, out_s)

def test_mutations():
    pop = Population(
        get_model(),
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    # M1 on individual 0

    before = clone_weights()
    pop.mutate_('svd_structured', individual = 0)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][0], before[k][0])
        assert allclose(pop.weight_down[k][1:], before[k][1:])

    # M2 on subset

    before = clone_weights()
    pop.mutate_('layer_selective_gaussian', individuals = [1, 2])

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][1:3], before[k][1:3])
        assert allclose(pop.weight_down[k][0], before[k][0])
        assert allclose(pop.weight_down[k][3], before[k][3])

    # M3 on all

    before = clone_weights()
    pop.mutate_('component_masking', all_individuals = True)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k], before[k])

    # M4

    before = clone_weights()
    pop.mutate_('full_gaussian', individual = 1)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][1], before[k][1])

    # M5

    before = clone_weights()
    pop.mutate_('neftune_style', individual = 2)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2], before[k][2])

    # unknown mutation

    with pytest.raises(AssertionError, match = 'unknown mutation type'):
        pop.mutate_('nonexistent_mutation', individual = 0)

def test_select():
    pop = Population(
        get_model(),
        pop_size = 4,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.9, 0.5, 0.2])

    # deterministic

    survivor_indices, culled_indices, elites = pop.select(
        'deterministic',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.25
    )

    assert survivor_indices.shape == (2,)
    assert culled_indices.shape == (2,)
    assert set(survivor_indices.tolist()) == {1, 2}
    assert set(culled_indices.tolist()) == {0, 3}

    # probabilistic

    survivor_indices, culled_indices, elites = pop.select(
        'probabilistic',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.25
    )

    assert survivor_indices.shape == (2,)
    assert culled_indices.shape == (2,)

    # FUSS

    survivor_indices, culled_indices, elites = pop.select(
        'fuss',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.25
    )

    assert survivor_indices.shape == (2,)
    assert culled_indices.shape == (2,)

    # custom selection - every other 2

    def select_every_other_two(fitnesses, num_select, **kwargs):
        sorted_indices = fitnesses.argsort(descending = True)
        return sorted_indices[::2][:num_select]

    pop_custom = Population(
        get_model(),
        pop_size = 4,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q'],
        selection_registry = dict(every_other_two = select_every_other_two)
    )

    survivor_indices, culled_indices, elites = pop_custom.select(
        'every_other_two',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.,
    )

    assert survivor_indices.shape == (2,)
    assert culled_indices.shape == (2,)

    # fitnesses sorted desc = [0.9, 0.5, 0.2, 0.1] -> indices [1, 2, 3, 0]
    # every other 2 -> indices [1, 3]

    assert set(survivor_indices.tolist()) == {1, 3}

    # unknown selection

    with pytest.raises(AssertionError, match = 'unknown selection type'):
        pop.select('nonexistent_selection', fitnesses)

def test_custom_mutation():
    pop = Population(
        get_model(),
        pop_size = 2,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    def mutation_random_signs(population, idx, **kwargs):
        for key in population.weight_down.keys():
            w = population.weight_down[key][idx]
            w.add_(torch.randint_like(w, 0, 2) * 2 - 1)

    register_mutation('random_signs', mutation_random_signs)

    before = {k: v.clone() for k, v in pop.weight_down.items()}

    # mutate random individual

    rand_idx = torch.randint(0, 2, (1,)).item()
    other_idx = 1 - rand_idx

    pop.mutate_('random_signs', individual = rand_idx)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][rand_idx], before[k][rand_idx])
        assert allclose(pop.weight_down[k][other_idx], before[k][other_idx])

def test_crossovers():
    pop = Population(
        get_model(),
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    parent_indices = torch.tensor([[0, 1], [0, 1]])
    child_indices = torch.tensor([2, 3])

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    # X1 DARE
    before = clone_weights()
    pop.crossover_('dare', parent_indices, child_indices)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])

    # X2 Layer-wise
    before = clone_weights()
    pop.crossover_('layer_wise', parent_indices, child_indices)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])

    # X3 SVD Subspace
    before = clone_weights()
    pop.crossover_('svd_subspace', parent_indices, child_indices)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])

    # X4 Extrapolative
    before = clone_weights()
    pop.crossover_('extrapolative', parent_indices, child_indices)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])

@param('num_parents', [2, 3])
def test_evolution_generation(num_parents):
    pop = Population(
        get_model(),
        pop_size = 6,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8, 0.6])

    # 1. survivor selection

    survivor_indices, culled_indices, elite_indices = pop.select(
        'deterministic',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.25
    )

    num_survivors = max(1, int(6 * 0.5))
    num_culled = 6 - num_survivors

    assert survivor_indices.shape == (num_survivors,)
    assert culled_indices.shape == (num_culled,)

    # 2. parent selection from survivors

    survivor_fitnesses = fitnesses[survivor_indices]
    num_children = len(culled_indices)

    parent_indices_in_survivors = pop.select_parents(
        'tournament',
        survivor_fitnesses,
        num_children = num_children,
        num_parents_per_child = num_parents
    )

    assert parent_indices_in_survivors.shape == (num_children, num_parents)

    # map back to original population indices
    parent_indices = survivor_indices[parent_indices_in_survivors]

    # 3. reproduction (crossover: average the selected parent weights)

    pop.crossover_('average', parent_indices, culled_indices)

    # 4. mutation on children (protecting elites)

    pop.mutate_('full_gaussian', all_individuals = True, ignore_individuals = elite_indices)

def test_islands_no_influence():
    pop = Population(
        get_model(),
        pop_size = 12,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.rand(12)

    # 1. verify survivor selection never crosses islands

    survivors, *_ = pop.select('deterministic', fitnesses, survive_frac = 0.5, elite_frac = 0.25, num_groups = 3)

    survivors = rearrange(survivors, '(g s) -> g s', g = 3)
    group_indices = rearrange(torch.arange(3), 'g -> g 1')

    assert (survivors // 4 == group_indices).all(), 'survivors crossed island boundaries'

    # 2. verify parent selection never crosses islands

    parents = pop.select_parents('tournament', fitnesses, num_children = 6, num_parents_per_child = 2, num_groups = 3)

    parents = rearrange(parents, '(g c) p -> g c p', g = 3)
    group_indices = rearrange(torch.arange(3), 'g -> g 1 1')

    assert (parents // 4 == group_indices).all(), 'parents crossed island boundaries'
