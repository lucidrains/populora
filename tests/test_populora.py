import pytest
param = pytest.mark.parametrize

import torch
import torch.nn as nn
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

    # X5 XES
    before = clone_weights()
    fitnesses = torch.tensor([0.9, 0.8, 0.1, 0.2])
    pop.crossover_('xes', parent_indices, child_indices, fitnesses = fitnesses)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])

def test_crossover_xes():
    pop = Population(
        get_model(),
        pop_size = 6,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.5, 0.6])

    # good parents: 0 and 1
    # bad parents should be chosen from [2, 3, 4, 5], but inverted fitness means [2, 3] will be strongly favored.
    parent_indices = torch.tensor([[0, 1]])
    child_indices = torch.tensor([4])

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    before = clone_weights()

    # Run with eta = 0.0 (should just be the mean of good and bad parents)
    pop.crossover_('xes', parent_indices, child_indices, fitnesses = fitnesses, eta = 0.0)
    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][4], before[k][4])

    # Run with larger eta
    pop.crossover_('xes', parent_indices, child_indices, fitnesses = fitnesses, eta = 1.0)

    # Verify that without fitnesses it raises an assertion
    with pytest.raises(AssertionError, match = 'XES crossover requires fitnesses'):
        pop.crossover_('xes', parent_indices, child_indices)

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

def test_parent_select_queen_bee():
    pop = Population(
        get_model(),
        pop_size = 6,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8, 0.6])

    # select 3 survivors, and 1 elite
    survivor_indices, culled_indices, elite_indices = pop.select(
        'deterministic',
        fitnesses,
        survive_frac = 0.5,
        elite_frac = 0.33
    )

    survivor_fitnesses = fitnesses[survivor_indices]

    # queen bee selection just uses the top individuals from fitnesses as queens
    parents = pop.select_parents('queen_bee', survivor_fitnesses, num_children = 3, num_parents_per_child = 2)
    assert parents.shape == (3, 2)

def test_migration():
    pop = Population(
        get_model(),
        pop_size = 8,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    before = clone_weights()

    pop.migrate_('fuss_roll', fitnesses, num_islands = 4, migrate_frac = 0.5, elite_frac = 0.5)

    for k in pop.weight_down.keys():
        w = pop.weight_down[k]
        w_before = before[k]

        assert allclose(w[0::2], w_before[0::2])
        assert allclose(w[1::2], torch.roll(w_before[1::2], shifts = 1, dims = 0))

    # custom callable

    def custom_migration(fitnesses, num_islands, **kwargs):
        indices = torch.arange(fitnesses.shape[-1], device = fitnesses.device)
        return torch.roll(indices, shifts = 1, dims = 0)

    before_custom = clone_weights()
    pop.migrate_(custom_migration, fitnesses, num_islands = 4)

    for k in pop.weight_down.keys():
        w = pop.weight_down[k]
        w_before = before_custom[k]

        assert allclose(w, torch.roll(w_before, shifts = 1, dims = 0))

def test_reinit_islands():
    pop = Population(
        get_model(),
        pop_size = 8,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    # es - reinit islands 0 and 2, leaving 1 and 3 untouched

    before = clone_weights()

    pop.reinit_islands_('es', islands = [0, 2], num_islands = 4, fitnesses = fitnesses)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][0:2], before[k][0:2])
        assert allclose(pop.weight_down[k][2:4], before[k][2:4])
        assert not allclose(pop.weight_down[k][4:6], before[k][4:6])
        assert allclose(pop.weight_down[k][6:8], before[k][6:8])

    # pool and breed - reinit island 1 from island 3

    before = clone_weights()

    pop.reinit_islands_('pool_and_breed', islands = 1, parent_islands = [3], num_islands = 4, fitnesses = fitnesses)

    for k in pop.weight_down.keys():
        assert allclose(pop.weight_down[k][0:2], before[k][0:2])
        assert not allclose(pop.weight_down[k][2:4], before[k][2:4])
        assert allclose(pop.weight_down[k][4:8], before[k][4:8])

    # custom callable

    before = clone_weights()

    def zero_out(population, island_idx, num_islands, **kwargs):
        island_size = population.pop_size // num_islands
        sl = slice(island_idx * island_size, (island_idx + 1) * island_size)
        for w_down, w_up in zip(population.weight_down.values(), population.weight_up.values()):
            w_down.data[sl] = 0.
            w_up.data[sl] = 0.

    pop.reinit_islands_(zero_out, islands = 3, num_islands = 4)

    for k in pop.weight_down.keys():
        assert allclose(pop.weight_down[k][0:6], before[k][0:6])
        assert allclose(pop.weight_down[k][6:8], torch.zeros_like(pop.weight_down[k][6:8]))

def test_merge():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    lora_targets = [
        'attn_layers.layers.0.1.to_q',
        'attn_layers.layers.0.1.to_v'
    ]

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = lora_targets
    )

    out_ind0 = pop(x, individual = 0)

    pop.merge_(individual = 0)

    out_merged = model(x)

    assert allclose(out_ind0, out_merged, atol = 1e-5)

def test_evolution_step_and_route():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    with pop.route(individual = 0):
        out_route = model(x)

    out_direct = pop(x, individual = 0)
    assert allclose(out_route, out_direct)

    fitnesses = torch.tensor([1.0, 2.0, 0.5, 3.0])
    res = pop.select('deterministic', fitnesses, survive_frac = 0.5, elite_frac = 0.25)
    parents = pop.select_parents('tournament', fitnesses, num_children = len(res.culled), culled = res.culled)
    pop.crossover_('average', parents, res.culled, fitnesses = fitnesses)
    pop.mutate_('full_gaussian', individuals = res.culled)
    assert len(res.survivors) == 2

def test_select_and_merge_and_repopulate():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    base_out_before = model(x).clone()
    fitnesses = torch.tensor([1.0, 4.0, 2.0, 3.0])

    pop.select_and_merge(fitnesses = fitnesses, topk = 2, temperature = 1.)

    base_out_after_merge = model(x).clone()
    assert not allclose(base_out_before, base_out_after_merge)

    weight_down_before = {k: v.clone() for k, v in pop.weight_down.items()}

    pop.repopulate()

    for k, v in pop.weight_down.items():
        assert not allclose(v, weight_down_before[k])

    out_ind = pop(x, individual = 0)
    assert out_ind.shape == (1, 16, 1000)
    assert not allclose(out_ind, base_out_after_merge)

@param('kwargs_factory', [
    lambda fitnesses, best_idx: dict(fitnesses = fitnesses, topk = 1),
    lambda fitnesses, best_idx: dict(indices = best_idx),
    lambda fitnesses, best_idx: dict(indices = (best_idx,)),
    lambda fitnesses, best_idx: dict(indices = torch.tensor(best_idx)),
])
def test_select_and_merge_single_individual(kwargs_factory):
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([1.0, 4.0, 2.0, 3.0])
    best_idx = fitnesses.argmax().item()

    out_best_before_merge = pop(x, individual = best_idx)

    pop.select_and_merge(**kwargs_factory(fitnesses, best_idx))

    out_after_merge = model(x)
    assert allclose(out_best_before_merge, out_after_merge, atol = 1e-5)

def test_culled_excluded_from_parents():
    pop = Population(
        get_model(),
        pop_size = 8,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    result = pop.select('deterministic', fitnesses, survive_frac = 0.5)

    culled_set = set(result.culled.tolist())
    survivors_set = set(result.survivors.tolist())

    # test passing culled
    parents = pop.select_parents('tournament', fitnesses, num_children = len(result.culled), culled = result.culled)
    parent_set = set(parents.flatten().tolist())

    assert parent_set.isdisjoint(culled_set)
    assert parent_set.issubset(survivors_set)

    # test passing SelectionResult directly
    parents_res = pop.select_parents('tournament', fitnesses, num_children = len(result.culled), culled = result)
    parent_res_set = set(parents_res.flatten().tolist())

    assert parent_res_set.isdisjoint(culled_set)
    assert parent_res_set.issubset(survivors_set)

def test_select_and_merge_best():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.95, 0.3, 0.5])
    best_idx = fitnesses.argmax().item()

    out_best_before = pop(x, individual = best_idx)

    merged_model = pop.select_and_merge_best(fitnesses)

    # Returns the exact same model reference
    assert merged_model is model

    # Unhooked base model output matches best individual
    out_after = model(x)
    assert allclose(out_best_before, out_after, atol = 1e-5)
    assert len(pop._hooks) == 0

    # Invoking population forward afterwards auto-registers hooks
    out_pop_again = pop(x, individual = best_idx)
    assert len(pop._hooks) > 0
    assert out_pop_again.shape == (1, 16, 1000)

def test_select_and_merge_with_z_score():
    model = get_model()
    pop = Population(model, pop_size = 16, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])

    fitnesses = torch.randn(16) * 100.0 - 500.0
    merged_model = pop.select_and_merge(fitnesses = fitnesses, use_z_score = True)
    assert merged_model is not None

def test_regularize_():
    model = get_model()
    pop = Population(model, pop_size = 16, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])

    initial_norm = pop.weight_down['attn_layers_layers_0_1_to_q'].norm().item()
    pop.regularize_(weight_decay = 0.10)
    decayed_norm = pop.weight_down['attn_layers_layers_0_1_to_q'].norm().item()

    assert decayed_norm < initial_norm

    # test soft thresholding
    pop.weight_down['attn_layers_layers_0_1_to_q'].data[0, 0, 0] = 0.001
    pop.regularize_(soft_threshold = 0.002)
    assert pop.weight_down['attn_layers_layers_0_1_to_q'].data[0, 0, 0] == 0.0

def test_adapt_mutation_epsilon():
    eps = 0.10
    eps_up = Population.adapt_mutation_epsilon(eps, success_rate = 0.30, target_success_rate = 0.20, factor = 1.15)
    assert eps_up > eps

    eps_down = Population.adapt_mutation_epsilon(eps, success_rate = 0.10, target_success_rate = 0.20, factor = 1.15)
    assert eps_down < eps
