import math

import pytest
param = pytest.mark.parametrize

import torch
import torch.nn as nn
from torch import allclose

from x_transformers import TransformerWrapper, Decoder
from einops import rearrange, repeat
from populora import Population, Populations, PopuLoRA, LoRA, Coevolve, HallOfFame, evaluate_population_distributed, register_mutation
from populora.populora import exists

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

def test_per_sample_individual_routing():
    model = get_model()

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (8, 16))
    ids = torch.tensor([0, 1, 2, 3, 1, 2, 3, 0])

    # per-sample individual ids aligned with batch dim (e.g. one individual per env)

    out = pop(x, individual = ids)
    assert out.shape == (*x.shape, 1000)

    # matches scalar routing per sample

    assert all(allclose(out[i], pop(x[i:i + 1], individual = idx)[0]) for i, idx in enumerate(ids.tolist()))

    # same via the `individuals` kwarg

    out_alt = pop(x, individuals = ids)
    assert allclose(out, out_alt)

    # 0-dim tensor index behaves like an int index

    assert allclose(pop(x[:1], individual = ids[0]), pop(x[:1], individual = int(ids[0])))

    # contiguous (p b) pattern with b > 1

    x_grouped = x[:4].repeat(4, 1)
    out_grouped = pop(x_grouped, individuals = [0, 1, 2, 3])
    assert all(allclose(out_grouped[g * 4 + b], pop(x[b:b + 1], individual = g)[0]) for g in range(4) for b in range(4))

def test_per_sample_individual_routing_gradient_flow():
    model = get_model()

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q'],
        requires_grad = False
    )

    x = torch.randint(0, 1000, (4, 16))
    ids = torch.tensor([0, 1, 2, 3])

    out = pop(x, individual = ids).mean()
    out.backward()

    # base trainable, adapters frozen

    to_q = model.get_submodule('attn_layers.layers.0.1.to_q')
    assert exists(to_q.weight.grad)
    assert not exists(pop.weight_down['attn_layers_layers_0_1_to_q'].grad)
    assert not exists(pop.weight_up['attn_layers_layers_0_1_to_q'].grad)

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

# tiered evolution - clone crossover (replacement without mixing) and the
# tiered 30/40/30 evolve mode

def test_clone_crossover():
    pop = Population(
        get_model(),
        pop_size = 6,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    parent_indices = torch.tensor([[1], [4]])
    child_indices = torch.tensor([2, 5])

    pop.crossover_('clone', parent_indices, child_indices)

    for w_down, w_up in zip(pop.weight_down.values(), pop.weight_up.values()):
        assert allclose(w_down[2], w_down[1])
        assert allclose(w_down[5], w_down[4])
        assert allclose(w_up[2], w_up[1])
        assert allclose(w_up[5], w_up[4])

def test_evolve_tiered():
    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    torch.manual_seed(7)
    fitnesses = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    before = {k: v.clone() for k, v in pop.weight_down.items()}

    # identity mutation - isolates the tiered selection and replacement scheme

    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(fitnesses, tiered = True, mutation_type = noop)

    # 30/40/30 split of pop_size 10

    assert len(res.elites) == 3
    assert len(res.mid) == 4
    assert len(res.culled) == 3

    # tiers partition the population

    assert set(res.elites.tolist()) | set(res.mid.tolist()) | set(res.culled.tolist()) == set(range(10))

    # top tier untouched

    for k in pop.weight_down.keys():
        assert allclose(pop.weight_down[k][res.elites], before[k][res.elites])

    # with an identity mutation, mid keeps its weights

    for k in pop.weight_down.keys():
        assert allclose(pop.weight_down[k][res.mid], before[k][res.mid])

    # culled replaced by exact copies of (distinct) top-tier agents

    w = pop.weight_down['attn_layers_layers_0_1_to_q']

    for c in res.culled.tolist():
        assert any(allclose(w[c], w[e]) for e in res.elites.tolist())

def test_evolve_tiered_burn_in():
    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    torch.manual_seed(3)
    fitnesses = torch.linspace(1.0, 0.0, 10)

    noop = lambda population, idx, **kwargs: None

    # gen 0: everyone eligible, mid + bottom mutated

    res0 = pop.evolve_(fitnesses, tiered = True, burn_in = 3, gen = 0, mutation_type = noop)
    touched0 = set(res0.mid.tolist()) | set(res0.culled.tolist())

    # gens 1-2: burn-in pauses everyone touched at gen 0, so no mid/bottom
    # individual may be touched again until gen 3

    for gen in (1, 2):
        res = pop.evolve_(fitnesses, tiered = True, burn_in = 3, gen = gen, mutation_type = noop)
        touched = set(res.mid.tolist()) | set(res.culled.tolist())
        assert touched0.isdisjoint(touched)

    # gen 3: paused individuals are eligible again

    res3 = pop.evolve_(fitnesses, tiered = True, burn_in = 3, gen = 3, mutation_type = noop)
    touched3 = set(res3.mid.tolist()) | set(res3.culled.tolist())
    assert touched0 & touched3

def test_evolve_tiered_groups():
    pop = Population(
        get_model(),
        pop_size = 12,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    torch.manual_seed(11)
    fitnesses = torch.randn(12)

    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(fitnesses, tiered = True, num_groups = 3, mutation_type = noop)

    # each island of 4 is tiered independently - 30/40/30 with integer rounding
    # gives 1 elite, 1 mid, 2 culled per island

    assert len(res.elites) == 3
    assert len(res.mid) == 3
    assert len(res.culled) == 6

    # culled clones come from elites of the same island

    w = pop.weight_down['attn_layers_layers_0_1_to_q']
    group_size = 4

    for g in range(3):
        island = range(g * group_size, (g + 1) * group_size)
        island_elites = [e for e in res.elites.tolist() if e in island]

        for c in res.culled.tolist():
            if c in island:
                assert any(allclose(w[c], w[e]) for e in island_elites)

def test_evolve_tiered_spec():
    # a custom tier spec - fractions and rules are both free

    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    torch.manual_seed(5)
    fitnesses = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    before = {k: v.clone() for k, v in pop.weight_down.items()}

    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(
        fitnesses,
        tiers = [(0.2, 'keep'), (0.2, 'mutate'), (0.6, 'replace')],
        mutation_type = noop
    )

    assert len(res.tiers) == 3
    assert len(res.elites) == 2
    assert len(res.mid) == 2
    assert len(res.culled) == 6

    # tiers partition the population

    assert set(res.elites.tolist()) | set(res.mid.tolist()) | set(res.culled.tolist()) == set(range(10))

    # keep untouched

    for k in pop.weight_down.keys():
        assert allclose(pop.weight_down[k][res.elites], before[k][res.elites])

    # replace copies the top tier exactly (noop mutation)

    w = pop.weight_down['attn_layers_layers_0_1_to_q']

    for c in res.culled.tolist():
        assert any(allclose(w[c], w[e]) for e in res.elites.tolist())

def test_evolve_tiered_strata_novelty():
    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    novelty = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # inverted vs fitness

    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(
        fitnesses,
        tiers = [(0.3, 'keep'), (0.7, 'replace')],
        strata = 'novelty',
        novelty = novelty,
        mutation_type = noop
    )

    # the keep tier follows novelty, not fitness

    assert set(res.elites.tolist()) == {7, 8, 9}
    assert set(res.culled.tolist()) == set(range(7))

    # novelty strata require the novelty tensor

    with pytest.raises(AssertionError, match = 'novelty'):
        pop.evolve_(fitnesses, tiers = [(0.3, 'keep'), (0.7, 'replace')], strata = 'novelty', mutation_type = noop)

def test_evolve_tiered_rules():
    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    torch.manual_seed(2)
    fitnesses = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    w = pop.weight_down['attn_layers_layers_0_1_to_q']

    noop = lambda population, idx, **kwargs: None

    # reinit - the whole bottom tier is freshly drawn

    before = w.clone()
    res = pop.evolve_(fitnesses, tiers = [(0.3, 'keep'), (0.7, 'reinit')], mutation_type = noop)

    for c in res.culled.tolist():
        assert not allclose(w[c], before[c])

    # crossover - children are the average of two higher-tier parents (noop mutation)

    res = pop.evolve_(fitnesses, tiers = [(0.3, 'keep'), (0.7, 'crossover')], mutation_type = noop)

    for c in res.culled.tolist():
        assert any(allclose(w[c], (w[a] + w[b]) / 2) for a in res.elites.tolist() for b in res.elites.tolist())

    # archive - replays archived individuals into the tier

    hof = HallOfFame()
    archived = []

    for i in range(4):
        hof.add(pop, i)
        _, (w_d, w_u) = pop.individual_weights(i)
        archived.append(w_d['attn_layers_layers_0_1_to_q'].clone())

    res = pop.evolve_(fitnesses, tiers = [(0.7, 'keep'), (0.3, 'archive')], mutation_type = noop, hof = hof)

    assert len(res.culled) == 3

    for c in res.culled.tolist():
        assert any(allclose(w[c], entry) for entry in archived)

    # archive without a hof raises

    with pytest.raises(AssertionError, match = 'hof'):
        pop.evolve_(fitnesses, tiers = [(0.7, 'keep'), (0.3, 'archive')], mutation_type = noop)

def test_evolve_tiered_validation():
    pop = Population(
        get_model(),
        pop_size = 10,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.randn(10)

    with pytest.raises(AssertionError, match = 'tier rule'):
        pop.evolve_(fitnesses, tiers = [(1.0, 'nonexistent')])

    with pytest.raises(AssertionError, match = 'sum to at most 1'):
        pop.evolve_(fitnesses, tiers = [(0.6, 'keep'), (0.6, 'mutate')])

    with pytest.raises(AssertionError, match = 'strata'):
        pop.evolve_(fitnesses, tiers = [(1.0, 'keep')], strata = 'bogus')

@param('num_parents', [2, 3])
def test_evolution_generation(num_parents):
    pop = Population(
        get_model(),
        pop_size = 6,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    fitnesses = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8, 0.6])

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

def test_parent_select_tournament_small_pool():
    from populora.populora import parent_select_tournament

    # clamp tournament to pool size for small islands

    fitnesses = torch.tensor([1.0, 2.0])

    parents = parent_select_tournament(
        fitnesses,
        num_children = 4,
        num_parents_per_child = 2,
        tournament_size = 3
    )

    assert parents.shape == (4, 2)
    assert parents.max().item() < fitnesses.shape[-1]

    # multi-group (island) case with one eligible parent per group

    fitnesses_grouped = torch.tensor([[1.0], [3.0]])
    parents_grouped = parent_select_tournament(
        fitnesses_grouped,
        num_children = 1,
        num_parents_per_child = 2,
        tournament_size = 3
    )

    assert parents_grouped.shape == (2, 1, 1)
    assert parents_grouped.max().item() < fitnesses_grouped.shape[-1]

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

def test_merge_best():
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

    merged_model = pop.merge_(fitnesses.argmax())

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
    assert exists(merged_model)

def test_shared_eval_seed():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(model, pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'], eval_seed = 42)

    def eval_env(population, idx):
        return population(x, individual = idx).abs().mean()

    for gen in range(3):
        pop.evaluate_distributed(eval_env)
        assert pop.eval_seed == 43 + gen

    # shared_seed off - eval seed untouched

    pop2 = Population(model, pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])

    for _ in range(3):
        pop2.evaluate_distributed(eval_env, shared_seed = False)

    assert pop2.eval_seed == 0

    # eval_seed None - shared seed disabled

    pop3 = Population(model, pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'], eval_seed = None)

    for _ in range(3):
        pop3.evaluate_distributed(eval_env)

    assert not exists(pop3.eval_seed)

def test_eval_seed_optional():
    # objects with _eval_seed = None are unaffected

    class MinimalPopulation:
        pop_size = 4
        device = 'cpu'
        _eval_seed = None

    pop = MinimalPopulation()
    fitnesses = evaluate_population_distributed(pop, lambda p, idx: float(idx))
    assert len(fitnesses) == 4

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

def test_save_and_load(tmp_path):
    pop = Population(get_model(), pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])
    pop.mutate_('full_gaussian', all_individuals = True, epsilon = 0.5)

    x = torch.randint(0, 256, (1, 16))
    out = pop(x, all_individuals = True)

    ckpt_path = tmp_path / 'ckpt.pt'
    pop.save(ckpt_path)

    # test from_checkpoint

    loaded_pop = Population.from_checkpoint(ckpt_path, get_model())
    assert torch.allclose(out, loaded_pop(x, all_individuals = True))

    # test in-place load

    pop_copy = Population(get_model(), pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])
    pop_copy.load(ckpt_path)

    assert torch.allclose(out, pop_copy(x, all_individuals = True))

def test_evolution_deterministic():
    x = torch.randint(0, 1000, (1, 16))

    def run_evolution(seed):
        torch.manual_seed(seed)

        pop = Population(get_model(), pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'], seed = seed)

        for _ in range(3):
            preds = pop(x, all_individuals = True)
            fitnesses = preds.abs().mean(dim = (1, 2))
            pop.evolve_(fitnesses, mutation_type = 'layer_selective_gaussian', survive_frac = 0.5)

        return {k: v.clone() for k, v in pop.weight_down.items()}

    a = run_evolution(42)
    b = run_evolution(42)
    c = run_evolution(43)

    for k in a.keys():
        assert allclose(a[k], b[k])

    assert any(not allclose(a[k], c[k]) for k in a.keys())

def test_to_lora():
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(model, pop_size = 4, low_rank = 4, lora_targets = ['attn_layers.layers.0.1.to_q'])

    out_pop = pop(x, individual = 0)

    # to_lora removes the population's hooks, so the delta is not double counted

    lora = pop.to_lora(0)
    out_lora = lora(x)

    assert not pop._hooks_registered
    assert allclose(out_pop, out_lora, atol = 1e-5)

    # gradient-trainable

    lora = pop.to_lora(1, requires_grad = True)

    key = 'attn_layers_layers_0_1_to_q'
    assert lora.weight_down[key].requires_grad

    loss = lora(x).float().sum()
    loss.backward()

    assert exists(lora.weight_down[key].grad)
    assert exists(lora.weight_up[key].grad)

def test_lora_merge_and_save_load(tmp_path):
    model = get_model()
    x = torch.randint(0, 1000, (1, 16))

    pop = Population(model, pop_size = 4, low_rank = 4, lora_targets = ['attn_layers.layers.0.1.to_q'])
    out_pop = pop(x, individual = 2)

    lora = pop.to_lora(2)

    # save / load roundtrip

    lora_path = tmp_path / 'lora.pt'
    lora.save(lora_path)
    lora.remove_hooks()

    lora_loaded = LoRA.from_checkpoint(lora_path, model)
    assert allclose(lora_loaded(x), out_pop, atol = 1e-5)

    # merge into base

    model_merged = lora_loaded.merge_()
    assert allclose(model_merged(x), out_pop, atol = 1e-5)

def test_save_individual_load_individual(tmp_path):
    key = 'attn_layers_layers_0_1_to_q'

    pop = Population(get_model(), pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])
    pop.mutate_('full_gaussian', individual = 0, epsilon = 0.5)

    path = tmp_path / 'individual.pt'
    pop.save_individual(path, individual = 0)

    pop2 = Population(get_model(), pop_size = 4, low_rank = 2, lora_targets = ['attn_layers.layers.0.1.to_q'])
    pop2.load_individual(path, individual = 0)

    assert allclose(pop.weight_down[key][0], pop2.weight_down[key][0])
    assert allclose(pop.weight_up[key][0], pop2.weight_up[key][0])
    assert not allclose(pop2.weight_down[key][1], pop2.weight_down[key][0])

# coevolution

def make_mlp(hidden_dim = 32, proposer = False):
    layers = [nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)]

    if proposer:
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)

def target_fn(x):
    return torch.sin(math.pi * x)

def make_coevolve(pop_size = 8, hidden_dim = 32, seed = 42):
    torch.manual_seed(seed)

    archive = []
    state = {}

    def probe_proposer(coevolve):
        z = torch.randn(1, 1)
        x = coevolve.proposer(z, all_individuals = True)  # (P, 1)

        if len(archive) > 0:
            archive_pts = torch.cat(archive, dim = 0)
            novelty_dist = (x.unsqueeze(1) - archive_pts.unsqueeze(0)).abs().min(dim = 1).values.squeeze(-1)
        else:
            novelty_dist = torch.full((x.shape[0],), float('inf'))

        state['novel'] = novelty_dist > 0.02
        state['novelty_dist'] = novelty_dist

        for i in range(x.shape[0]):
            if state['novel'][i]:
                archive.append(x[i:i + 1].clone())

        return x

    def probe_solver(coevolve, proposer_outputs):
        x_eval = torch.cat(archive + [proposer_outputs], dim = 0)
        state['x_eval'] = x_eval

        x_rep = repeat(x_eval, 'a 1 -> (s a) 1', s = pop_size)
        preds = coevolve.solver(x_rep, all_individuals = True)
        return rearrange(preds, '(s a) 1 -> s a', s = pop_size)  # (S, A + P)

    def fitness_solver(solver_outputs):
        errors = (solver_outputs - target_fn(state['x_eval']).T) ** 2  # (S, A + P)

        return -errors.mean(dim = 1)  # (S,) - accuracy over the archive

    def fitness_proposer(proposer_outputs, solver_outputs):
        x = proposer_outputs

        errors = (solver_outputs - target_fn(state['x_eval']).T) ** 2

        induced = errors[:, -x.shape[0]:].mean(dim = 0)
        induced = induced * state['novel'].float()  # only genuinely new proposals earn induced error credit

        novelty = 1.0 * state['novelty_dist'].clamp(max = 1.0)
        spread = 0.5 * (x.unsqueeze(1) - x.unsqueeze(0)).abs().mean(dim = 1).squeeze(-1)

        return induced + novelty + spread  # (P,) - error induced + novelty + spread

    return Coevolve(
        populations = dict(
            proposer = dict(
                population = Population(make_mlp(hidden_dim, proposer = True), pop_size = pop_size, low_rank = 4, lora_targets = ['0', '2']),
                probe = probe_proposer,
                fitness = fitness_proposer
            ),
            solver = dict(
                population = Population(make_mlp(hidden_dim), pop_size = pop_size, low_rank = 4, lora_targets = ['0', '2']),
                probe = probe_solver,
                fitness = fitness_solver
            )
        ),
        evolve_kwargs = dict(
            proposer = dict(survive_frac = 0.5, elite_frac = 0.2),
            solver = dict(survive_frac = 0.5, elite_frac = 0.2)
        )
    )

def evaluate_grid(coevolve, pop_size, grid = None):
    grid = torch.linspace(-1, 1, 129).reshape(1, -1) if grid is None else grid

    grid_rep = repeat(grid, '1 n -> (s n) 1', s = pop_size)
    preds = coevolve.solver(grid_rep, all_individuals = True)
    preds = rearrange(preds, '(s n) 1 -> s n', s = pop_size)

    return (preds - target_fn(grid)) ** 2  # (S, n)

def test_coevolve_api():
    pop_size = 8
    coevolve = make_coevolve(pop_size = pop_size)

    assert set(coevolve.populations.keys()) == {'proposer', 'solver'}
    assert len(coevolve) == 2
    assert coevolve.generation == 0
    assert coevolve.history['proposer'] == []

    fitnesses = coevolve.step()

    assert coevolve.generation == 1
    assert set(fitnesses.keys()) == {'proposer', 'solver'}
    assert fitnesses['proposer'].shape == (pop_size,)
    assert fitnesses['solver'].shape == (pop_size,)
    assert len(coevolve.history['proposer']) == 1
    assert len(coevolve.history['solver']) == 1

    # fitnesses derived from the other population's outputs - a solver facing a fresh
    # proposer cannot have positive fitness, and a fresh proposer induces error

    assert fitnesses['solver'].max() < 0.
    assert fitnesses['proposer'].max() > 0.

    for _ in range(4):
        fitnesses = coevolve.step()

    assert coevolve.generation == 5
    assert len(coevolve.history['solver']) == 5

    # per-step fitness function override

    fitnesses = coevolve.step(fitness_fns = dict(
        proposer = lambda proposer_outputs: torch.zeros(pop_size),
        solver = lambda solver_outputs: torch.zeros(pop_size)
    ))

    assert coevolve.generation == 6

    # attribute and indexing access to populations, individual forward

    x = coevolve.proposer(torch.randn(1, 1), individual = 0)
    assert x.shape == (1, 1)
    assert coevolve['proposer'] is coevolve.populations['proposer']
    assert coevolve.proposer is coevolve.populations['proposer']

def test_coevolve_signature_di():
    # dependencies are detected from the function signatures - parameters named after
    # populations receive their outputs, computed once per step in dependency order;
    # `coevolve`, `generation`, and `pop` are injected from the container

    calls = []

    def probe_A(coevolve, generation):
        calls.append(('probe_A', generation))
        return coevolve.A(torch.randn(1, 1), all_individuals = True)

    def probe_B(coevolve, A_outputs):
        calls.append(('probe_B', A_outputs.shape))
        return coevolve.B(torch.randn(1, 1), all_individuals = True)

    def fitness_A(A_outputs, B_outputs, pop):
        calls.append(('fitness_A', pop.pop_size))
        return torch.randn(pop.pop_size)

    def fitness_B(B_outputs, A_outputs, generation):
        calls.append(('fitness_B', generation))
        return torch.randn(B_outputs.shape[0])

    coevolve = Coevolve(
        populations = dict(
            A = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = probe_A, fitness = fitness_A),
            B = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = probe_B, fitness = fitness_B)
        )
    )

    fitnesses = coevolve.step()

    assert set(fitnesses.keys()) == {'A', 'B'}
    assert fitnesses['A'].shape == (8,)
    assert fitnesses['B'].shape == (8,)

    # probes run in dependency order (B's probe names A's outputs, so A is probed
    # first), each population probed exactly once, then the fitness functions

    assert calls[0] == ('probe_A', 0)
    assert calls[1] == ('probe_B', (8, 1))
    assert {call[0] for call in calls[2:]} == {'fitness_A', 'fitness_B'}

    assert calls[1][0] == 'probe_B'

    # generation advanced for the next step

    coevolve.step()

    assert calls[4][0] == 'probe_A'
    assert calls[4][1] == 1

def test_coevolve_arms_race():
    pop_size = 8
    coevolve = make_coevolve(pop_size = pop_size, seed = 0)

    initial_grid_mse = evaluate_grid(coevolve, pop_size).mean(dim = 1).min().item()
    hardnesses = []

    for gen in range(150):
        eps = 0.15 * (0.995 ** gen)
        fitnesses = coevolve.step(evolve_kwargs = dict(
            proposer = dict(epsilon = eps),
            solver = dict(epsilon = eps)
        ))
        hardnesses.append(fitnesses['proposer'].max().item())

    final_grid_mse = evaluate_grid(coevolve, pop_size).mean(dim = 1).min().item()

    # the arms race should drive the solver to a good global fit ...

    assert final_grid_mse < initial_grid_mse / 2, f'no improvement: {initial_grid_mse} -> {final_grid_mse}'
    assert final_grid_mse < 0.1, f'coevolution did not converge, final grid mse {final_grid_mse}'

    # ... while proposers keep finding weak spots (game does not collapse)

    assert max(hardnesses[-10:]) > 0.02, 'proposer population collapsed, no hardness induced'

def test_coevolve_uneven_pop_sizes():
    def probe_solver(coevolve, proposer_outputs):
        # each of the 8 solvers sees all 4 proposed points

        preds = coevolve.solver(repeat(proposer_outputs, 'p 1 -> (s p) 1', s = 8), all_individuals = True)
        preds = rearrange(preds, '(s p) 1 -> s p', s = 8)
        assert preds.shape == (8, 4)
        return preds

    coevolve = Coevolve(
        populations = dict(
            proposer = dict(
                population = Population(make_mlp(proposer = True), pop_size = 4, low_rank = 2, lora_targets = ['0', '2']),
                probe = lambda coevolve: coevolve.proposer(torch.randn(1, 1), all_individuals = True),
                fitness = lambda proposer_outputs, solver_outputs: torch.randn(4)
            ),
            solver = dict(
                population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']),
                probe = probe_solver,
                fitness = lambda solver_outputs: torch.randn(8)
            )
        )
    )

    fitnesses = coevolve.step()
    assert fitnesses['proposer'].shape == (4,)
    assert fitnesses['solver'].shape == (8,)

def test_coevolve_save_load():
    pop_size = 4
    coevolve = make_coevolve(pop_size = pop_size, seed = 0)

    coevolve.step()

    x = torch.randn(1, 1)
    preds_before = coevolve.proposer(x, all_individuals = True)

    coevolve.save('/tmp/coevolve_test.pt')

    coevolve2 = make_coevolve(pop_size = pop_size, seed = 1)
    coevolve2.load('/tmp/coevolve_test.pt')

    preds_after = coevolve2.proposer(x, all_individuals = True)

    assert allclose(preds_before, preds_after)

def test_coevolve_save_load_history():
    # generation counter and per-population history are checkpointed alongside
    # the weights, so an evolution can be resumed

    pop_size = 4
    coevolve = make_coevolve(pop_size = pop_size, seed = 0)

    for _ in range(3):
        coevolve.step()

    coevolve.save('/tmp/coevolve_history_test.pt')

    coevolve2 = make_coevolve(pop_size = pop_size, seed = 1)
    coevolve2.load('/tmp/coevolve_history_test.pt')

    assert coevolve2.generation == 3
    assert coevolve2.history == coevolve.history
    assert len(coevolve2.history['proposer']) == 3

def test_coevolve_missing_probe():
    # outputs of a population are requested by name but it has no probe - caught at
    # registration time

    with pytest.raises(AssertionError, match = 'has no probe'):
        Coevolve(
            populations = dict(
                A = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), fitness = lambda A_outputs: torch.randn(8)),
                B = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = lambda: torch.randn(8), fitness = lambda A_outputs, B_outputs: torch.randn(8))
            )
        )

def test_coevolve_missing_fitness():
    coevolve = Coevolve(
        populations = dict(
            proposer = dict(population = Population(make_mlp(proposer = True), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = lambda: torch.randn(8)),
            solver = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = lambda: torch.randn(8), fitness = lambda: torch.randn(8))
        )
    )

    with pytest.raises(AssertionError):
        coevolve.step()

def test_coevolve_unknown_parameter():
    with pytest.raises(TypeError, match = 'cannot resolve parameter'):
        Coevolve(
            populations = dict(
                A = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = lambda: torch.randn(8), fitness = lambda nonsense: torch.randn(8))
            )
        )

    # the same applies to probes

    with pytest.raises(TypeError, match = 'cannot resolve parameter'):
        Coevolve(
            populations = dict(
                proposer = dict(
                    population = Population(make_mlp(proposer = True), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']),
                    probe = lambda solver_ouputs: torch.randn(8),
                    fitness = lambda proposer_outputs: torch.randn(8)
                )
            )
        )

def test_coevolve_circular_probe_dependencies():
    # probes that transitively depend on their own outputs raise at registration,
    # with the full cycle path reported

    def make(populations):
        return Coevolve(populations = populations)

    def pop(name, probe, fitness):
        return dict(
            population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']),
            probe = probe,
            fitness = fitness
        )

    # two-population cycle

    with pytest.raises(RuntimeError, match = 'A -> B -> A'):
        make(dict(
            A = pop('A', lambda B_outputs: torch.randn(8), lambda A_outputs: torch.randn(8)),
            B = pop('B', lambda A_outputs: torch.randn(8), lambda B_outputs: torch.randn(8))
        ))

    # three-population cycle reports the whole path, not just the re-entered node

    with pytest.raises(RuntimeError, match = 'A -> B -> C -> A'):
        make(dict(
            A = pop('A', lambda B_outputs: torch.randn(8), lambda A_outputs: torch.randn(8)),
            B = pop('B', lambda C_outputs: torch.randn(8), lambda B_outputs: torch.randn(8)),
            C = pop('C', lambda A_outputs: torch.randn(8), lambda C_outputs: torch.randn(8))
        ))

    # a probe depending on its own outputs directly

    with pytest.raises(RuntimeError, match = 'A -> A'):
        make(dict(
            A = pop('A', lambda A_outputs: torch.randn(8), lambda A_outputs: torch.randn(8))
        ))

    # fitness circles stay allowed - every population is probed before any fitness is
    # derived, so fitness_A may score B's outputs and fitness_B may score A's

    def probe_A(coevolve):
        return coevolve.A(torch.randn(1, 1), all_individuals = True)

    def probe_B(coevolve):
        return coevolve.B(torch.randn(1, 1), all_individuals = True)

    coevolve = make(dict(
        A = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = probe_A, fitness = lambda A_outputs, B_outputs: torch.randn(8)),
        B = dict(population = Population(make_mlp(), pop_size = 8, low_rank = 2, lora_targets = ['0', '2']), probe = probe_B, fitness = lambda A_outputs, B_outputs: torch.randn(8))
    ))

    coevolve.step()

def test_coevolve_chain_three_populations():
    # three populations in a chain - proposer -> solver -> judge. each population's
    # probe consumes the previous one's outputs, so probes run proposer, solver,
    # judge in order, and every fitness sees every population's outputs

    pop_size = 6
    torch.manual_seed(0)

    proposer = Population(make_mlp(16, proposer = True), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
    solver = Population(make_mlp(16), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
    judge = Population(make_mlp(16), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

    calls = []

    def probe_proposer(coevolve):
        calls.append('probe_proposer')
        return coevolve.proposer(torch.randn(1, 1), all_individuals = True)  # (P, 1)

    def probe_solver(coevolve, proposer_outputs):
        calls.append('probe_solver')
        return coevolve.solver(proposer_outputs.repeat(solver.pop_size, 1), all_individuals = True)  # (S * P, 1)

    def probe_judge(coevolve, solver_outputs):
        calls.append('probe_judge')
        return coevolve.judge(solver_outputs, all_individuals = True)  # (J * S * P, 1)

    def fitness_judge(judge_outputs):
        return torch.zeros(pop_size)

    def fitness_solver(solver_outputs):
        return torch.zeros(pop_size)

    def fitness_proposer(proposer_outputs, solver_outputs, judge_outputs):
        assert proposer_outputs.shape == (pop_size, 1)
        assert solver_outputs.shape == (pop_size * pop_size, 1)
        assert judge_outputs.shape == (pop_size * pop_size, 1)
        return torch.zeros(pop_size)

    coevolve = Coevolve(
        populations = dict(
            proposer = dict(population = proposer, probe = probe_proposer, fitness = fitness_proposer),
            solver = dict(population = solver, probe = probe_solver, fitness = fitness_solver),
            judge = dict(population = judge, probe = probe_judge, fitness = fitness_judge)
        )
    )

    coevolve.step()

    # probes run down the chain, each exactly once, before the fitnesses

    assert calls == ['probe_proposer', 'probe_solver', 'probe_judge']

    # every fitness function sees every population's outputs (a fitness circle closes
    # over the chain - the proposer is scored by the judge's verdicts)

    fitnesses = coevolve.step(fitness_fns = dict(
        proposer = lambda proposer_outputs, solver_outputs, judge_outputs: torch.ones(pop_size),
        solver = lambda proposer_outputs, solver_outputs, judge_outputs: torch.ones(pop_size),
        judge = lambda proposer_outputs, solver_outputs, judge_outputs: torch.ones(pop_size)
    ))

    assert fitnesses['proposer'].shape == (pop_size,)
    assert coevolve.generation == 2
