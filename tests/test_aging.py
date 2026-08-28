import pytest

import torch
from torch import nn

from einops import rearrange, repeat

from populora import Population

def make_pop(pop_size, hidden = 8, seed = None):
    model = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    return Population(model, pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'], seed = seed)

def test_age_bookkeeping():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    assert torch.equal(pop.ages, torch.zeros(8, dtype = torch.long))

    for gen in range(3):
        res = pop.evolve_(fitnesses, elite_frac = 0.)
        ages = pop.ages

        # survivors age one generation, the culled are reborn at zero
        assert torch.equal(ages[res.culled], torch.zeros(len(res.culled), dtype = torch.long))
        assert torch.equal(ages[res.survivors], torch.full((len(res.survivors),), gen + 1))

def test_max_age_programmed_death():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    # champion keeps serving while young
    res1 = pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25)
    assert set(res1.survivors.tolist()) == {0, 1, 2, 3}

    res2 = pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25)
    assert set(res2.culled.tolist()) == {4, 5, 6, 7}

    # at gen 2 the champions (age 2) are retired despite still being the fittest
    res2 = pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25)
    assert set(res2.culled.tolist()) == {0, 1, 2, 3}
    assert set(res2.survivors.tolist()) == {4, 5, 6, 7}
    assert set(res2.elites.tolist()) == {4, 5}

    # the newborns are age 1 again; the retired are reborn
    assert set(pop.ages[res2.survivors].tolist()) == {1}
    assert set(pop.ages[res2.culled].tolist()) == {0}

def test_elite_max_age_only():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    pop.evolve_(fitnesses, elite_max_age = 2, elite_frac = 0.25)
    pop.evolve_(fitnesses, elite_max_age = 2, elite_frac = 0.25)
    res = pop.evolve_(fitnesses, elite_max_age = 2, elite_frac = 0.25)

    # the top-fitness individuals (age 2) survive by merit but lose elite
    # protection - the elite slots go to the next-best young individuals
    assert set(res.elites.tolist()) == {4, 5}
    assert 0 in res.survivors.tolist()
    assert len(res.survivors) == 4

def test_aging_decay_soft():
    fitnesses = torch.tensor([8., 7., 6., 5., 4.6, 4.5, 4.4, 4.3])

    # decay 1.0 is a pure no-op
    pop_identity = make_pop(8)
    pop_identity.evolve_(fitnesses, aging_decay = 1.0, elite_frac = 0.)
    res = pop_identity.evolve_(fitnesses, aging_decay = 1.0, elite_frac = 0.)
    assert set(res.survivors.tolist()) == {0, 1, 2, 3}

    # a strong discount lets newborns displace veterans without any hard cutoff
    pop = make_pop(8)
    res = pop.evolve_(fitnesses, aging_decay = 0.5, elite_frac = 0.)
    assert set(res.survivors.tolist()) == {0, 1, 2, 3}  # no age yet - raw rankings

    res = pop.evolve_(fitnesses, aging_decay = 0.5, elite_frac = 0.)
    assert set(res.survivors.tolist()) == {4, 5, 6, 7}  # veterans discounted 2x

    res = pop.evolve_(fitnesses, aging_decay = 0.5, elite_frac = 0.)
    assert set(res.survivors.tolist()) == {0, 1, 2, 3}  # newborns now discounted

def test_ages_roundtrip_save_load():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    for _ in range(3):
        pop.evolve_(fitnesses)

    pkg = pop.state_dict_pkg()
    clone = make_pop(8)
    clone.load(pkg)
    assert torch.equal(clone.ages, pop.ages)

def test_migrate_reorders_ages():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.arange(8, dtype = torch.long))

    def reverse(fitnesses, num_islands, **kwargs):
        return torch.arange(pop.pop_size - 1, -1, -1, device = fitnesses.device)

    pop.migrate_(reverse, torch.arange(8, dtype = torch.float32), num_islands = 2)
    assert torch.equal(pop.ages, torch.arange(7, -1, -1, dtype = torch.long))

def test_reinit_resets_age():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.arange(8, dtype = torch.long))

    pop.reinit_individuals_([1, 5])
    assert pop.ages[[1, 5]].tolist() == [0, 0]

def test_aging_tiered_replacement():
    pop = make_pop(10)
    fitnesses = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    noop = lambda population, idx, **kwargs: None

    for gen in range(2):
        pop.evolve_(fitnesses, tiered = True, mutation_type = noop, elite_max_age = 2, gen = gen)

    res = pop.evolve_(fitnesses, tiered = True, mutation_type = noop, elite_max_age = 2, gen = 2)

    # the gen-0 keep tier (age 2) is demoted out of the keep tier into the
    # mutate tier - no one is forced-culled, the replace tier is untouched
    assert len(res.elites) == 0
    assert set(res.mid.tolist()) == set(range(7))
    assert set(res.culled.tolist()) == {7, 8, 9}

# edge cases - the aging knobs as pure selections filters, exercised directly
# against the current age state

def test_max_age_retires_everyone():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    # max_age 1 - nothing survives two consecutive generations
    res1 = pop.evolve_(fitnesses, max_age = 1, elite_frac = 0.25)
    assert set(res1.culled.tolist()) == {4, 5, 6, 7}

    res2 = pop.evolve_(fitnesses, max_age = 1, elite_frac = 0.25)
    assert set(res2.culled.tolist()) == {0, 1, 2, 3}
    assert set(res2.survivors.tolist()) == {4, 5, 6, 7}

def test_max_age_empty_pool_after_age():
    # everyone aged past the limit - the selection pool is empty, all are culled
    pop = make_pop(8)
    pop._ages.data.fill_(2)

    res = pop.select('deterministic', torch.arange(8., dtype = torch.float32), max_age = 1, survive_frac = 0.5)

    assert len(res.survivors) == 0
    assert len(res.elites) == 0
    assert set(res.culled.tolist()) == set(range(8))

def test_select_aging_groups():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.tensor([2, 2, 0, 0, 1, 1, 1, 1]))

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    res = pop.select('deterministic', fitnesses, max_age = 2, elite_frac = 0.25, num_groups = 2)

    # the island-aged champions retire; each island keeps its 2 best eligible
    assert set(res.survivors.tolist()) == {2, 3, 4, 5}
    assert set(res.culled.tolist()) == {0, 1, 6, 7}
    assert set(res.elites.tolist()) == {2, 4}

def test_max_age_without_elites():
    pop = make_pop(8)
    pop._ages.data.fill_(1)

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    res = pop.select('deterministic', fitnesses, max_age = 2, survive_frac = 0.5, elite_frac = 0.)

    assert set(res.survivors.tolist()) == {0, 1, 2, 3}
    assert len(res.elites) == 0

def test_max_age_survive_all():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.tensor([2, 2, 2, 2, 0, 0, 0, 0]))

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    res = pop.select('deterministic', fitnesses, max_age = 2, survive_frac = 1.0, elite_frac = 0.25)

    assert set(res.survivors.tolist()) == {4, 5, 6, 7}
    assert set(res.culled.tolist()) == {0, 1, 2, 3}

def test_max_age_elite_backfill_when_few_eligible():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.tensor([2, 2, 2, 2, 2, 0, 0, 0]))

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    res = pop.select('deterministic', fitnesses, max_age = 2, survive_frac = 0.5, elite_frac = 0.5)

    # only 3 eligible - elites truncate to the survivor budget and backfill
    assert set(res.survivors.tolist()) == {5, 6, 7}
    assert set(res.elites.tolist()) == {5, 6, 7}

def test_aging_combined_max_age_and_decay():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.tensor([2, 2, 0, 0, 1, 1, 1, 1]))

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    res = pop.select('deterministic', fitnesses, max_age = 2, aging_decay = 0.25, survive_frac = 0.5)

    # forced retirement wins even with a discount in play
    assert set(res.survivors.tolist()) == {2, 3, 4, 5}
    assert set(res.culled.tolist()) == {0, 1, 6, 7}

def test_aging_validation():
    pop = make_pop(8)
    fitnesses = torch.arange(8., dtype = torch.float32)

    with pytest.raises(AssertionError, match = 'max_age must be positive'):
        pop.evolve_(fitnesses, max_age = 0)
    with pytest.raises(AssertionError, match = 'elite_max_age must be positive'):
        pop.evolve_(fitnesses, elite_max_age = 0)
    with pytest.raises(AssertionError, match = 'aging_decay must be in'):
        pop.evolve_(fitnesses, aging_decay = 1.5)
    with pytest.raises(AssertionError, match = 'aging_decay must be in'):
        pop.evolve_(fitnesses, aging_decay = 0.)

def test_select_aging_does_not_bookkeep():
    pop = make_pop(8)
    pop._ages.data.copy_(torch.arange(8, dtype = torch.long))

    pop.select('deterministic', torch.arange(8., dtype = torch.float32), max_age = 4)

    # select is a pure ranking - only evolve_ advances the ages
    assert torch.equal(pop.ages, torch.arange(8, dtype = torch.long))

def test_tiered_aging_decay_axis():
    # the fitness axis is discounted with age, so an old top-fit individual
    # loses its keep-tier slot to the newborns
    pop = make_pop(10)
    pop._ages.data.zero_()
    pop._ages[0] = 1

    fitnesses = torch.tensor([1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(fitnesses, tiered = True, mutation_type = noop, aging_decay = 0.25, gen = 0)

    assert set(res.elites.tolist()) == {1, 2, 3}

def test_tiered_max_age_retire_all():
    pop = make_pop(10)
    pop._ages.data.fill_(5)

    fitnesses = torch.tensor([1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    noop = lambda population, idx, **kwargs: None

    res = pop.evolve_(fitnesses, tiered = True, mutation_type = noop, max_age = 4, gen = 0)

    # everyone is too old - the whole population lands in the replace tier
    assert set(res.culled.tolist()) == set(range(10))
    assert set(pop.ages.tolist()) == {0}

def test_from_checkpoint_roundtrip():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    for _ in range(3):
        pop.evolve_(fitnesses, max_age = 5)

    pkg = pop.state_dict_pkg()
    clone = Population.from_checkpoint(pkg, make_pop(8).model)

    assert torch.equal(clone.ages, pop.ages)

def test_repopulate_resets_ages_and_dtype():
    pop = make_pop(8)
    pop._ages.data.fill_(3)

    pop.repopulate_()
    assert torch.equal(pop.ages, torch.zeros(8, dtype = torch.long))
    assert pop.ages.dtype == torch.long

def test_retire_refill_reinit():
    pop = make_pop(8)
    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25)
    pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25)
    res = pop.evolve_(fitnesses, max_age = 2, elite_frac = 0.25, retire_refill = 'reinit')

    # the retired champions were re-randomized, not recombined from survivors
    assert set(res.culled.tolist()) == {0, 1, 2, 3}

    for key in pop.weight_down:
        for idx in res.culled.tolist():
            survivor_set = set(res.survivors.tolist())
            assert not any(
                torch.equal(pop.weight_down[key][idx], pop.weight_down[key][s])
                for s in survivor_set
            )

    assert set(pop.ages[res.culled].tolist()) == {0}

def test_retire_refill_es_islands():
    pop = make_pop(8, hidden = 4)
    pop._ages.data.zero_()

    fitnesses = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])

    # 2 islands - both islands' old champions retire and get ES-refilled
    pop.evolve_(fitnesses, max_age = 1, elite_frac = 0.25, num_groups = 2)
    res = pop.evolve_(fitnesses, max_age = 1, elite_frac = 0.25, num_groups = 2, retire_refill = 'es')

    assert set(res.survivors.tolist()) == {2, 3, 6, 7}
    assert set(res.culled.tolist()) == {0, 1, 4, 5}
    assert set(pop.ages[res.culled].tolist()) == {0}

    # the ES-refilled slots differ from the plain crossover children - the
    # refill resamples around the island mean with elite std noise
    for key in pop.weight_down:
        refills = pop.weight_down[key][res.culled]

        for s in res.survivors.tolist():
            w_s = pop.weight_down[key][s]
            assert not any(torch.allclose(w, w_s) for w in refills)

def test_retire_refill_validation():
    pop = make_pop(8)
    fitnesses = torch.arange(8., dtype = torch.float32)

    with pytest.raises(AssertionError, match = 'unknown retire_refill'):
        pop.evolve_(fitnesses, max_age = 3, retire_refill = 'bogus')
    with pytest.raises(AssertionError, match = 'retire_refill requires max_age'):
        pop.evolve_(fitnesses, retire_refill = 'es')

def test_e2e_island_rastrigin_es_refill():
    # the 4-island Rastrigin scenario - retirement + island-ES refill, with
    # migration; the population must improve and no one may outlive the cap
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    model[0].weight.data.zero_(); model[0].bias.data.zero_()
    model[2].weight.data.zero_(); model[2].bias.data.zero_()

    pop = Population(model, pop_size = 16, low_rank = 2, lora_targets = ['0', '2'])

    def points():
        return torch.stack([
            (w_up.float() * w_down.float()).sum(dim = (1, 2))
            for w_down, w_up in zip(pop.weight_down.values(), pop.weight_up.values())
        ], dim = -1) * 3.6

    def rastrigin(p):
        return 20. + p.square().sum(dim = -1) - 10. * torch.cos(2. * torch.pi * p).sum(dim = -1)

    def evaluate():
        fitnesses = -rastrigin(points())
        return fitnesses, float((-(fitnesses)).min())

    _, first = evaluate()
    best_ever = None

    for gen in range(15):
        fitnesses, best = evaluate()
        best_ever = min(best_ever, best) if best_ever is not None else best

        pop.evolve_(fitnesses, epsilon = 0.3, num_groups = 4, max_age = 4, retire_refill = 'es')
        pop.migrate_('fuss_roll', fitnesses, num_islands = 4, migrate_frac = 0.1, elite_frac = 0.25)

    assert best_ever < first                      # made progress on the Rastrigin plateau
    assert int(pop.ages.max()) <= 4               # age bookkeeping holds under retirement
    assert int(pop.ages.min()) == 0               # there are always newborns

# toy task - the population fits a curve on a fixed grid while the target
# switches every `phase_len` generations; aging cycles the slots so a stale
# elite can never lock the population onto a dead target

PHASE_LEN = 15
NUM_GENS = 45
GRID_POINTS = 33

def target_soft(x, phase):
    return torch.sin(torch.pi * x) if phase % 2 == 0 else torch.sin(2. * torch.pi * x)

def run_toy_task(seed, aging_kwargs):
    torch.manual_seed(seed)

    pop_size = 8
    pop = make_pop(pop_size, hidden = 8)

    grid = torch.linspace(-1., 1., GRID_POINTS).reshape(1, -1)
    grid_rep = repeat(grid, '1 n -> (s n) 1', s = pop_size)

    post_switch_means = []
    bests = []

    for gen in range(NUM_GENS):
        phase = gen // PHASE_LEN
        target = target_soft(grid, phase)

        preds = pop(grid_rep, all_individuals = True)
        preds = rearrange(preds, '(s n) 1 -> s n', s = pop_size)
        fitnesses = -((preds - target) ** 2).mean(dim = 1)

        post_switch_means.append((fitnesses.max().item(), fitnesses.mean().item()))
        bests.append(fitnesses.max().item())

        pop.evolve_(fitnesses, epsilon = 0.08, **aging_kwargs)

    return pop, post_switch_means

def test_aging_helps_on_shifting_target():
    # mean fitness over the first 5 generations after each switch, and final
    # best, averaged over seeds - the aging runs must beat the unaged one

    results = {'none': [], 'max_age': []}

    for seed in (0, 1, 2):
        _, hist_none = run_toy_task(seed, {})
        _, hist_age = run_toy_task(seed, dict(max_age = 6))

        def recovery(hist):
            sums = []
            for gen in range(PHASE_LEN, NUM_GENS):
                if gen % PHASE_LEN < 5:
                    sums.append(hist[gen][0])
            return sum(sums) / len(sums)

        results['none'].append(recovery(hist_none))
        results['max_age'].append(recovery(hist_age))

    assert sum(results['max_age']) > sum(results['none'])

    # and aging must not sacrifice the achievable best - its best-ever fitness
    # stays within reach of the unaged run's

    for seed in (0, 1, 2):
        _, hist_none = run_toy_task(seed, {})
        pop, hist = run_toy_task(seed, dict(max_age = 6))
        final_best = max(best for best, _ in hist)
        none_best = max(best for best, _ in hist_none)
        assert final_best > none_best - 0.1
