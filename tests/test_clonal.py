from __future__ import annotations

import torch
from torch import nn

from populora import ClonalState, Population, clonal_adapt_sigma_, clonal_reproduce_, clonal_reset_


def make_pop(pop_size = 4):
    torch.manual_seed(0)
    return Population(
        nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)),
        pop_size = pop_size,
        low_rank = 2,
        lora_targets = ['0', '2'],
    )


def test_clonal_reset_and_reproduce():
    pop = make_pop()
    clonal_reset_(pop)

    # sigma = 0 collapses every child onto the champion exactly
    clonal_reproduce_(pop, champion = 2, sigma = 0.)

    for key, w_down in pop.weight_down.items():
        w_up = pop.weight_up[key]
        for i in range(1, pop.pop_size):
            assert torch.equal(w_down[i], w_down[0])
            assert torch.equal(w_up[i], w_up[0])

    assert pop._ages.tolist() == [1, 0, 0, 0]


def test_clonal_adapt_sigma_burst_and_rule():
    state = ClonalState(sigma = 1.0, stall_patience = 1, burst = 4.0, sigma_max = 16.0)

    # children worse than the parent -> 1/5 rule shrinks sigma
    fits = torch.tensor([1.0, 0.0, 0.0, 0.0])
    clonal_adapt_sigma_(state, fits, parent_fitness = 1.0)
    assert state.sigma < 1.0

    # stagnation triggers the burst, then the 1/5 rule adjusts it
    before = state.sigma
    state.stall = 1
    clonal_adapt_sigma_(state, fits, parent_fitness = 1.0)
    assert state.sigma > before

    # a better child counts as success and grows sigma
    state = ClonalState(sigma = 1.0)
    fits = torch.tensor([1.0, 2.0, 0.0, 0.0])
    clonal_adapt_sigma_(state, fits, parent_fitness = 1.0)
    assert state.sigma > 1.0


def test_clonal_reproduce_subspace_centroid_and_consolidate_elites():
    pop = make_pop(pop_size = 8)
    clonal_reset_(pop, sigma = 3.0)
    fits = torch.arange(8, dtype = torch.float32)

    champ_d = {k: v[7].clone() for k, v in pop.weight_down.items()}
    clonal_reproduce_(pop, champion = 7, sigma = 0.5, fitnesses = fits)

    # slot 0 preserves the exact champion; slots 2.. are antithetic pairs around slot 1
    for key, w_down in pop.weight_down.items():
        assert torch.allclose(w_down[0], champ_d[key])
        assert torch.allclose(0.5 * (w_down[2] + w_down[3]), w_down[1], atol = 1e-5)

    w0_before = pop.model[0].weight.clone()
    pop.consolidate_elites_(fits, alpha = 0.35, trigger_ratio = 1.0)
    assert not torch.allclose(w0_before, pop.model[0].weight)
