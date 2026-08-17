import torch
from torch import nn

from populora import HallOfFame, Population

def make_pop(pop_size):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return Population(model, pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

def test_add_champion_capacity_and_generations():
    pop = make_pop(8)
    fitnesses = torch.arange(8, dtype = torch.float32)

    hof = HallOfFame(capacity = 3)
    assert len(hof) == 0

    for gen in range(5):
        hof.add_champion(pop, fitnesses, generation = gen)

    assert len(hof) == 3
    assert [e.generation for e in hof.entries] == [2, 3, 4]

def test_sample_modes():
    pop = make_pop(4)
    fitnesses = torch.arange(4, dtype = torch.float32)

    hof = HallOfFame()

    for gen in range(6):
        hof.add_champion(pop, fitnesses, generation = gen)

    assert hof.sample(10, mode = 'uniform').tolist() == list(range(6))
    assert hof.sample(2, mode = 'latest').tolist() == [4, 5]

    uniform = hof.sample(3, mode = 'uniform')
    assert uniform.numel() == 3 and len(set(uniform.tolist())) == 3

def test_replay_loads_archived_weights():
    pop = make_pop(4)
    fitnesses = torch.arange(4, dtype = torch.float32)

    hof = HallOfFame()

    for gen in range(4):
        hof.add_champion(pop, fitnesses, generation = gen)

    replay_pop = make_pop(4)
    slots = hof.replay(replay_pop, [1, 3])
    assert slots.tolist() == [0, 1]

    for slot, entry_idx in ((0, 1), (1, 3)):
        entry = hof.entries[entry_idx]
        _, (wd, wu) = replay_pop.individual_weights(slot)

        for key in wd:
            assert torch.equal(wd[key], entry.weight_down[key])
            assert torch.equal(wu[key], entry.weight_up[key])

def test_probe_returns_none_when_empty():
    pop = make_pop(4)

    hof = HallOfFame()
    assert hof.probe(pop, torch.randn(3, 4), 2) is None

def test_probe_routes_through_archived_champions():
    pop = make_pop(4)
    fitnesses = torch.arange(4, dtype = torch.float32)

    hof = HallOfFame()

    for gen in range(4):
        hof.add_champion(pop, fitnesses, generation = gen)

    x = torch.randn(3, 4)
    k = 2

    seed = 123
    out = hof.probe(pop, x, k, generator = torch.Generator().manual_seed(seed))

    assert out.shape == (k, 3, 2)

    # the same champions, replayed and probed by hand over the same base model, must match

    replay_pop = make_pop(k)
    replay_pop.model.load_state_dict(pop.model.state_dict())
    slots = hof.replay(replay_pop, hof.sample(k, generator = torch.Generator().manual_seed(seed)))
    manual = replay_pop(x.repeat(k, 1), individuals = slots).reshape(k, *x.shape[:-1], -1)

    for i in range(k):
        assert torch.allclose(out[i], manual[i])
