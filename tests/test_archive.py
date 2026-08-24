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

def test_entries_stored_on_cpu_and_replayed_to_device():
    # archived champions live on cpu, so long runs do not accumulate adapter
    # memory on the accelerator - replay moves them back through load_individual

    pop = make_pop(4)
    fitnesses = torch.arange(4, dtype = torch.float32)

    hof = HallOfFame()

    for gen in range(3):
        hof.add_champion(pop, fitnesses, generation = gen)

    for entry in hof.entries:
        assert all(w.device.type == 'cpu' for w in entry.weight_down.values())
        assert all(w.device.type == 'cpu' for w in entry.weight_up.values())

    replay_pop = make_pop(2)
    slots = hof.replay(replay_pop, [0, 2])

    for slot, entry_idx in zip(slots.tolist(), [0, 2]):
        entry = hof.entries[entry_idx]
        _, (wd, wu) = replay_pop.individual_weights(slot)

        for key in wd:
            assert wd[key].device.type == replay_pop.device.type
            assert torch.equal(wd[key], entry.weight_down[key].to(wd[key].device))

def test_probe_replaces_undersized_replay_without_hook_leak():
    # probing with a growing k replaces the internal replay population - the
    # replaced one must remove its hooks from the shared base model, or they
    # pile up forever

    pop = make_pop(4)
    fitnesses = torch.arange(4, dtype = torch.float32)

    hof = HallOfFame()

    for gen in range(4):
        hof.add_champion(pop, fitnesses, generation = gen)

    x = torch.randn(2, 4)
    base = pop.model

    def total_forward_hooks(module):
        return sum(len(m._forward_hooks) for m in base.modules())

    hooks_before = total_forward_hooks(base)

    for k in (1, 2, 3):
        assert hof.probe(pop, x, k).shape == (k, 2, 2)

    assert len(hof._replay._hooks) > 0
    assert total_forward_hooks(base) == hooks_before + len(hof._replay.lora_targets)

    # a different base population gets its own replay population - the old
    # one is detached again

    other = make_pop(4)
    assert hof.probe(other, x, 2).shape == (2, 2, 2)

    assert total_forward_hooks(base) == hooks_before
