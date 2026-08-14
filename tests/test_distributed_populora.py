import time

import torch
import torch.distributed as dist
from torch import allclose, nn
from x_transformers import Decoder, TransformerWrapper

from populora import Coevolve, Population, broadcast_object, distributed_rank, distributed_world_size, is_distributed, is_main_rank, sync_population

def test_distributed_evolution():
    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 16,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

    pop = Population(
        model,
        pop_size = 6,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    def eval_env(population, idx):
        time.sleep(0.01)
        return population(x, individual = idx).abs().mean()

    for gen in range(2):
        fitnesses = pop.evaluate_distributed(eval_env)

        # eval seed auto-synced across ranks

        assert pop.eval_seed > 0

        if is_main_rank():
            print(f'gen {gen:02d} | best: {fitnesses.max():.3f} | mean: {fitnesses.mean():.3f}')

        pop.evolve_(fitnesses)

    assert len(fitnesses) == 6

def test_distributed_coevolve():
    # coevolution distributes the probes across ranks round-robin by population -
    # with two populations on two ranks, each rank evaluates exactly one probe per
    # step, and the outputs are broadcast so every rank derives the same fitnesses

    if not is_distributed():
        return

    pop_size = 4

    proposer = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
    solver = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

    calls = []

    def probe_proposer(coevolve):
        calls.append('proposer')
        return coevolve.proposer(torch.randn(1, 1), all_individuals = True)

    def probe_solver(coevolve, proposer_outputs):
        calls.append('solver')
        return coevolve.solver(proposer_outputs.repeat(solver.pop_size, 1), all_individuals = True)

    def fitness_solver(solver_outputs, proposer_outputs):
        target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
        errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
        return -errors.mean(dim = 1)

    def fitness_proposer(proposer_outputs, solver_outputs):
        target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
        errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
        return errors.mean(dim = 0)

    coevolve = Coevolve(populations = dict(
        proposer = dict(population = proposer, probe = probe_proposer, fitness = fitness_proposer),
        solver = dict(population = solver, probe = probe_solver, fitness = fitness_solver)
    ))

    # probes are split across ranks round-robin by dependency order - each rank
    # runs exactly the probes it owns, whatever the populations are called

    order = coevolve._dependency_order()
    my_probes = [name for i, name in enumerate(order) if i % distributed_world_size() == distributed_rank()]

    for gen in range(5):
        fitnesses = coevolve.step(distributed = True)

        assert calls == my_probes
        calls.clear()

        # every rank derived the same fitnesses from the broadcast outputs

        local_total = fitnesses['proposer'].sum() + fitnesses['solver'].sum()
        dist.all_reduce(local_total, op = dist.ReduceOp.SUM)
        assert allclose(local_total / distributed_world_size(), fitnesses['proposer'].sum() + fitnesses['solver'].sum())

        if is_main_rank():
            print(f'gen {gen} | proposer best {fitnesses["proposer"].max().item():.3f} | solver best {fitnesses["solver"].max().item():.3f}')

    # probe noise is drawn under a preserved rng, so the evolution step stays in
    # sync - every rank ends with identical populations

    x = torch.randn(1, 1)
    out = coevolve.solver(x.repeat(solver.pop_size, 1), all_individuals = True)

    local_total = out.abs().sum()
    dist.all_reduce(local_total, op = dist.ReduceOp.SUM)
    assert allclose(local_total / distributed_world_size(), out.abs().sum())

def agree_across_ranks(value, tol = 1e-4):
    t = torch.tensor(float(value), dtype = torch.float64)
    dist.all_reduce(t, op = dist.ReduceOp.SUM)
    return abs(t.item() / distributed_world_size() - float(value)) < tol

def lora_weights(pop):
    return (*pop.weight_down.values(), *pop.weight_up.values())

def lora_stat(pop):
    return sum(w.pow(2).sum().item() for w in lora_weights(pop))

def test_sync_population_lora_only():
    # only lora weights are broadcast by default - perturb a non-src rank and make
    # sure the sync restores cross-rank agreement, base model untouched

    if not is_distributed():
        return

    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 16,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

    torch.manual_seed(0)
    pop = Population(
        model,
        pop_size = 6,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    if distributed_rank() != 0:
        for w in lora_weights(pop):
            w.data.add_(torch.randn_like(w.data) * 0.1)

    sync_population(pop)

    assert agree_across_ranks(lora_stat(pop))

    # opting in to sync_base_model also brings the base model in sync

    if distributed_rank() != 0:
        for param in pop.model.parameters():
            param.data.add_(torch.randn_like(param.data) * 0.1)

    sync_population(pop, sync_base_model = True)

    model_stat = sum(w.pow(2).sum().item() for w in pop.model.parameters())
    assert agree_across_ranks(model_stat)

def test_broadcast_object():
    if not is_distributed():
        return

    # nested dicts / lists / tuples of tensors broadcast through the pickled
    # fallback (constructed deterministically so every rank holds the expected
    # values)

    torch.manual_seed(0)
    payload = dict(
        a = torch.arange(8.).reshape(2, 4),
        b = [torch.randn(3), (torch.randn(2, 2),)],
        c = torch.tensor([1, 2, 3])
    )

    received = broadcast_object(payload if distributed_rank() == 0 else None, src = 0)

    assert isinstance(received, dict)
    assert allclose(received['a'], payload['a'])
    assert allclose(received['b'][0], payload['b'][0])
    assert allclose(received['b'][1][0], payload['b'][1][0])
    assert allclose(received['c'], payload['c'])

    # mixed tensor / non-tensor values

    exotic = dict(x = torch.randn(2), note = 'hello')

    received = broadcast_object(exotic if distributed_rank() == 0 else None, src = 0)

    assert received['note'] == 'hello'
    assert allclose(received['x'], exotic['x'])

def test_distributed_coevolve_nested_outputs():
    # probes may return dicts / lists of tensors - the distributed step must
    # reconstruct them identically on every rank, whatever the populations are
    # called (here: arbitrary names)

    if not is_distributed():
        return

    pop_size = 4

    proposer = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
    solver = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

    def probe_alpha(coevolve):
        return coevolve.alpha(torch.randn(1, 1), all_individuals = True)

    def probe_beta(coevolve, alpha_outputs):
        preds = coevolve.beta(alpha_outputs.repeat(solver.pop_size, 1), all_individuals = True)
        return dict(preds = preds, errors = [preds * 2, (preds + 1)])

    def fitness_beta(beta_outputs, alpha_outputs):
        preds = beta_outputs['preds']
        errors = preds.reshape(solver.pop_size, -1)
        return -errors.pow(2).mean(dim = 1)

    def fitness_alpha(alpha_outputs, beta_outputs):
        preds = beta_outputs['preds']
        errors = preds.reshape(solver.pop_size, -1)
        return errors.mean(dim = 0)

    coevolve = Coevolve(populations = dict(
        alpha = dict(population = proposer, probe = probe_alpha, fitness = fitness_alpha),
        beta = dict(population = solver, probe = probe_beta, fitness = fitness_beta)
    ))

    for gen in range(3):
        fitnesses = coevolve.step(distributed = True)

        outputs = coevolve.last_outputs
        assert isinstance(outputs['beta'], dict)
        assert 'errors' in outputs['beta']

        local_total = outputs['beta']['preds'].pow(2).sum() + outputs['beta']['errors'][0].pow(2).sum()
        dist.all_reduce(local_total, op = dist.ReduceOp.SUM)
        assert agree_across_ranks((local_total / distributed_world_size()).item())

    assert agree_across_ranks(lora_stat(solver))

if __name__ == '__main__':
    test_distributed_evolution()
    test_distributed_coevolve()
    test_sync_population_lora_only()
    test_broadcast_object()
    test_distributed_coevolve_nested_outputs()
    print('distributed tests passed')
