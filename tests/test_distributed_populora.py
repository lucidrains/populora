import time

import torch
import torch.distributed as dist
from torch import allclose, nn
from x_transformers import Decoder, TransformerWrapper

from populora import Coevolve, Population, distributed_rank, distributed_world_size, is_distributed, is_main_rank

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

    for gen in range(5):
        fitnesses = coevolve.step(distributed = True)

        # probes were split across ranks - rank 0 ran the proposer, rank 1 the solver

        assert calls == ['proposer'] if distributed_rank() == 0 else calls == ['solver']
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

if __name__ == '__main__':
    test_distributed_evolution()
    test_distributed_coevolve()
    print('distributed tests passed')
