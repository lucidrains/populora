# /// script
# dependencies = [
#   "torch",
#   "fire",
#   "populora",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

# task: sequential parity with resets - stream of 0/1 symbols plus 2 (clear); predict the
# running xor of 1-bits since the last reset. two Memory-wrapped policies: gru (GRU cell
# written as Linear layers, evolved like everything else) and explicit ({-1,+1} register)

from __future__ import annotations

import fire
import torch
from torch import nn

from populora import Memory, Population, evolve, rollout

# task

def make_sequences(batch, seq_len, device):
    n = torch.randint(0, 4, (batch, seq_len), device = device)
    syms = torch.where(n < 2, n, torch.ones_like(n) * 2)  # 0, 1 are bits; 2 and 3 both reset
    parity = torch.zeros(batch, seq_len, device = device)
    p = torch.zeros(batch, device = device)
    for t in range(seq_len):
        p = torch.where(syms[:, t] == 2, torch.zeros_like(p), p)
        p = (p + (syms[:, t] == 1).float()) % 2
        parity[:, t] = p
    return syms.float(), parity

# policies

class GRUCellPolicy(nn.Module):
    def __init__(self, obs_dim, hidden, act_dim):
        super().__init__()
        self.i2h = nn.Linear(obs_dim, hidden * 3)
        self.h2h = nn.Linear(hidden, hidden * 3)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, mem, obs):
        gates = self.i2h(obs) + self.h2h(mem)
        r, z, n = gates.chunk(3, dim = -1)
        r, z = r.sigmoid(), z.sigmoid()
        n = torch.tanh(n)
        mem = (1. - z) * n + z * mem
        return self.head(mem), mem

class ExplicitMemoryPolicy(nn.Module):
    def __init__(self, obs_dim, hidden, act_dim):
        super().__init__()
        self.write = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.forget = nn.Sequential(
            nn.Linear(obs_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.read = nn.Sequential(
            nn.Linear(obs_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, mem, obs):
        x = torch.cat((obs, mem), dim = -1)
        mem = torch.where(self.forget(x) > 0, torch.ones_like(mem), mem)
        mem = torch.where(self.write(obs) > 0, -mem, mem)
        return self.read(torch.cat((obs, mem), dim = -1)), mem

# training - the fitness scores the whole population at once: fresh random
# sequences, one routed forward per timestep (memory threaded by `rollout`),
# per-step accuracy meaned over each individual's share of the batch

def run(
    policy: str = 'gru',
    pop_size: int = 128,
    low_rank: int = 4,
    hidden: int = 16,
    seq_len: int = 32,
    num_seqs: int = 32,
    max_generations: int = 300,
    target_acc: float = 0.995,
    seed: int = 42,
):
    torch.manual_seed(seed)

    if policy == 'gru':
        net = GRUCellPolicy(1, hidden, 1)
        init_memory = torch.zeros(1, hidden)
    else:
        net = ExplicitMemoryPolicy(1, hidden, 1)
        init_memory = torch.ones(1, hidden)

    pop = Population(
        Memory(net, init_memory = init_memory),
        pop_size = pop_size,
        low_rank = low_rank,
    )

    def fitness(pop):
        bits, parity = make_sequences(pop.pop_size * num_seqs, seq_len, pop.device)
        logits = rollout(pop, bits, all_individuals = True)
        correct = (logits > 0) == (parity > 0)
        return correct.float().mean(dim = 1).reshape(pop.pop_size, num_seqs).mean(dim = 1)

    model, history = evolve(
        pop,
        fitness,
        num_generations = max_generations,
        target_fitness = target_acc,
        patience = 3,
        tiered = True,
        return_history = True,
    )

    gen_to_solve = next(
        (i + 1 for i, h in enumerate(history) if h['best_fitness'] >= target_acc),
        max_generations,
    )
    best_acc = max(h['best_fitness'] for h in history)

    # generalization - the merged policy re-rolled on longer sequences it never
    # saw. for the explicit policy this only passes if the evolved state
    # machine really does toggle on every 1 bit and clear on every 2

    bits, parity = make_sequences(pop_size * 4, seq_len * 4, model.device)
    logits = rollout(model, bits)
    long_acc = ((logits > 0) == (parity > 0)).float().mean().item()

    solved = gen_to_solve < max_generations
    print(f'{policy} | {"solved" if solved else "not solved"} in {gen_to_solve} generations | best acc {best_acc:.3f} | acc on {seq_len * 4}-bit sequences {long_acc:.3f}')

def main(
    policy: str = 'gru',
    seeds: list[int] = [42],
    **kwargs
):
    seed = kwargs.pop('seed', None)
    seeds = [seed] if seed is not None else seeds

    for s in seeds:
        run(policy = policy, seed = s, **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
