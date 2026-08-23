# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "fire",
#   "gymnasium",
#   "populora",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
import fire
from torch import nn

from populora import Memory, interact_with_env, make_categorical_action

# one logit per action - in memory mode the action callable receives only the
# tuple's first slot (the action distribution), so factories work unchanged

action = make_categorical_action(sample = False)

class GRUMemoryPolicy(nn.Module):
    # batch-first GRU whose hidden state is the carried memory - emits
    # (action_logits, hidden), fed back in as the 1st arg on the next step

    def __init__(self, obs_dim, act_dim, hidden = 32):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, batch_first = True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, mem, obs):
        _, mem = self.gru(obs.unsqueeze(1), mem.unsqueeze(0))
        mem = mem.squeeze(0)
        return self.head(mem), mem

class ActionDistAsMemoryPolicy(nn.Module):
    # contrived - the memory is the previous step's action distribution
    # concatenated onto the observation. the memory slot carries anything:
    # hidden states, kv caches, previous outputs, whatever the researcher wants

    def __init__(self, obs_dim, act_dim, hidden = 32):
        super().__init__()
        self.proj = nn.Linear(obs_dim + act_dim, hidden)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, mem, obs):
        x = torch.cat((obs, mem), dim = -1)
        h = torch.relu(self.proj(x))
        logits = self.head(h)
        return logits, logits

def run_cartpole_memory_experiment(
    policy_type: str = 'gru',
    target_avg_reward: float = 490.0,
    pop_size: int = 64,
    low_rank: int = 4,
    max_generations: int = 100,
    horizon: int = 1000,
    epsilon: float = 0.15,
    seed: int = 42,
    verbose: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if policy_type == 'gru':
        policy = Memory(
            GRUMemoryPolicy(obs_dim, act_dim),
            init_memory = torch.zeros(1, 32)
        )
    elif policy_type == 'action_dist':
        policy = Memory(
            ActionDistAsMemoryPolicy(obs_dim, act_dim),
            init_memory = torch.zeros(1, act_dim)
        )
    else:
        raise ValueError(f'unknown policy_type {policy_type}, choose from "gru" or "action_dist"')

    interactor = interact_with_env(env, seed = seed)
    pop = interactor.population(
        policy,
        pop_size = pop_size,
        low_rank = low_rank,
        eval_seed = seed,
    )

    _, history = interactor.evolve(
        pop,
        action = action,
        num_generations = max_generations,
        horizon = horizon,
        target_fitness = target_avg_reward,
        progress = verbose,
        return_history = True,
        evolve_kwargs = dict(
            survive_frac = 0.5,
            elite_frac = 0.25,
            crossover_type = 'extrapolative',
            epsilon = epsilon,
        ),
    )

    best_reward = max(h['best_fitness'] for h in history)
    gen_to_solve = next(
        (i + 1 for i, h in enumerate(history) if h['best_fitness'] >= target_avg_reward),
        max_generations,
    )

    return {
        "solved": best_reward >= target_avg_reward,
        "gen_to_solve": gen_to_solve,
        "best_reward": best_reward,
        "mean_reward": history[-1]['mean_fitness'],
    }

def main(
    policy_type: str = 'gru',
    seeds: list[int] = [10, 20, 30, 40, 50],
    **kwargs
):
    results = [
        run_cartpole_memory_experiment(policy_type = policy_type, seed = s, **kwargs)
        for s in seeds
    ]

    solved_count = sum(1 for r in results if r["solved"])
    solve_gens = [r["gen_to_solve"] for r in results if r["solved"]]
    avg_gen = float(np.mean(solve_gens)) if solve_gens else float('nan')
    avg_best = float(np.mean([r["best_reward"] for r in results]))
    avg_mean = float(np.mean([r["mean_reward"] for r in results]))

    print(f"Results across {len(seeds)} seeds ({policy_type}):")
    print(f"  Solved: {solved_count}/{len(seeds)} ({solved_count/len(seeds)*100:.1f}%)")
    print(f"  Avg Gens to Solve: {avg_gen:.1f}")
    print(f"  Avg Best Reward: {avg_best:.1f}")
    print(f"  Avg Mean Reward: {avg_mean:.1f}")

if __name__ == "__main__":
    fire.Fire(main)
