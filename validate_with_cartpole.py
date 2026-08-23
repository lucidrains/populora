# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "fire",
#   "gymnasium",
#   "x-mlps-pytorch",
#   "tqdm",
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
from tqdm import tqdm
from x_mlps_pytorch import MLP

from populora import interact_with_env, make_categorical_action
from populora.populora import exists

# one logit per action, sampled from the temperature-scaled categorical

def run_cartpole_experiment(
    target_avg_reward: float = 490.0,
    pop_size: int = 64,
    low_rank: int = 4,
    max_generations: int = 100,
    horizon: int = 1000,
    epsilon: float = 0.15,
    temperature: float = 1.0,
    dtype: str = 'float32',
    seed: int = 42,
    verbose: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v1')

    interactor = interact_with_env(env, seed = seed)
    pop = interactor.population(
        MLP(env.observation_space.shape[0], 32, env.action_space.n),
        pop_size = pop_size,
        low_rank = low_rank,
        eval_seed = seed,
        dtype = dtype,  # 'float32' | 'bfloat16' | 'float16' | 'float8_e4m3fn'
    )

    action = make_categorical_action(sample = True, temperature = temperature)

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
    seeds: list[int] = [10, 20, 30, 40, 50],
    seed: int | None = None,
    **kwargs
):
    # a single explicit --seed overrides the multi-seed sweep

    if exists(seed):
        seeds = [seed]

    results = [run_cartpole_experiment(seed = s, **kwargs) for s in tqdm(seeds, desc = 'validating cartpole')]

    solved_count = sum(1 for r in results if r["solved"])
    solve_gens = [r["gen_to_solve"] for r in results if r["solved"]]
    avg_gen = float(np.mean(solve_gens)) if solve_gens else float('nan')
    avg_best = float(np.mean([r["best_reward"] for r in results]))
    avg_mean = float(np.mean([r["mean_reward"] for r in results]))

    print(f"Results across {len(seeds)} seeds:")
    print(f"  Solved: {solved_count}/{len(seeds)} ({solved_count/len(seeds)*100:.1f}%)")
    print(f"  Avg Gens to Solve: {avg_gen:.1f}")
    print(f"  Avg Best Reward: {avg_best:.1f}")
    print(f"  Avg Mean Reward: {avg_mean:.1f}")

if __name__ == "__main__":
    fire.Fire(main)
