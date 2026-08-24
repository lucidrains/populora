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

import fire
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from x_mlps_pytorch import MLP

from populora import interact_with_env, make_action
from populora.populora import exists

# gymnasium's inverted pendulum - torque in [-2, 2], policy emits a squashed
# gaussian or unimodal beta (both (-1, 1) via --distribution), rescaled to the
# env bounds by passing `to_range` to interact_with_env - an example of
# rescaling a (-1, 1) policy output to an env's action range

def _flag(value: bool | str, default: bool = True) -> bool:
    # fire hands **kwargs through as strings, which are always truthy - coerce
    # so --flag False actually turns the flag off

    if isinstance(value, bool):
        return value

    return str(value).lower() not in ('false', '0', 'none')

def run_inverted_pendulum_experiment(
    target_avg_reward: float = -250.0,
    pop_size: int = 64,
    num_envs: int = 64,
    low_rank: int = 4,
    max_generations: int = 100,
    horizon: int = 1000,
    epsilon: float = 0.15,
    distribution: str = 'squashed_gaussian',  # 'squashed_gaussian' | 'beta'
    temperature: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 0.5,
    beta_rescale_neg_one_one: bool = True,
    mean_concentration: bool = True,
    num_episodes: int = 1,
    dtype: str = 'float32',
    seed: int = 42,
    verbose: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert distribution in ('squashed_gaussian', 'beta'), f'unknown continuous action distribution {distribution!r}'

    beta_rescale_neg_one_one = _flag(beta_rescale_neg_one_one)
    mean_concentration = _flag(mean_concentration)

    env = gym.make_vec('Pendulum-v1', num_envs = num_envs)
    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    # a beta with beta_rescale_neg_one_one = False emits on (0, 1) instead of
    # (-1, 1) - the interactor picks up the action fn's own from_range, so no
    # manual wiring is needed here

    interactor = interact_with_env(env, seed = seed, to_range = (-2., 2.))

    pop = interactor.population(
        MLP(obs_dim, 32, 2 * action_dim),
        pop_size = pop_size,
        low_rank = low_rank,
        eval_seed = seed,
        dtype = dtype,
    )

    action_fn = make_action(
        distribution,
        sample = True,
        temperature = temperature,
        min_log_std = min_log_std,
        max_log_std = max_log_std,
        beta_rescale_neg_one_one = beta_rescale_neg_one_one,
        mean_concentration = mean_concentration,
    )

    _, history = interactor.evolve(
        pop,
        action = action_fn,
        num_generations = max_generations,
        horizon = horizon,
        num_episodes = num_episodes,
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
    if exists(seed):
        seeds = [seed]

    results = [run_inverted_pendulum_experiment(seed = s, **kwargs) for s in tqdm(seeds, desc = 'validating inverted pendulum')]

    solved_count = sum(1 for r in results if r["solved"])
    solve_gens = [r["gen_to_solve"] for r in results if r["solved"]]
    avg_gen = float(np.mean(solve_gens)) if solve_gens else float('nan')
    avg_best = float(np.mean([r["best_reward"] for r in results]))
    avg_mean = float(np.mean([r["mean_reward"] for r in results]))

    print(f"Results across {len(seeds)} seeds:")
    print(f"  Solved: {solved_count}/{len(seeds)} ({solved_count/len(seeds)*100:.1f}%)")
    print(f"  Avg Gens to Solve: {avg_gen:.1f}" if solve_gens else "  Avg Gens to Solve: n/a (none solved)")
    print(f"  Avg Best Reward: {avg_best:.1f}")
    print(f"  Avg Mean Reward: {avg_mean:.1f}")

if __name__ == "__main__":
    fire.Fire(main)
