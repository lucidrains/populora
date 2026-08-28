# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "tqdm",
#   "fire",
#   "torch-einops-utils",
#   "x-mlps-pytorch",
#   "gymnasium[box2d]",
#   "populora",
#   "moviepy",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

from __future__ import annotations

import os
import shutil
from collections import deque

import fire
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from x_mlps_pytorch import MLP

from populora import Population, interact_with_env, make_action

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def make_env(distribution: str, render_mode = None, **env_kwargs):
    if distribution == 'categorical':
        return gym.make('LunarLander-v3', render_mode = render_mode, **env_kwargs)

    return gym.make('LunarLanderContinuous-v3', render_mode = render_mode, **env_kwargs)

def make_record_env(distribution: str, video_folder: str, name_prefix: str):
    # record the raw env - continuous action spaces already emit (-1, 1)

    env = make_env(distribution, render_mode = 'rgb_array')
    return gym.wrappers.RecordVideo(env, video_folder = video_folder, name_prefix = name_prefix)
def evaluate_individual(
    env: gym.Env,
    pop: Population,
    individual_idx: int,
    action_fn,
) -> float:
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = action_fn(pop(state_tensor, individual = individual_idx).squeeze(0)).numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return total_reward

def validate_with_lunar(
    target_avg_reward: float = 100.0,
    avg_generations: int = 5,
    pop_size: int = 128,
    low_rank: int = 4,
    max_generations: int = 300,
    seed: int = 42,
    distribution: str = 'beta',  # 'categorical' | 'squashed_gaussian' | 'beta' (beta uses the mean-concentration reparam by default)
    sample_actions: bool = True,
    temperature: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 0.5,
    num_episodes: int = 2,
    es_every_generations: int = 25,
    es_topk: float | None = None,
    es_temperature: float = 1.0,
    queen_bee_mating: bool = False,
    num_elites: int = 1,
    weight_decay: float = 1e-3,
    soft_threshold: float = 0.0,
    adaptive_epsilon: bool = True,
    epsilon_init: float = 0.15,
    epsilon_tau: float | None = None,
    sigma_granularity: str = 'weight',
    epsilon: float = 0.15,
    survive_frac: float = 0.5,
    elite_frac: float = 0.25,
    crossover_type: str = 'extrapolative',
    mutation_type: str = 'full_gaussian',
    horizon: int = 1000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    shutil.rmtree('videos', ignore_errors = True)
    os.makedirs('videos', exist_ok = True)
    print('\nlogging videos of the best individual per generation to ./videos/\n')

    env = make_env(distribution)
    state_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    action_dim = env.action_space.shape[0] if continuous else env.action_space.n

    interactor = interact_with_env(env, seed = seed)

    output_dim = 2 * action_dim if continuous else action_dim

    pop = Population(
        MLP(state_dim, 64, 64, output_dim),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1.0', 'layers.2'],
        eval_seed = seed,
        adaptive_epsilon = adaptive_epsilon,
        epsilon_init = epsilon_init,
        epsilon_tau = epsilon_tau,
        sigma_granularity = sigma_granularity,
    )

    tau = pop.epsilon_tau if adaptive_epsilon else 'n/a'
    mut_eps = f'per-individual ({sigma_granularity})' if adaptive_epsilon else f'fixed {epsilon}'

    print(f'[PopuLoRA Lunar] pop_size: {pop_size} | low_rank: {low_rank} | generations: {max_generations} | episodes: {num_episodes} | distribution: {distribution} | horizon: {horizon}')
    print(f'  mutation: {mut_eps} | adaptive_epsilon: {adaptive_epsilon} | sigma_granularity: {sigma_granularity} | epsilon_init: {epsilon_init} | epsilon_tau: {tau}')
    print(f'  crossover: {crossover_type} | mutation_type: {mutation_type} | survive_frac: {survive_frac} | elite_frac: {elite_frac} | sample_actions: {sample_actions} | temperature: {temperature}')

    recent_rewards = deque(maxlen = avg_generations)
    pbar = tqdm(range(max_generations), desc = 'validating lunar lander')

    parent_selection_type = 'queen_bee' if queen_bee_mating else 'tournament'

    action_fn = make_action(
        distribution,
        sample = sample_actions,
        temperature = temperature,
        min_log_std = min_log_std,
        max_log_std = max_log_std,
    )

    for gen in pbar:
        # each individual scored on its mean return over `num_episodes` seeded episodes

        fitnesses = interactor.evaluate(
            pop,
            action = action_fn,
            horizon = horizon,
            num_episodes = num_episodes,
            seed = pop.eval_seed
        )

        result = pop.select('deterministic', fitnesses = fitnesses, survive_frac = survive_frac, elite_frac = elite_frac)
        best_reward = fitnesses.max().item()
        mean_reward = fitnesses.mean().item()
        recent_rewards.append(mean_reward)

        avg_recent = float(np.mean(recent_rewards))
        pbar.write(f'gen {gen:03d} | best_reward: {best_reward:6.1f} | mean_reward: {mean_reward:6.1f} | avg_last_{avg_generations}_generations: {avg_recent:6.1f}')

        best_idx = fitnesses.argmax().item()
        record_env = make_record_env(distribution, video_folder = 'videos', name_prefix = f'gen_{gen:03d}')
        evaluate_individual(record_env, pop, best_idx, action_fn = action_fn)
        record_env.close()

        if len(recent_rewards) >= avg_generations and avg_recent > target_avg_reward:
            print(f'\n✓ PopuLoRA LunarLander Validation Passed! Last {avg_generations} generations avg reward: {avg_recent:.2f} > {target_avg_reward}')
            env.close()
            return

        if es_every_generations > 0 and divisible_by(gen + 1, es_every_generations):
            pbar.write(f'[ES] generation {gen:03d} | select_and_merge + repopulate')
            pop.select_and_merge(fitnesses = fitnesses, topk = es_topk, temperature = es_temperature, use_z_score = True)
            pop.repopulate()
        else:
            parents = pop.select_parents(parent_selection_type, fitnesses = fitnesses, num_children = len(result.culled), num_elites = num_elites, culled = result.culled)

            if pop.adaptive_epsilon:
                # per-individual mutation rate: log-normal self-adaptation,
                # recombined from parents (geometric mean), per-param via sigma_granularity
                pop._sigma_recombine_(result.culled, parents)
                epsilon = pop._sigma_epsilon_(result.culled)

            pop.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses).mutate_(mutation_type, individuals = result.culled, epsilon = epsilon).regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

    env.close()
    assert False, f'LunarLander average cumulative reward failed to reach > {target_avg_reward} (got {avg_recent:.2f})'

if __name__ == '__main__':
    fire.Fire(validate_with_lunar)
