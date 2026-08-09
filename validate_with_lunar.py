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
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import fire
from x_mlps_pytorch import MLP

from populora import Population

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def evaluate_individual(
    env: gym.Env,
    pop: Population,
    individual_idx: int,
    sample_actions: bool = True,
    action_std: float = 0.10
) -> float:
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean_action = pop(state_tensor, individual = individual_idx).squeeze(0).tanh()
            if sample_actions and action_std > 0.:
                action = torch.normal(mean_action, action_std).clamp(-1.0, 1.0).numpy()
            else:
                action = mean_action.numpy()

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
    sample_actions: bool = True,
    action_std: float = 0.10,
    es_every_generations: int = 25,
    es_topk: int | float | None = None,
    es_temperature: float = 1.0,
    queen_bee_mating: bool = False,
    num_elites: int = 1,
    weight_decay: float = 1e-3,
    soft_threshold: float = 0.0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    shutil.rmtree('videos', ignore_errors = True)
    os.makedirs('videos', exist_ok = True)
    print('\nlogging videos of the best individual per generation to ./videos/\n')

    env = gym.make('LunarLanderContinuous-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    pop = Population(
        MLP(state_dim, 64, 64, action_dim),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1.0', 'layers.2']
    )

    print(f'[PopuLoRA Lunar] population size: {pop_size} | total generations: {max_generations}')

    recent_rewards = deque(maxlen = avg_generations)
    pbar = tqdm(range(max_generations), desc = 'validating lunar lander')

    parent_selection_type = 'queen_bee' if queen_bee_mating else 'tournament'

    for gen in pbar:
        fitnesses = torch.zeros(pop_size)

        for i in range(pop_size):
            fitnesses[i] = evaluate_individual(
                env, pop, i,
                sample_actions = sample_actions,
                action_std = action_std
            )

        result = pop.select('deterministic', fitnesses = fitnesses, survive_frac = 0.5, elite_frac = 0.25)
        best_reward = fitnesses.max().item()
        mean_reward = fitnesses.mean().item()
        recent_rewards.append(mean_reward)

        avg_recent = float(np.mean(recent_rewards))
        pbar.write(f'gen {gen:03d} | best_reward: {best_reward:6.1f} | mean_reward: {mean_reward:6.1f} | avg_last_{avg_generations}_generations: {avg_recent:6.1f}')

        best_idx = fitnesses.argmax().item()
        record_env = gym.make('LunarLanderContinuous-v3', render_mode = 'rgb_array')
        record_env = gym.wrappers.RecordVideo(record_env, video_folder = 'videos', name_prefix = f'gen_{gen:03d}', disable_logger = True)
        evaluate_individual(
            record_env, pop, best_idx,
            sample_actions = False, # deterministic video evaluation for best policy
            action_std = action_std
        )
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
            pop.crossover_('extrapolative', parents, result.culled, fitnesses = fitnesses).mutate_('full_gaussian', individuals = result.culled, epsilon = 0.15).regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

    env.close()
    assert False, f'LunarLander average cumulative reward failed to reach > {target_avg_reward} (got {avg_recent:.2f})'

if __name__ == '__main__':
    fire.Fire(validate_with_lunar)
