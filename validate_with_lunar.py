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

def evaluate_individual(env: gym.Env, pop: Population, individual_idx: int) -> float:
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = pop(state_tensor, individual = individual_idx).squeeze(0).tanh().numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return total_reward

def validate_with_lunar(
    target_avg_reward: float = 100.0,
    avg_window: int = 128,
    pop_size: int = 128,
    low_rank: int = 4,
    max_generations: int = 300,
    seed: int = 42,
    es_every_generations: int = 25,
    es_topk: int | float | None = None,
    es_temperature: float = 1.0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    shutil.rmtree("videos", ignore_errors=True)
    os.makedirs("videos", exist_ok=True)
    print("\nlogging videos of the best individual per generation to ./videos/\n")

    env = gym.make("LunarLanderContinuous-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    pop = Population(
        MLP(state_dim, 64, 64, action_dim),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1.0', 'layers.2']
    )

    recent_rewards = deque(maxlen = avg_window)
    pbar = tqdm(range(max_generations), desc = "validating lunar lander")

    for gen in pbar:
        fitnesses = torch.zeros(pop_size)

        for i in range(pop_size):
            fitnesses[i] = evaluate_individual(env, pop, i)

        result = pop.select('deterministic', fitnesses = fitnesses, survive_frac = 0.5, elite_frac = 0.25)
        best_reward = fitnesses.max().item()
        recent_rewards.append(best_reward)

        avg_recent = float(np.mean(recent_rewards))
        pbar.write(f"gen {gen:03d} | best_reward: {best_reward:6.1f} | avg_last_{avg_window}_episodes: {avg_recent:6.1f}")

        best_idx = fitnesses.argmax().item()
        record_env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        record_env = gym.wrappers.RecordVideo(record_env, video_folder="videos", name_prefix=f"gen_{gen:03d}", disable_logger=True)
        evaluate_individual(record_env, pop, best_idx)
        record_env.close()

        if len(recent_rewards) >= avg_window and avg_recent > target_avg_reward:
            print(f"\n✓ PopuLoRA LunarLander Validation Passed! Last {avg_window} episodes avg reward: {avg_recent:.2f} > {target_avg_reward}")
            env.close()
            return

        if es_every_generations > 0 and divisible_by(gen + 1, es_every_generations):
            pbar.write(f"[ES] generation {gen:03d} | select_and_merge + repopulate")
            pop.select_and_merge(fitnesses = fitnesses, topk = es_topk, temperature = es_temperature)
            pop.repopulate()
        else:
            parents = pop.select_parents('tournament', fitnesses = fitnesses, num_children = len(result.culled))
            pop.crossover_('extrapolative', parents, result.culled, fitnesses = fitnesses)
            pop.mutate_('full_gaussian', individuals = result.culled, epsilon = 0.15)

    env.close()
    assert False, f"LunarLander average cumulative reward failed to reach > {target_avg_reward} (got {avg_recent:.2f})"

if __name__ == '__main__':
    fire.Fire(validate_with_lunar)
