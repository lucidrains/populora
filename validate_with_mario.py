# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "einops",
#     "numpy>=2.2.5",
#     "tqdm",
#     "fire",
#     "imageio",
#     "imageio-ffmpeg",
#     "torch-einops-utils",
#     "x-mlps-pytorch",
#     "populora",
#     "gym-super-mario-bros",
#     "nes-py",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

from __future__ import annotations

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import shutil
import warnings
from pathlib import Path

import fire
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# nes_py / gym_super_mario_bros numpy 2.x patch

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# silence gym-notices "upgrade to Gymnasium" deprecation banner printed at import

import gym_notices.notices
gym_notices.notices.notices = {}

import nes_py._rom
nes_py._rom.ROM.prg_rom_stop = property(lambda self: int(self.prg_rom_start) + int(self.prg_rom_size) * 1024)
nes_py._rom.ROM.chr_rom_stop = property(lambda self: int(self.chr_rom_start) + int(self.chr_rom_size) * 1024)

import gym_super_mario_bros.smb_env
SMB = gym_super_mario_bros.smb_env.SuperMarioBrosEnv
SMB._x_position = property(lambda self: int(self.ram[0x6d]) * 256 + int(self.ram[0x86]))
SMB._y_position = property(lambda self: int(self.ram[0x0b]))

from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from x_mlps_pytorch import MLP

from populora import (
    Population,
    distributed_device,
    distributed_world_size,
    is_distributed,
    is_main_rank,
    make_categorical_action,
    partition_indices,
    preserve_rng,
    sync_seed,
)

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

ACTION_SETS = dict(
    right = RIGHT_ONLY,
    simple = SIMPLE_MOVEMENT,
    complex = COMPLEX_MOVEMENT,
)

def parse_level(level):
    s = str(level).strip()
    return f"{s}-1" if '-' not in s else s

def make_mario_env(level = '1-1', actions = 'right', seed = None):
    level_str = parse_level(level)
    world, stage = (int(x) for x in level_str.split('-'))
    env = SuperMarioBrosEnv(rom_mode = 'vanilla', target = (world, stage))
    env = JoypadSpace(env, ACTION_SETS[actions])
    if exists(seed):
        env.seed(seed)
    return env

# frame preprocessing

GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype = torch.float32)

def preprocess(frames, img_size, device = 'cpu'):
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    stacked = np.stack([f if f.dtype == np.uint8 else f.astype(np.uint8) for f in frames])
    x = torch.from_numpy(stacked).float().to(device) / 255.0
    gray_w = GRAY_WEIGHTS.to(device)
    gray = torch.einsum('p h w c, c -> p h w', x, gray_w)
    gray = F.interpolate(gray[:, None], size = (img_size, img_size), mode = 'bilinear', align_corners = False)
    return gray.reshape(x.shape[0], -1)

# record agent video

def record_agent(
    pop: Population,
    individual_idx: int,
    level,
    actions,
    video_path,
    img_size,
    max_stagnant_steps = 180,
    sample_actions = True,
    policy_temperature = 1.0,
    seed = None
):
    rec_env = make_mario_env(level = level, actions = actions, seed = seed)
    obs, frames = rec_env.reset().copy(), []
    max_x, stagnant_steps = 0, 0

    action_fn = make_categorical_action(
        sample = sample_actions and policy_temperature > 0,
        temperature = policy_temperature
    )

    while True:
        frames.append(rec_env.render(mode = 'rgb_array').copy())
        with torch.no_grad():
            logits = pop(preprocess(obs, img_size, device = pop.device), individual = individual_idx)
            action = action_fn(logits).item()

        obs, reward, done, info = rec_env.step(action)
        obs = obs.copy()

        x_pos = int(info.get('x_pos', 0))
        if x_pos > max_x:
            max_x, stagnant_steps = x_pos, 0
        else:
            stagnant_steps += 1

        if done or (max_stagnant_steps > 0 and stagnant_steps >= max_stagnant_steps):
            break

    os.makedirs(os.path.dirname(video_path), exist_ok = True)
    imageio.mimsave(video_path, frames, fps = 30)
    rec_env.close()

# main validation script

def validate_with_mario(
    level = '1-1',
    actions = 'right',
    pop_size = 150,
    low_rank = 4,
    img_size = 42,
    max_generations = 2000,
    max_stagnant_steps = 180,
    seed = 42,
    crossover_type = 'extrapolative',
    parent_selection_type = 'tournament',
    survivor_selection_type = 'deterministic',
    mutation_type = 'full_gaussian',
    epsilon = 0.35,
    weight_decay = 1e-3,
    soft_threshold = 0.0,
    es_every_generations = 25,
    es_topk: int | float | None = None,
    es_temperature = 1.0,
    sample_actions = True,
    policy_temperature = 1.0,
    survive_frac = 0.5,
    elite_frac = 0.10,
    tournament_size = 5,
    render_video = True,
    video_dir = "./videos-mario",
    checkpoint_dir = "./checkpoints-mario",
    resume_from: str | Path | None = None,
):
    level_str = parse_level(level)

    sync_seed(seed)
    device = distributed_device('mps' if torch.backends.mps.is_available() else 'cpu')

    if is_main_rank():
        mode = 'distributed' if is_distributed() else 'single node'
        world_size = distributed_world_size()
        print(f'[PopuLoRA Mario] running in {mode} mode with {world_size} process(es) on {device}')
        print(f'[PopuLoRA Mario] population size: {pop_size} | total generations: {max_generations}')
        shutil.rmtree(video_dir, ignore_errors = True)
        os.makedirs(video_dir, exist_ok = True)
        os.makedirs(checkpoint_dir, exist_ok = True)

    assigned_indices = partition_indices(pop_size)
    envs = [make_mario_env(level = level_str, actions = actions, seed = seed + idx) for idx in assigned_indices]
    num_actions = envs[0].action_space.n
    state_dim = img_size * img_size

    pop = Population(
        MLP(state_dim, 512, 256, 128, num_actions),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1.0', 'layers.2.0', 'layers.3'],
        eval_seed = seed
    ).to(device)

    if exists(resume_from) and Path(resume_from).exists() and is_main_rank():
        pop.load(resume_from)

    # rank-local evaluation state (CPU: distributed collectives don't support MPS)

    max_x_tensor = torch.zeros(pop_size, dtype = torch.long, device = 'cpu')
    flag_get_tensor = torch.zeros(pop_size, dtype = torch.bool, device = 'cpu')

    def evaluate_episodes(pop, indices):
        num_local = len(indices)
        rewards = np.zeros(num_local, dtype = np.float32)
        done = np.zeros(num_local, dtype = bool)

        max_x_seen = np.zeros(num_local, dtype = np.int32)
        stagnant_steps = np.zeros(num_local, dtype = np.int32)

        # reset all envs under the shared seed

        for env in envs:
            env.seed(pop.eval_seed)

        obs_list = [env.reset().copy() for env in envs]

        action_fn = make_categorical_action(
            sample = sample_actions and policy_temperature > 0,
            temperature = policy_temperature
        )

        while not done.all():
            with torch.no_grad():
                obs_batch = preprocess(obs_list, img_size, device = device)
                logits = pop(obs_batch, individuals = indices)
                chosen_actions = action_fn(logits).cpu().numpy()

            for i in range(num_local):
                if done[i]:
                    continue

                obs, reward, d, info = envs[i].step(int(chosen_actions[i]))
                obs_list[i] = obs.copy()
                rewards[i] += reward

                x_pos = int(info.get('x_pos', 0))
                if x_pos > max_x_seen[i]:
                    max_x_seen[i], stagnant_steps[i] = x_pos, 0
                else:
                    stagnant_steps[i] += 1

                if info.get('flag_get', False):
                    flag_get_tensor[indices[i]] = True

                if d or (max_stagnant_steps > 0 and stagnant_steps[i] >= max_stagnant_steps):
                    done[i] = True

        for i, idx in enumerate(indices):
            max_x_tensor[idx] = int(max_x_seen[i])

        return rewards

    pbar = tqdm(range(max_generations), desc = 'validating mario') if is_main_rank() else range(max_generations)

    best_overall_x = 0
    winning_generation = None

    for gen in pbar:
        max_x_tensor.zero_()
        flag_get_tensor.zero_()

        fitnesses = pop.evaluate_distributed(evaluate_episodes, batch_eval = True, device = device)

        if is_distributed():
            dist.all_reduce(max_x_tensor, op = dist.ReduceOp.SUM)
            flag_int = flag_get_tensor.long()
            dist.all_reduce(flag_int, op = dist.ReduceOp.SUM)
            flag_get_tensor = flag_int.bool()

        best_reward = fitnesses.max().item()
        mean_reward = fitnesses.mean().item()
        best_player_idx = fitnesses.argmax().item()
        best_x_this_gen = int(max_x_tensor[best_player_idx].item())

        if is_main_rank():
            if render_video:
                with preserve_rng():
                    record_agent(pop, best_player_idx, level_str, actions, f"{video_dir}/gen_{gen:03d}.mp4", img_size, max_stagnant_steps, sample_actions, policy_temperature, seed)

            if best_x_this_gen > best_overall_x:
                best_overall_x = best_x_this_gen
                pop.save(f"{checkpoint_dir}/mario_{level_str}_best.pt")

            pbar.write(f'gen {gen:03d} | best_reward: {best_reward:8.1f} | mean_reward: {mean_reward:8.1f} | best_x: {best_x_this_gen:5d}')

        if flag_get_tensor.any():
            winning_player = int(torch.where(flag_get_tensor)[0][0])
            winning_generation = gen
            if is_main_rank():
                print(f"\n✓ PopuLoRA Mario Validation Passed! Level {level_str} beaten at generation {gen} by Player {winning_player}! Flag reached!\n")
            break

        if es_every_generations > 0 and divisible_by(gen + 1, es_every_generations):
            if is_main_rank():
                pbar.write(f'[ES] generation {gen:03d} | select_and_merge + repopulate')
            pop.select_and_merge(fitnesses = fitnesses, topk = es_topk, temperature = es_temperature, use_z_score = True)
            pop.repopulate()
        else:
            pop.evolve_(
                fitnesses,
                survive_frac = survive_frac,
                elite_frac = elite_frac,
                selection_type = survivor_selection_type,
                parent_selection_type = parent_selection_type,
                crossover_type = crossover_type,
                mutation_type = mutation_type,
                epsilon = epsilon,
                weight_decay = weight_decay,
                soft_threshold = soft_threshold,
                tournament_size = tournament_size
            )

    for env in envs:
        env.close()

    return winning_generation

if __name__ == "__main__":
    fire.Fire(validate_with_mario)
