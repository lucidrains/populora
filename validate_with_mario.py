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
import shutil
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import fire
import imageio

warnings.filterwarnings('ignore')

# NumPy 2.x compatibility patches for nes_py & gym_super_mario_bros

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import nes_py._rom
nes_py._rom.ROM.prg_rom_stop = property(lambda self: int(self.prg_rom_start) + int(self.prg_rom_size) * 1024)
nes_py._rom.ROM.chr_rom_stop = property(lambda self: int(self.chr_rom_start) + int(self.chr_rom_size) * 1024)

import gym_super_mario_bros.smb_env
SMB = gym_super_mario_bros.smb_env.SuperMarioBrosEnv
SMB._x_position = property(lambda self: int(self.ram[0x6d]) * 256 + int(self.ram[0x86]))
SMB._y_position = property(lambda self: int(self.ram[0x0b]))
SMB._score = property(lambda self: sum(int(self.ram[0x07de + i]) * 10**(5 - i) for i in range(6)))
SMB._time = property(lambda self: sum(int(self.ram[0x07f8 + i]) * 10**(2 - i) for i in range(3)))
SMB._coins = property(lambda self: int(self.ram[0x07ed]) * 10 + int(self.ram[0x07ee]))
SMB._world = property(lambda self: int(self.ram[0x075f]) + 1)
SMB._stage = property(lambda self: int(self.ram[0x075c]) + 1)
SMB._player_status = property(lambda self: int(self.ram[0x00ed]))
SMB._player_state = property(lambda self: int(self.ram[0x000e]))

from x_mlps_pytorch import MLP
from populora import Population

from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

ACTION_SETS = dict(
    right = RIGHT_ONLY,
    simple = SIMPLE_MOVEMENT,
    complex = COMPLEX_MOVEMENT,
)

def parse_level(level: str | int) -> str:
    s = str(level).strip()
    return f"{s}-1" if '-' not in s else s

def make_mario_env(level: str | int = '1-1', actions: str = 'right', seed: int | None = None):
    level_str = parse_level(level)
    world, stage = (int(x) for x in level_str.split('-'))
    env = SuperMarioBrosEnv(rom_mode = 'vanilla', target = (world, stage))
    env = JoypadSpace(env, ACTION_SETS[actions])
    if exists(seed):
        env.seed(seed)
    return env

# frame preprocessing - rgb (240, 256, 3) -> grayscale (img_size, img_size)

GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype = torch.float32)

def preprocess(frames: list[np.ndarray] | np.ndarray, img_size: int, device: str = 'cpu'):
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    stacked = np.stack([f if f.dtype == np.uint8 else f.astype(np.uint8) for f in frames])
    x = torch.from_numpy(stacked).float().to(device) / 255.0
    gray_w = GRAY_WEIGHTS.to(device)
    gray = torch.einsum('p h w c, c -> p h w', x, gray_w)
    gray = F.interpolate(gray[:, None], size = (img_size, img_size), mode = 'bilinear', align_corners = False)
    return gray.reshape(x.shape[0], -1)

# video recorder

def record_agent(
    pop: Population,
    individual_idx: int,
    level: str | int,
    actions: str,
    video_path: str,
    img_size: int,
    max_stagnant_steps: int = 180,
    sample_actions: bool = True,
    policy_temperature: float = 1.0,
    seed: int | None = None
):
    rec_env = make_mario_env(level = level, actions = actions, seed = seed)
    obs, frames = rec_env.reset().copy(), []
    max_x, stagnant_steps = 0, 0

    while True:
        frames.append(rec_env.render(mode = 'rgb_array').copy())
        with torch.no_grad():
            logits = pop(preprocess(obs, img_size, device = pop.device), individual = individual_idx)
            if sample_actions and policy_temperature > 0:
                probs = F.softmax(logits / policy_temperature, dim = -1)
                action = torch.multinomial(probs, num_samples = 1).item()
            else:
                action = logits.argmax(dim = -1).item()
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
    return info

# main validation script

def validate_with_mario(
    level: str | int = '1-1',
    actions: str = 'right',
    pop_size: int = 150,
    low_rank: int = 4,
    img_size: int = 42,
    max_generations: int = 2000,
    max_stagnant_steps: int = 180,
    seed: int = 42,
    crossover_type: str = 'extrapolative',
    parent_selection_type: str = 'tournament',
    survivor_selection_type: str = 'deterministic',
    mutation_type: str = 'full_gaussian',
    epsilon: float = 0.35,
    weight_decay: float = 1e-3,
    soft_threshold: float = 0.0,
    es_every_generations: int = 25,
    es_topk: int | float | None = None,
    es_temperature: float = 1.0,
    sample_actions: bool = True,
    policy_temperature: float = 1.0,
    render_video: bool = True,
    video_dir: str = "./videos-mario",
    checkpoint_dir: str = "./checkpoints-mario",
    resume_from: str | Path | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    level_str = parse_level(level)

    shutil.rmtree(video_dir, ignore_errors = True)
    os.makedirs(video_dir, exist_ok = True)
    os.makedirs(checkpoint_dir, exist_ok = True)

    print(f"\nlogging videos of the best individual to {video_dir}\n")

    probe = make_mario_env(level = level_str, actions = actions, seed = seed)
    probe.reset()
    num_actions = probe.action_space.n
    probe.close()

    state_dim = img_size * img_size
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"level {level_str} | actions: {actions} ({num_actions}) | obs: {state_dim}d | max_stagnant_steps: {max_stagnant_steps} | sample_actions: {sample_actions} | device: {device}\n")

    pop = Population(
        MLP(state_dim, 512, 256, 128, num_actions),
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = ['layers.0.0', 'layers.1.0', 'layers.2.0', 'layers.3']
    ).to(device)

    if exists(resume_from) and Path(resume_from).exists():
        print(f"resuming population from checkpoint: {resume_from}")
        pop.load(resume_from)

    envs = [make_mario_env(level = level_str, actions = actions, seed = seed + i) for i in range(pop_size)]
    pbar = tqdm(range(max_generations), desc = 'validating mario')

    best_overall_x = 0
    winning_generation = None

    for gen in pbar:
        obs_list = [env.reset().copy() for env in envs]
        total_rewards = np.zeros(pop_size, dtype = np.float32)
        done = np.zeros(pop_size, dtype = bool)
        flag_get = np.zeros(pop_size, dtype = bool)

        max_x_seen = np.zeros(pop_size, dtype = np.int32)
        stagnant_steps = np.zeros(pop_size, dtype = np.int32)

        while not done.all():
            with torch.no_grad():
                obs_batch = preprocess(obs_list, img_size, device = device)
                logits = pop(obs_batch, all_individuals = True)
                if sample_actions and policy_temperature > 0:
                    probs = F.softmax(logits / policy_temperature, dim = -1)
                    chosen_actions = torch.multinomial(probs.view(-1, num_actions), num_samples = 1).squeeze(-1).view(pop_size).cpu().numpy()
                else:
                    chosen_actions = logits.argmax(dim = -1).cpu().numpy()

            for i in range(pop_size):
                if done[i]:
                    continue
                obs, reward, d, info = envs[i].step(int(chosen_actions[i]))
                obs_list[i] = obs.copy()
                total_rewards[i] += reward

                x_pos = int(info.get('x_pos', 0))
                if x_pos > max_x_seen[i]:
                    max_x_seen[i] = x_pos
                    stagnant_steps[i] = 0
                else:
                    stagnant_steps[i] += 1

                if info.get('flag_get', False):
                    flag_get[i] = True
                if d or (max_stagnant_steps > 0 and stagnant_steps[i] >= max_stagnant_steps):
                    done[i] = True

        fitnesses = torch.tensor(total_rewards, dtype = torch.float32, device = pop.device)
        best_reward = fitnesses.max().item()
        mean_reward = fitnesses.mean().item()
        best_player_idx = fitnesses.argmax().item()
        best_x_this_gen = int(max_x_seen[best_player_idx])

        if render_video:
            record_agent(
                pop, best_player_idx, level_str, actions,
                f"{video_dir}/gen_{gen:03d}.mp4", img_size = img_size,
                max_stagnant_steps = max_stagnant_steps,
                sample_actions = sample_actions, policy_temperature = policy_temperature, seed = seed
            )

        if best_x_this_gen > best_overall_x:
            best_overall_x = best_x_this_gen
            pop.save(f"{checkpoint_dir}/mario_{level_str}_best.pt")
            if render_video:
                record_agent(
                    pop, best_player_idx, level_str, actions,
                    f"{video_dir}/mario_{level_str}_highscore.mp4", img_size = img_size,
                    max_stagnant_steps = max_stagnant_steps,
                    sample_actions = sample_actions, policy_temperature = policy_temperature, seed = seed
                )

        pbar.write(f'gen {gen:03d} | best_reward: {best_reward:8.1f} | mean_reward: {mean_reward:8.1f} | best_x: {best_x_this_gen:5d}')

        if flag_get.any():
            winning_player = int(np.where(flag_get)[0][0])
            winning_generation = gen
            print(f"\n✓ PopuLoRA Mario Validation Passed! Level {level_str} beaten at generation {gen} by Player {winning_player}! Flag reached!\n")
            if render_video:
                record_agent(
                    pop, winning_player, level_str, actions,
                    f"{video_dir}/mario_{level_str}_winner_gen{gen}.mp4", img_size = img_size,
                    max_stagnant_steps = max_stagnant_steps,
                    sample_actions = sample_actions, policy_temperature = policy_temperature, seed = seed
                )
            break

        result = pop.select(survivor_selection_type, fitnesses = fitnesses, survive_frac = 0.5, elite_frac = 0.10)

        if es_every_generations > 0 and divisible_by(gen + 1, es_every_generations):
            pbar.write(f'[ES] generation {gen:03d} | select_and_merge + repopulate')
            pop.select_and_merge(fitnesses = fitnesses, topk = es_topk, temperature = es_temperature, use_z_score = True)
            pop.repopulate()
        else:
            parents = pop.select_parents(parent_selection_type, fitnesses = fitnesses, num_children = len(result.culled), culled = result.culled, tournament_size = 5)
            pop.crossover_(crossover_type, parents, result.culled, fitnesses = fitnesses) \
               .mutate_(mutation_type, individuals = result.culled, epsilon = epsilon) \
               .regularize_(weight_decay = weight_decay, soft_threshold = soft_threshold)

    for env in envs:
        env.close()

    if winning_generation is None:
        print(f"\nFinished {max_generations} generations. Best X reached: {best_overall_x}")

    best_video = Path(video_dir) / f"mario_{level_str}_highscore.mp4"
    if best_video.exists():
        print(f"\nbest video: {best_video}")

    return winning_generation

if __name__ == "__main__":
    fire.Fire(validate_with_mario)
