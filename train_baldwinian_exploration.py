from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from populora import Baldwin, Population, interact_with_env

# environments

class SwitchbackFourRoomsVecEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int = 32, horizon: int = 240):
        self.num_envs = num_envs
        self.horizon = horizon
        self.autoresets = True
        self.single_observation_space = gym.spaces.Box(low = -2.0, high = 2.0, shape = (12,), dtype = np.float32)
        self.single_action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

        self.segments = np.array([
            [-1.0, -1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0,  1.0],
            [-1.0, -1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0,  1.0],
            [ 0.0, -1.0,  0.0, -0.36],
            [ 0.0, -0.04, 0.0,  0.35],
            [ 0.0,  0.85, 0.0,  1.0],
            [-1.0,  0.0, -0.36, 0.0],
            [-0.04, 0.0,  0.35, 0.0],
            [ 0.85, 0.0,  1.0,  0.0],
            [ 0.35, -0.45, 0.35,  0.0],
            [-0.45,  0.35, 0.0,   0.35],
        ], dtype = np.float32)

        angles = np.linspace(0, 2 * np.pi, 8, endpoint = False, dtype = np.float32)
        self.ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis = -1)
        self.start_pos = np.array([-0.75, -0.75], dtype = np.float32)
        self.goal = np.array([0.45, 0.45], dtype = np.float32)
        self.goal_radius = 0.35

        self.pos = np.zeros((num_envs, 2), dtype = np.float32)
        self.vel = np.zeros((num_envs, 2), dtype = np.float32)
        self.steps = np.zeros(num_envs, dtype = np.int32)
        self.visited_grids = np.zeros((num_envs, 12, 12), dtype = bool)
        self.max_cells = np.zeros(num_envs, dtype = np.int32)
        self.max_rooms = np.zeros(num_envs, dtype = np.int32)
        self.rooms_visited = np.zeros((num_envs, 4), dtype = bool)

        self.ep_t = 0
        self.ep_done = np.zeros(num_envs, dtype = bool)
        self.goal_step = np.full(num_envs, horizon, dtype = np.int32)
        self.traj_pos = np.zeros((num_envs, horizon + 1, 2), dtype = np.float32)
        self.traj_vel = np.zeros((num_envs, horizon + 1, 2), dtype = np.float32)
        self.traj_rays = np.zeros((num_envs, horizon + 1, 8), dtype = np.float32)
        self.traj_grids = np.zeros((num_envs, horizon + 1, 12, 12), dtype = bool)
        self.traj_cells = np.zeros((num_envs, horizon + 1), dtype = np.int32)
        self.traj_rooms = np.zeros((num_envs, horizon + 1), dtype = np.int32)

    def _raycast(self, pos: np.ndarray) -> np.ndarray:
        p, r = pos[:, None, None, :], self.ray_dirs[None, :, None, :]
        q = self.segments[None, None, :, :2]
        s = self.segments[None, None, :, 2:] - q
        r_cross_s = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
        qp = q - p
        qp_cross_s = qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]
        qp_cross_r = qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]
        denom = np.where(np.abs(r_cross_s) < 1e-8, 1e-8, r_cross_s)
        t, u = qp_cross_s / denom, qp_cross_r / denom
        valid = (np.abs(r_cross_s) >= 1e-8) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        dists = np.where(valid, t, 2.0).min(axis = -1)
        return np.clip(dists, 0.0, 2.0) / 2.0 - 0.5

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self._raycast(self.pos)], axis = -1).astype(np.float32)

    def _mark(self, idx: np.ndarray):
        gx = np.clip(((self.pos[idx, 0] + 1.0) * 6.0).astype(np.int32), 0, 11)
        gy = np.clip(((self.pos[idx, 1] + 1.0) * 6.0).astype(np.int32), 0, 11)
        self.visited_grids[idx, gx, gy] = True
        x, y = self.pos[idx, 0], self.pos[idx, 1]
        self.rooms_visited[idx, 0] |= (x < 0) & (y < 0)
        self.rooms_visited[idx, 1] |= (x < 0) & (y >= 0)
        self.rooms_visited[idx, 2] |= (x >= 0) & (y >= 0)
        self.rooms_visited[idx, 3] |= (x >= 0) & (y < 0)
        self.max_cells[idx] = np.maximum(self.max_cells[idx], self.visited_grids[idx].sum(axis = (1, 2)))
        self.max_rooms[idx] = np.maximum(self.max_rooms[idx], self.rooms_visited[idx].sum(axis = 1))

    def reset(self, seed = None, options = None):
        self.pos[:] = self.start_pos
        self.vel[:] = 0.0
        self.steps[:] = 0
        self.visited_grids[:] = False
        self.rooms_visited[:] = False
        self.rooms_visited[:, 0] = True
        self._mark(np.arange(self.num_envs))

        self.ep_t = 0
        self.ep_done[:] = False
        self.goal_step[:] = self.horizon
        self.traj_pos[:, 0] = self.pos
        self.traj_vel[:, 0] = self.vel
        self.traj_rays[:, 0] = (self._raycast(self.pos) + 0.5) * 2.0
        self.traj_grids[:, 0] = self.visited_grids
        self.traj_cells[:, 0] = self.visited_grids.sum(axis = (1, 2))
        self.traj_rooms[:, 0] = self.rooms_visited.sum(axis = 1)
        return self._get_obs(), {}

    def _collides(self, p1: np.ndarray) -> np.ndarray:
        a, b = self.segments[None, :, :2], self.segments[None, :, 2:]
        ab, ap = b - a, p1[:, None, :] - a
        t = np.clip(np.sum(ap * ab, axis = -1) / (np.sum(ab * ab, axis = -1) + 1e-8), 0.0, 1.0)
        dist = np.linalg.norm(p1[:, None, :] - (a + t[..., None] * ab), axis = -1)
        return (dist < 0.032).any(axis = -1)

    def step(self, action):
        act = np.clip(np.asarray(action, dtype = np.float32), -1.0, 1.0)
        self.vel = 0.72 * self.vel + 0.035 * act
        speed = np.linalg.norm(self.vel, axis = -1, keepdims = True)
        self.vel = np.where(speed > 0.075, self.vel * (0.075 / np.maximum(speed, 1e-8)), self.vel)

        for axis, other in ((0, 1), (1, 0)):
            cand = np.clip(self.pos.copy(), -0.96, 0.96)
            cand[:, axis] = np.clip(self.pos[:, axis] + self.vel[:, axis], -0.96, 0.96)
            hit = self._collides(cand)
            self.pos[:, axis] = np.where(hit, self.pos[:, axis], cand[:, axis])
            self.vel[:, axis] = np.where(hit, 0.0, self.vel[:, axis])
            self.vel[:, other] = np.where(hit, self.vel[:, other] * 0.6, self.vel[:, other])

        self.steps += 1
        self._mark(np.arange(self.num_envs))
        dist_goal = np.linalg.norm(self.pos - self.goal[None, :], axis = -1)
        reached = (self.pos[:, 0] > 0.05) & (self.pos[:, 1] > 0.05) & (dist_goal < self.goal_radius)
        rew = -0.015 * np.sum(act ** 2, axis = -1) + np.where(reached, 100.0, 0.0)
        terminated, truncated = reached, self.steps >= self.horizon
        dones = terminated | truncated

        if self.ep_t < self.horizon:
            t_next = self.ep_t + 1
            cur_rays = (self._raycast(self.pos) + 0.5) * 2.0
            for i in range(self.num_envs):
                if not self.ep_done[i]:
                    self.traj_pos[i, t_next] = self.pos[i]
                    self.traj_vel[i, t_next] = self.vel[i]
                    self.traj_rays[i, t_next] = cur_rays[i]
                    self.traj_grids[i, t_next] = self.visited_grids[i]
                    self.traj_cells[i, t_next] = int(self.max_cells[i])
                    self.traj_rooms[i, t_next] = int(self.max_rooms[i])
                    if dones[i]:
                        self.ep_done[i] = True
                        if terminated[i]:
                            self.goal_step[i] = t_next
                else:
                    self.traj_pos[i, t_next] = self.traj_pos[i, self.ep_t]
                    self.traj_vel[i, t_next] = 0.0
                    self.traj_rays[i, t_next] = self.traj_rays[i, self.ep_t]
                    self.traj_grids[i, t_next] = self.traj_grids[i, self.ep_t]
                    self.traj_cells[i, t_next] = self.traj_cells[i, self.ep_t]
                    self.traj_rooms[i, t_next] = self.traj_rooms[i, self.ep_t]
            self.ep_t = t_next

        if np.any(dones):
            idx = np.where(dones)[0]
            self.pos[idx] = self.start_pos
            self.vel[idx] = 0.0
            self.steps[idx] = 0
            self.visited_grids[idx] = False
            self.rooms_visited[idx] = False
            self.rooms_visited[idx, 0] = True
            self._mark(idx)
        return self._get_obs(), rew.astype(np.float32), terminated, truncated, {}


class DeceptiveMountainCarVecEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int = 32, horizon: int = 250, power: float = 0.00125):
        self.num_envs = num_envs
        self.horizon = horizon
        self.power = power
        self.autoresets = True
        self.single_observation_space = gym.spaces.Box(
            low = np.array([-1.2, -0.07], dtype = np.float32),
            high = np.array([0.6, 0.07], dtype = np.float32),
            dtype = np.float32,
        )
        self.single_action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (1,), dtype = np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

        self.pos = np.zeros(num_envs, dtype = np.float32)
        self.vel = np.zeros(num_envs, dtype = np.float32)
        self.steps = np.zeros(num_envs, dtype = np.int32)
        self.max_pos = np.full(num_envs, -0.5, dtype = np.float32)
        self.visited_bins = np.zeros((num_envs, 20), dtype = bool)
        self.max_bins = np.zeros(num_envs, dtype = np.int32)

        self.ep_t = 0
        self.ep_done = np.zeros(num_envs, dtype = bool)
        self.goal_step = np.full(num_envs, horizon, dtype = np.int32)
        self.traj_pos = np.zeros((num_envs, horizon + 1), dtype = np.float32)
        self.traj_vel = np.zeros((num_envs, horizon + 1), dtype = np.float32)
        self.traj_bins = np.zeros((num_envs, horizon + 1), dtype = np.int32)

    def _get_obs(self) -> np.ndarray:
        # Raw, unscaled [position, velocity]
        return np.stack([self.pos, self.vel], axis = -1).astype(np.float32)

    def _mark(self, idx: np.ndarray):
        self.max_pos[idx] = np.maximum(self.max_pos[idx], self.pos[idx])
        b = np.clip(((self.pos[idx] + 1.2) / 1.8 * 20.0).astype(np.int32), 0, 19)
        self.visited_bins[idx, b] = True
        self.max_bins[idx] = np.maximum(self.max_bins[idx], self.visited_bins[idx].sum(axis = 1))

    def reset(self, seed = None, options = None):
        self.pos[:] = -0.5
        self.vel[:] = 0.0
        self.steps[:] = 0
        self.visited_bins[:] = False
        self._mark(np.arange(self.num_envs))

        self.ep_t = 0
        self.ep_done[:] = False
        self.goal_step[:] = self.horizon
        self.traj_pos[:, 0] = self.pos
        self.traj_vel[:, 0] = self.vel
        self.traj_bins[:, 0] = self.visited_bins.sum(axis = 1)
        return self._get_obs(), {}

    def step(self, action):
        act = np.clip(np.asarray(action, dtype = np.float32).reshape(self.num_envs), -1.0, 1.0)
        self.vel = np.clip(self.vel + act * self.power - 0.0025 * np.cos(3.0 * self.pos), -0.07, 0.07)
        self.pos = np.clip(self.pos + self.vel, -1.2, 0.6)
        self.vel = np.where((self.pos <= -1.2) & (self.vel < 0.0), 0.0, self.vel)
        self.steps += 1
        self._mark(np.arange(self.num_envs))

        reached = self.pos >= 0.45
        rew = -0.1 * (act ** 2) + np.where(reached, 100.0, 0.0)
        terminated, truncated = reached, self.steps >= self.horizon
        dones = terminated | truncated

        if self.ep_t < self.horizon:
            t_next = self.ep_t + 1
            for i in range(self.num_envs):
                if not self.ep_done[i]:
                    self.traj_pos[i, t_next] = self.pos[i]
                    self.traj_vel[i, t_next] = self.vel[i]
                    self.traj_bins[i, t_next] = int(self.max_bins[i])
                    if dones[i]:
                        self.ep_done[i] = True
                        if terminated[i]:
                            self.goal_step[i] = t_next
                else:
                    self.traj_pos[i, t_next] = self.traj_pos[i, self.ep_t]
                    self.traj_vel[i, t_next] = 0.0
                    self.traj_bins[i, t_next] = self.traj_bins[i, self.ep_t]
            self.ep_t = t_next

        if np.any(dones):
            idx = np.where(dones)[0]
            self.pos[idx] = -0.5
            self.vel[idx] = 0.0
            self.steps[idx] = 0
            self.visited_bins[idx] = False
            self._mark(idx)
        return self._get_obs(), rew.astype(np.float32), terminated, truncated, {}

# innate zero-TTT evaluation & trajectory extraction

@torch.no_grad()
def evaluate_innate_zero_ttt_four_rooms(pop: Population, ind_idx: int, horizon: int = 240) -> tuple[float, float]:
    starts = np.array([
        [-0.75, -0.75], [-0.82, -0.70], [-0.68, -0.80], [-0.75, -0.62],
        [-0.62, -0.75], [-0.85, -0.85], [-0.65, -0.65], [-0.78, -0.58],
    ], dtype = np.float32)
    env = SwitchbackFourRoomsVecEnv(num_envs = len(starts), horizon = horizon)
    env.reset()
    env.pos[:] = starts
    env.visited_grids[:] = False
    env.rooms_visited[:] = False
    env.rooms_visited[:, 0] = True
    env.max_cells[:] = 0
    env.max_rooms[:] = 0
    env._mark(np.arange(len(starts)))

    obs = env._get_obs()
    active = np.ones(len(starts), dtype = bool)
    cells_at_done = np.zeros(len(starts), dtype = np.float32)
    rooms_at_done = np.zeros(len(starts), dtype = np.float32)

    for _ in range(horizon):
        act = torch.tanh(pop(torch.from_numpy(obs), individual = ind_idx)).cpu().numpy()
        prev_cells = env.visited_grids.sum(axis = (1, 2)).astype(np.float32)
        prev_rooms = env.rooms_visited.sum(axis = 1).astype(np.float32)
        obs, _, term, trunc, _ = env.step(act)
        cur_cells = np.maximum(prev_cells, env.max_cells.astype(np.float32))
        cur_rooms = np.maximum(prev_rooms, env.max_rooms.astype(np.float32))
        newly_done = active & (term | trunc)
        if np.any(newly_done):
            cells_at_done[newly_done] = cur_cells[newly_done]
            rooms_at_done[newly_done] = cur_rooms[newly_done]
            active[newly_done] = False
        if not np.any(active):
            break

    if np.any(active):
        cells_at_done[active] = env.max_cells[active].astype(np.float32)
        rooms_at_done[active] = env.max_rooms[active].astype(np.float32)

    return float(cells_at_done.mean()), float(rooms_at_done.mean())


def _extract_learned_curve(pop: Population, ind_idx: int, use_baldwin: bool, g_step: int, horizon: int) -> np.ndarray:
    curve = np.zeros(horizon + 1, dtype = np.float32)
    if use_baldwin and pop.baldwin is not None and len(pop.baldwin.slot_learned_trace) > 0:
        trace = torch.stack(pop.baldwin.slot_learned_trace, dim = 0)[:, ind_idx].cpu().numpy()
        n = min(horizon, len(trace))
        curve[1 : n + 1] = trace[:n]
        if n < horizon:
            curve[n + 1 :] = curve[n]
        if 0 < g_step < horizon and pop.baldwin.horizon_projection:
            curve[g_step:] = curve[g_step] * (float(horizon) / float(g_step))
    return curve


def record_four_rooms_rollout(env: SwitchbackFourRoomsVecEnv, pop: Population, ind_idx: int, use_baldwin: bool, horizon: int) -> dict[str, np.ndarray]:
    g_step = int(env.goal_step[ind_idx])
    return {
        'positions': env.traj_pos[ind_idx].copy(),
        'velocities': env.traj_vel[ind_idx].copy(),
        'rays': env.traj_rays[ind_idx].copy(),
        'grids': env.traj_grids[ind_idx].copy(),
        'cells': env.traj_cells[ind_idx].copy(),
        'rooms': env.traj_rooms[ind_idx].copy(),
        'learned': _extract_learned_curve(pop, ind_idx, use_baldwin, g_step, horizon),
        'goal_step': np.array(g_step),
        'segments': env.segments.copy(),
        'ray_dirs': env.ray_dirs.copy(),
        'goal': env.goal.copy(),
        'goal_radius': np.array(env.goal_radius),
    }


def record_mountain_car_rollout(env: DeceptiveMountainCarVecEnv, pop: Population, ind_idx: int, use_baldwin: bool, horizon: int) -> dict[str, np.ndarray]:
    g_step = int(env.goal_step[ind_idx])
    return {
        'pos': env.traj_pos[ind_idx].copy(),
        'vel': env.traj_vel[ind_idx].copy(),
        'bins': env.traj_bins[ind_idx].copy(),
        'learned': _extract_learned_curve(pop, ind_idx, use_baldwin, g_step, horizon),
        'goal_step': np.array(g_step),
    }

# selection regimes & experiment runners

MODES = [
    ('baseline',            'Baseline (No TTT, Reward)',           False, 1.0, 0.0, False),
    ('baldwin_reward',      'Baldwin (TTT, Reward Only)',          True,  1.0, 0.0, False),
    ('baldwin_pc_only',     'Baldwin (TTT, PC Only)',              True,  0.0, 1.0, False),
    ('baldwin_alternating', 'Baldwin (TTT, Alternating PC/Rew)',   True,  0.0, 1.0, True),
    ('baldwin_mixture',     'Baldwin (TTT, Mixture: Rew + PC)',    True,  1.0, 1.0, False),
]


def run_four_rooms_experiments(
    seeds: tuple[int, ...] = (42, 43, 44),
    num_generations: int = 28,
    pop_size: int = 32,
    horizon: int = 240,
    pc_scale: float = 400.0,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    print('\n' + '=' * 96)
    print(f'EXPERIMENT 1: Switchback Four-Rooms Continuous Maze ({num_generations} Generations, Horizon {horizon})')
    print('=' * 96)

    results = {
        m[0]: {k: [] for k in ('max_ext', 'mean_ext', 'mean_cells', 'max_cells', 'mean_rooms', 'max_rooms', 'mean_lrn', 'solved_count', 'innate_cells', 'innate_rooms')}
        for m in MODES
    }
    best_rollouts: dict[str, dict[str, np.ndarray]] = {}
    best_scores = {m[0]: -1e9 for m in MODES}

    for mode_key, _, use_b, rw, pw_mult, is_alt in MODES:
        pw = pw_mult * pc_scale
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            env = SwitchbackFourRoomsVecEnv(num_envs = pop_size, horizon = horizon)
            interactor = interact_with_env(env, seed = s)
            net = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 2))
            model = Baldwin(
                net,
                latent_layer = net[2],
                lora_memory = net[2],
                action_dim = 2,
                lora_lr = 2.5e-2,
                predictor_lr = 2e-2,
                update_every = 2,
                trust_region = 0.8,
                anchor_decay = 0.005,
                reward_weight = 1.0 if is_alt else rw,
                predictor_learning_weight = pw,
                alternate_every = 3 if is_alt else None,
            ) if use_b else net

            pop = Population(model, pop_size = pop_size, low_rank = 4, eval_seed = s)
            pop.mutate_('full_gaussian', all_individuals = True, epsilon = 0.25)

            gen_max_ext = np.zeros(num_generations, dtype = np.float32)
            gen_mean_ext = np.zeros(num_generations, dtype = np.float32)
            gen_mean_cells = np.zeros(num_generations, dtype = np.float32)
            gen_max_cells = np.zeros(num_generations, dtype = np.float32)
            gen_mean_rooms = np.zeros(num_generations, dtype = np.float32)
            gen_max_rooms = np.zeros(num_generations, dtype = np.float32)
            gen_mean_lrn = np.zeros(num_generations, dtype = np.float32)
            gen_solved = np.zeros(num_generations, dtype = np.float32)

            for g in range(num_generations):
                env.max_cells[:] = 0
                env.max_rooms[:] = 0
                fit = interactor.evaluate(pop, action = torch.tanh, horizon = horizon, num_episodes = 1, seed = s)
                ext = pop.baldwin.last_extrinsic_fitness if use_b else fit
                lrn = pop.baldwin.last_learned_fitness if use_b else torch.zeros_like(fit)

                gen_max_ext[g] = float(ext.max().item())
                gen_mean_ext[g] = float(ext.mean().item())
                gen_mean_cells[g] = float(env.max_cells.mean())
                gen_max_cells[g] = float(env.max_cells.max())
                gen_mean_rooms[g] = float(env.max_rooms.mean())
                gen_max_rooms[g] = float(env.max_rooms.max())
                gen_mean_lrn[g] = float(lrn.mean().item())
                gen_solved[g] = float((ext > 50.0).sum().item())

                if mode_key == 'baldwin_pc_only':
                    scores = env.max_cells.astype(np.float32) + 10.0 * env.max_rooms.astype(np.float32)
                else:
                    scores = ext.cpu().numpy() + 0.5 * env.max_cells.astype(np.float32)
                cand_idx = int(np.argmax(scores))
                cand_score = float(scores[cand_idx])

                record_cond = (s == seeds[0] and g == num_generations - 1) if mode_key in ('baseline', 'baldwin_reward') else (g >= num_generations // 2 and cand_score > best_scores[mode_key])
                if record_cond:
                    best_scores[mode_key] = cand_score
                    best_rollouts[mode_key] = record_four_rooms_rollout(env, pop, cand_idx, use_b, horizon)

                if g < num_generations - 1:
                    pop.evolve_(fit, survive_frac = 0.5, crossover_type = 'extrapolative', mutation_type = 'full_gaussian', epsilon = 0.25)

            champ_idx = int(np.argmax(env.max_cells)) if mode_key == 'baldwin_pc_only' else int(torch.argmax(ext).item())
            innate_c, innate_r = evaluate_innate_zero_ttt_four_rooms(pop, champ_idx, horizon = horizon)

            results[mode_key]['max_ext'].append(np.maximum.accumulate(gen_max_ext))
            results[mode_key]['mean_ext'].append(gen_mean_ext)
            results[mode_key]['mean_cells'].append(gen_mean_cells)
            results[mode_key]['max_cells'].append(gen_max_cells)
            results[mode_key]['mean_rooms'].append(gen_mean_rooms)
            results[mode_key]['max_rooms'].append(gen_max_rooms)
            results[mode_key]['mean_lrn'].append(gen_mean_lrn)
            results[mode_key]['solved_count'].append(gen_solved)
            results[mode_key]['innate_cells'].append(np.array(innate_c))
            results[mode_key]['innate_rooms'].append(np.array(innate_r))

            print(
                f'{mode_key:20s} s={s} | Solved={int(gen_solved[-1]):2d}/{pop_size} (Peak={int(gen_solved.max()):2d}) | '
                f'MaxExt={gen_max_ext.max():+6.2f} | '
                f'MeanCells={gen_mean_cells[-1]:5.1f} (Max={int(gen_max_cells.max()):2d}) | '
                f'MeanRooms={gen_mean_rooms[-1]:.2f} (Max={int(gen_max_rooms.max())}) | '
                f'Innate0TTT={innate_c:4.1f}c ({innate_r:.2f}r) | '
                f'MeanLrn={gen_mean_lrn[-1]:.4f}',
                flush = True,
            )

    return results, best_rollouts


def run_mountain_car_experiments(
    seeds: tuple[int, ...] = (42, 43, 44),
    num_generations: int = 18,
    pop_size: int = 32,
    horizon: int = 250,
    pc_scale: float = 50000.0,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    print('\n' + '=' * 96)
    print(f'EXPERIMENT 2: Honest Deceptive MountainCar (Raw [pos, vel], {num_generations} Generations, Horizon {horizon})')
    print('=' * 96)

    results = {
        m[0]: {k: [] for k in ('max_ext', 'max_pos', 'mean_pos', 'mean_bins', 'mean_lrn', 'solved_count')}
        for m in MODES
    }
    best_rollouts: dict[str, dict[str, np.ndarray]] = {}
    best_scores = {m[0]: -1e9 for m in MODES}

    for mode_key, _, use_b, rw, pw_mult, is_alt in MODES:
        pw = pw_mult * pc_scale
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            env = DeceptiveMountainCarVecEnv(num_envs = pop_size, horizon = horizon, power = 0.00125)
            interactor = interact_with_env(env, seed = s)
            net = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
            model = Baldwin(
                net,
                latent_layer = net[2],
                lora_memory = net[2],
                action_dim = 1,
                lora_lr = 8e-3,
                predictor_lr = 2e-2,
                update_every = 2,
                trust_region = 0.8,
                anchor_decay = 0.005,
                reward_weight = 1.0 if is_alt else rw,
                predictor_learning_weight = pw,
                alternate_every = 3 if is_alt else None,
                novelty_bandwidth = 0.001,
            ) if use_b else net

            pop = Population(model, pop_size = pop_size, low_rank = 4, eval_seed = s)
            pop.mutate_('full_gaussian', all_individuals = True, epsilon = 0.25)

            gen_max_ext = np.zeros(num_generations, dtype = np.float32)
            gen_max_pos = np.zeros(num_generations, dtype = np.float32)
            gen_mean_pos = np.zeros(num_generations, dtype = np.float32)
            gen_mean_bins = np.zeros(num_generations, dtype = np.float32)
            gen_mean_lrn = np.zeros(num_generations, dtype = np.float32)
            gen_solved = np.zeros(num_generations, dtype = np.float32)

            for g in range(num_generations):
                env.max_pos[:] = -0.5
                env.max_bins[:] = 0
                fit = interactor.evaluate(pop, action = torch.tanh, horizon = horizon, num_episodes = 1, seed = s)
                ext = pop.baldwin.last_extrinsic_fitness if use_b else fit
                lrn = pop.baldwin.last_learned_fitness if use_b else torch.zeros_like(fit)

                gen_max_ext[g] = float(ext.max().item())
                gen_max_pos[g] = float(env.max_pos.max())
                gen_mean_pos[g] = float(env.max_pos.mean())
                gen_mean_bins[g] = float(env.max_bins.mean())
                gen_mean_lrn[g] = float(lrn.mean().item())
                gen_solved[g] = float((ext > 50.0).sum().item())

                if mode_key == 'baldwin_pc_only':
                    scores = env.max_bins.astype(np.float32) + 10.0 * env.max_pos + 0.2 * ext.cpu().numpy()
                else:
                    scores = ext.cpu().numpy() + env.max_bins.astype(np.float32)
                cand_idx = int(np.argmax(scores))
                cand_score = float(scores[cand_idx])

                if g >= num_generations // 2 and cand_score > best_scores[mode_key]:
                    best_scores[mode_key] = cand_score
                    best_rollouts[mode_key] = record_mountain_car_rollout(env, pop, cand_idx, use_b, horizon)

                if g < num_generations - 1:
                    pop.evolve_(fit, survive_frac = 0.5, crossover_type = 'extrapolative', mutation_type = 'full_gaussian', epsilon = 0.25)

            results[mode_key]['max_ext'].append(np.maximum.accumulate(gen_max_ext))
            results[mode_key]['max_pos'].append(gen_max_pos)
            results[mode_key]['mean_pos'].append(gen_mean_pos)
            results[mode_key]['mean_bins'].append(gen_mean_bins)
            results[mode_key]['mean_lrn'].append(gen_mean_lrn)
            results[mode_key]['solved_count'].append(gen_solved)

            print(
                f'{mode_key:20s} s={s} | Gen0={int(gen_solved[0]):2d}/{pop_size} -> Final={int(gen_solved[-1]):2d}/{pop_size} (Peak={int(gen_solved.max()):2d}) | '
                f'MaxExt={gen_max_ext.max():+6.2f} | '
                f'MaxX={gen_max_pos.max():+.2f} (Mean={gen_mean_pos[-1]:+.2f}) | '
                f'MeanBins={gen_mean_bins[-1]:4.1f}/20 | '
                f'MeanLrn={gen_mean_lrn[-1]:.4f}',
                flush = True,
            )

    return results, best_rollouts

# plotting & video rendering

MODE_COLORS = {
    'baseline':            '#7f8c8d',
    'baldwin_reward':      '#2980b9',
    'baldwin_pc_only':     '#e67e22',
    'baldwin_alternating': '#9b59b6',
    'baldwin_mixture':     '#27ae60',
}

MODE_SHORT_LABELS = {
    'baseline':            'Baseline (No TTT)',
    'baldwin_reward':      'Baldwin (Reward Only)',
    'baldwin_pc_only':     'Baldwin (PC Learning Only)',
    'baldwin_alternating': 'Baldwin (Alternating PC/Rew)',
    'baldwin_mixture':     'Baldwin (Mixture: Rew + PC)',
}


def render_comparison_plot(fr_results: dict, mc_results: dict, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize = (18.0, 10.0), dpi = 140)
    fig.patch.set_facecolor('#0f141c')
    for ax in axes.flat:
        ax.set_facecolor('#171f2c')
        for spine in ax.spines.values():
            spine.set_color('#2e3b4e')
        ax.tick_params(colors = '#cfd8dc')
        ax.xaxis.label.set_color('#eceff1')
        ax.yaxis.label.set_color('#eceff1')
        ax.title.set_color('#ffffff')
        ax.grid(True, color = '#263244', linestyle = '--', linewidth = 0.7, alpha = 0.7)

    gens_fr = np.arange(len(fr_results['baseline']['max_ext'][0]))
    gens_mc = np.arange(len(mc_results['baseline']['max_ext'][0]))

    def plot_curve(ax, gens, data_dict, metric_key):
        for m_key, *_ in MODES:
            arr = np.stack(data_dict[m_key][metric_key], axis = 0)
            mean, sem = arr.mean(axis = 0), arr.std(axis = 0) / np.sqrt(arr.shape[0])
            c = MODE_COLORS[m_key]
            lw = 2.6 if m_key in ('baldwin_mixture', 'baldwin_alternating') else 2.0
            ax.plot(gens, mean, label = MODE_SHORT_LABELS[m_key], color = c, linewidth = lw)
            ax.fill_between(gens, mean - sem, mean + sem, color = c, alpha = 0.16)

    plot_curve(axes[0, 0], gens_fr, fr_results, 'max_ext')
    axes[0, 0].set_title('Four-Rooms Maze: Best Extrinsic Return', fontweight = 'bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best Episode Extrinsic Return')
    axes[0, 0].legend(loc = 'upper left', facecolor = '#171f2c', edgecolor = '#2e3b4e', labelcolor = '#eceff1', fontsize = 8.0)

    plot_curve(axes[0, 1], gens_fr, fr_results, 'mean_cells')
    axes[0, 1].set_title('Four-Rooms Maze: Mean Cell Coverage (12x12 Grid)', fontweight = 'bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Mean Unique Maze Cells Visited')

    plot_curve(axes[0, 2], gens_fr, fr_results, 'mean_lrn')
    axes[0, 2].set_title('Four-Rooms Maze: Predictive Coding Learning (ΔL_PC)', fontweight = 'bold')
    axes[0, 2].set_xlabel('Generation')
    axes[0, 2].set_ylabel('Mean Cumulative PC Learning Gain')

    keys = [m[0] for m in MODES]
    short_names = ['Baseline\n(No TTT)', 'Baldwin\n(Reward)', 'Baldwin\n(PC Only)', 'Baldwin\n(Alternating)', 'Baldwin\n(Mixture)']
    means_innate = [float(np.mean(fr_results[k]['innate_cells'])) for k in keys]
    sems_innate = [float(np.std(fr_results[k]['innate_cells']) / np.sqrt(len(fr_results[k]['innate_cells']))) for k in keys]
    bars = axes[1, 0].bar(
        short_names,
        means_innate,
        yerr = sems_innate,
        color = [MODE_COLORS[k] for k in keys],
        capsize = 4,
        edgecolor = '#eceff1',
        linewidth = 1.0,
        error_kw = {'ecolor': '#eceff1', 'elinewidth': 1.3},
    )
    for bar, val in zip(bars, means_innate):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.6,
            f'{val:.1f}',
            ha = 'center',
            va = 'bottom',
            color = '#ffffff',
            fontweight = 'bold',
            fontsize = 9.0,
        )
    axes[1, 0].set_title('Genetic Assimilation: Zero-TTT Innate Exploration', fontweight = 'bold')
    axes[1, 0].set_ylabel('Cells Visited (TTT Disabled, 8 Perturbed Starts)')
    axes[1, 0].set_ylim(0, max(means_innate) * 1.28)

    plot_curve(axes[1, 1], gens_mc, mc_results, 'mean_bins')
    axes[1, 1].set_title('Honest MountainCar: Phase-Space Bin Coverage', fontweight = 'bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Mean Position Bins Visited (out of 20)')

    plot_curve(axes[1, 2], gens_mc, mc_results, 'solved_count')
    axes[1, 2].set_title('Honest MountainCar: Population Solve Rate (/32)', fontweight = 'bold')
    axes[1, 2].set_xlabel('Generation')
    axes[1, 2].set_ylabel('Individuals Reaching Hilltop (x >= 0.45)')

    fig.suptitle(
        'Evolving Innate Exploration Policies via Baldwinian Selection on Reward + Predictive Coding Learning',
        color = '#ffffff',
        fontsize = 14.5,
        fontweight = 'bold',
        y = 0.985,
    )
    plt.tight_layout(rect = [0, 0, 1, 0.96])
    fig.savefig(out_path, dpi = 140)
    plt.close(fig)
    print(f'Saved comparison figure -> {out_path}')


def render_four_rooms_video(rollouts: dict[str, dict[str, np.ndarray]], out_path: Path, fps: int = 25):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    mode_order = [m[0] for m in MODES]
    horizon = rollouts['baldwin_mixture']['positions'].shape[0] - 1

    fig = plt.figure(figsize = (18.5, 9.0), dpi = 100)
    fig.patch.set_facecolor('#0d1219')
    gs = fig.add_gridspec(2, 10, height_ratios = [1.35, 1.0], hspace = 0.30, wspace = 0.28)

    maze_axes = [fig.add_subplot(gs[0, 2 * i : 2 * i + 2]) for i in range(5)]
    ax_cells = fig.add_subplot(gs[1, 0:5])
    ax_lrn = fig.add_subplot(gs[1, 5:10])

    for ax in (ax_cells, ax_lrn):
        ax.set_facecolor('#151d28')
        for spine in ax.spines.values():
            spine.set_color('#2e3b4e')
        ax.tick_params(colors = '#cfd8dc', labelsize = 9)
        ax.xaxis.label.set_color('#eceff1')
        ax.yaxis.label.set_color('#eceff1')
        ax.title.set_color('#ffffff')
        ax.grid(True, color = '#243040', linestyle = '--', linewidth = 0.7, alpha = 0.7)

    ax_cells.set_xlim(0, horizon)
    max_c = max(int(rollouts[k]['cells'].max()) for k in mode_order)
    ax_cells.set_ylim(0, max(28, max_c + 4))
    ax_cells.set_xlabel('Episode Step t')
    ax_cells.set_ylabel('Unique Maze Cells Visited')
    ax_cells.set_title('Live Spatial Exploration: Unique Maze Cells Visited', fontweight = 'bold', fontsize = 11)

    ax_lrn.set_xlim(0, horizon)
    max_l = max(float(rollouts[k]['learned'].max()) for k in mode_order)
    ax_lrn.set_ylim(0, max(0.05, max_l * 1.15))
    ax_lrn.set_xlabel('Episode Step t')
    ax_lrn.set_ylabel('Cumulative PC Learning Gain')
    ax_lrn.set_title('Live Predictive Coding Learning Progress (∑ ΔL_PC)', fontweight = 'bold', fontsize = 11)

    cell_lines, lrn_lines, cell_dots, lrn_dots = {}, {}, {}, {}
    for k in mode_order:
        c = MODE_COLORS[k]
        lw = 2.5 if k in ('baldwin_mixture', 'baldwin_alternating') else 2.0
        (cell_lines[k],) = ax_cells.plot([], [], color = c, linewidth = lw, label = MODE_SHORT_LABELS[k])
        (lrn_lines[k],) = ax_lrn.plot([], [], color = c, linewidth = lw, label = MODE_SHORT_LABELS[k])
        (cell_dots[k],) = ax_cells.plot([], [], 'o', color = c, markersize = 6)
        (lrn_dots[k],) = ax_lrn.plot([], [], 'o', color = c, markersize = 6)

    ax_cells.legend(loc = 'upper left', facecolor = '#151d28', edgecolor = '#2e3b4e', labelcolor = '#eceff1', fontsize = 8.0)
    ax_lrn.legend(loc = 'upper left', facecolor = '#151d28', edgecolor = '#2e3b4e', labelcolor = '#eceff1', fontsize = 8.0)

    trails, agents, ray_lines, status_texts, cell_patches = {}, {}, {}, {}, {}
    segments = rollouts['baldwin_mixture']['segments']
    ray_dirs = rollouts['baldwin_mixture']['ray_dirs']
    goal = rollouts['baldwin_mixture']['goal']
    goal_r = float(rollouts['baldwin_mixture']['goal_radius'])

    for i, k in enumerate(mode_order):
        ax = maze_axes[i]
        ax.set_facecolor('#131a24')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(MODE_COLORS[k])
            spine.set_linewidth(2.0 if k in ('baldwin_mixture', 'baldwin_alternating') else 1.2)

        ax.set_title(MODE_SHORT_LABELS[k], color = MODE_COLORS[k], fontweight = 'bold', fontsize = 10.0, pad = 8)

        patches_grid = []
        cell_w = 2.0 / 12.0
        for gx in range(12):
            row_patches = []
            for gy in range(12):
                rect = Rectangle(
                    (-1.0 + gx * cell_w, -1.0 + gy * cell_w),
                    cell_w,
                    cell_w,
                    facecolor = MODE_COLORS[k],
                    edgecolor = 'none',
                    alpha = 0.0,
                    zorder = 1,
                )
                ax.add_patch(rect)
                row_patches.append(rect)
            patches_grid.append(row_patches)
        cell_patches[k] = patches_grid

        ax.add_patch(Circle((goal[0], goal[1]), goal_r, facecolor = '#2ecc71', edgecolor = '#a3e4d7', alpha = 0.22, linestyle = '--', linewidth = 1.2, zorder = 2))
        ax.plot(goal[0], goal[1], '*', color = '#2ecc71', markersize = 12, zorder = 3)
        ax.plot(-0.75, -0.75, 's', color = '#90a4ae', markersize = 6, zorder = 3)

        for seg in segments:
            ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color = '#eceff1', linewidth = 2.4, solid_capstyle = 'round', zorder = 4)

        ray_lines[k] = [ax.plot([], [], color = '#f1c40f', linewidth = 0.8, alpha = 0.45, zorder = 5)[0] for _ in range(8)]
        (trails[k],) = ax.plot([], [], color = MODE_COLORS[k], linewidth = 2.2, alpha = 0.9, zorder = 6)
        (agents[k],) = ax.plot([], [], 'o', color = '#ffffff', markeredgecolor = MODE_COLORS[k], markeredgewidth = 2.5, markersize = 8.0, zorder = 7)

        status_texts[k] = ax.text(
            0.03, 0.96, '', transform = ax.transAxes, va = 'top', ha = 'left',
            color = '#eceff1', fontsize = 8.0, fontweight = 'bold',
            bbox = dict(boxstyle = 'round,pad=0.25', facecolor = '#0d1219', edgecolor = '#2e3b4e', alpha = 0.82),
            zorder = 8,
        )

    title_text = fig.suptitle('', color = '#ffffff', fontsize = 13.5, fontweight = 'bold', y = 0.975)
    plt.subplots_adjust(left = 0.04, right = 0.98, top = 0.90, bottom = 0.07)

    frame_steps = list(range(0, horizon + 1, 2))
    if frame_steps[-1] != horizon:
        frame_steps.append(horizon)
    frame_steps.extend([horizon] * 18)

    writer = imageio.get_writer(str(out_path), fps = fps, codec = 'libx264', quality = 8, macro_block_size = 1)
    try:
        for t in frame_steps:
            t_arr = np.arange(t + 1)
            title_text.set_text(f'Switchback Four-Rooms Maze — Evolved Exploration Policies (Step {t:3d} / {horizon})')
            for k in mode_order:
                r = rollouts[k]
                pos, grid = r['positions'][t], r['grids'][t]
                for gx in range(12):
                    for gy in range(12):
                        cell_patches[k][gx][gy].set_alpha(0.26 if grid[gx, gy] else 0.0)

                trails[k].set_data(r['positions'][: t + 1, 0], r['positions'][: t + 1, 1])
                agents[k].set_data([pos[0]], [pos[1]])

                for j in range(8):
                    end_p = pos + ray_dirs[j] * float(r['rays'][t, j])
                    ray_lines[k][j].set_data([pos[0], end_p[0]], [pos[1], end_p[1]])

                g_step = int(r['goal_step'])
                solved_str = f'GOAL! (t={g_step})' if t >= g_step and g_step < horizon else f'Rooms: {int(r["rooms"][t])}/4'
                status_texts[k].set_text(f'Cells: {int(r["cells"][t]):2d} | {solved_str}\nPC Lrn: {float(r["learned"][t]):.3f}')

                cell_lines[k].set_data(t_arr, r['cells'][: t + 1])
                lrn_lines[k].set_data(t_arr, r['learned'][: t + 1])
                cell_dots[k].set_data([t], [r['cells'][t]])
                lrn_dots[k].set_data([t], [r['learned'][t]])

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)

    print(f'Saved Four-Rooms exploration video -> {out_path}')


def render_mountain_car_video(rollouts: dict[str, dict[str, np.ndarray]], out_path: Path, fps: int = 25):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mode_order = [m[0] for m in MODES]
    horizon = rollouts['baldwin_mixture']['pos'].shape[0] - 1

    fig = plt.figure(figsize = (18.5, 7.8), dpi = 100)
    fig.patch.set_facecolor('#0d1219')
    gs = fig.add_gridspec(2, 10, height_ratios = [1.15, 1.0], hspace = 0.34, wspace = 0.28)

    valley_axes = [fig.add_subplot(gs[0, 2 * i : 2 * i + 2]) for i in range(5)]
    ax_phase = fig.add_subplot(gs[1, 0:5])
    ax_lrn = fig.add_subplot(gs[1, 5:10])

    for ax in (ax_phase, ax_lrn):
        ax.set_facecolor('#151d28')
        for spine in ax.spines.values():
            spine.set_color('#2e3b4e')
        ax.tick_params(colors = '#cfd8dc', labelsize = 9)
        ax.xaxis.label.set_color('#eceff1')
        ax.yaxis.label.set_color('#eceff1')
        ax.title.set_color('#ffffff')
        ax.grid(True, color = '#243040', linestyle = '--', linewidth = 0.7, alpha = 0.7)

    ax_phase.set_xlim(-1.25, 0.62)
    ax_phase.set_ylim(-0.078, 0.078)
    ax_phase.axvline(0.45, color = '#2ecc71', linestyle = '--', linewidth = 1.4, alpha = 0.8)
    ax_phase.set_xlabel('Position x')
    ax_phase.set_ylabel('Velocity dx/dt')
    ax_phase.set_title('Phase-Space Exploration Portrait (Raw x vs. dx/dt)', fontweight = 'bold', fontsize = 11)

    ax_lrn.set_xlim(0, horizon)
    max_l = max(float(rollouts[k]['learned'].max()) for k in mode_order)
    ax_lrn.set_ylim(0, max(0.005, max_l * 1.18))
    ax_lrn.set_xlabel('Episode Step t')
    ax_lrn.set_ylabel('Cumulative PC Learning Gain')
    ax_lrn.set_title('Live Predictive Coding Learning Progress (∑ ΔL_PC)', fontweight = 'bold', fontsize = 11)

    phase_lines, phase_dots, lrn_lines, lrn_dots = {}, {}, {}, {}
    for k in mode_order:
        c = MODE_COLORS[k]
        lw = 2.4 if k in ('baldwin_mixture', 'baldwin_alternating') else 1.9
        (phase_lines[k],) = ax_phase.plot([], [], color = c, linewidth = lw, alpha = 0.88, label = MODE_SHORT_LABELS[k])
        (phase_dots[k],) = ax_phase.plot([], [], 'o', color = c, markersize = 6)
        (lrn_lines[k],) = ax_lrn.plot([], [], color = c, linewidth = lw, label = MODE_SHORT_LABELS[k])
        (lrn_dots[k],) = ax_lrn.plot([], [], 'o', color = c, markersize = 6)

    ax_phase.legend(loc = 'upper left', facecolor = '#151d28', edgecolor = '#2e3b4e', labelcolor = '#eceff1', fontsize = 7.8)
    ax_lrn.legend(loc = 'upper left', facecolor = '#151d28', edgecolor = '#2e3b4e', labelcolor = '#eceff1', fontsize = 7.8)

    xs = np.linspace(-1.2, 0.6, 200)
    ys = np.sin(3.0 * xs)
    cars, trails, status_texts = {}, {}, {}

    for i, k in enumerate(mode_order):
        ax = valley_axes[i]
        ax.set_facecolor('#131a24')
        ax.set_xlim(-1.25, 0.65)
        ax.set_ylim(-1.18, 1.25)
        ax.set_xticks([-1.2, -0.5, 0.0, 0.45])
        ax.set_yticks([])
        ax.tick_params(colors = '#90a4ae', labelsize = 8)
        for spine in ax.spines.values():
            spine.set_color(MODE_COLORS[k])
            spine.set_linewidth(2.0 if k in ('baldwin_mixture', 'baldwin_alternating') else 1.2)

        ax.set_title(MODE_SHORT_LABELS[k], color = MODE_COLORS[k], fontweight = 'bold', fontsize = 10.0, pad = 8)
        ax.plot(xs, ys, color = '#90a4ae', linewidth = 2.2, zorder = 2)
        ax.fill_between(xs, -1.18, ys, color = '#1c2636', alpha = 0.85, zorder = 1)
        ax.axvline(0.45, color = '#2ecc71', linestyle = '--', linewidth = 1.4, zorder = 3)
        ax.plot(0.45, np.sin(3.0 * 0.45), '*', color = '#2ecc71', markersize = 13, zorder = 4)

        (trails[k],) = ax.plot([], [], color = MODE_COLORS[k], linewidth = 2.6, alpha = 0.85, zorder = 5)
        (cars[k],) = ax.plot([], [], 'o', color = '#ffffff', markeredgecolor = MODE_COLORS[k], markeredgewidth = 2.6, markersize = 9.5, zorder = 6)

        status_texts[k] = ax.text(
            0.04, 0.94, '', transform = ax.transAxes, va = 'top', ha = 'left',
            color = '#eceff1', fontsize = 8.0, fontweight = 'bold',
            bbox = dict(boxstyle = 'round,pad=0.25', facecolor = '#0d1219', edgecolor = '#2e3b4e', alpha = 0.82),
            zorder = 7,
        )

    title_text = fig.suptitle('', color = '#ffffff', fontsize = 13.5, fontweight = 'bold', y = 0.975)
    plt.subplots_adjust(left = 0.04, right = 0.98, top = 0.89, bottom = 0.08)

    frame_steps = list(range(0, horizon + 1, 2))
    if frame_steps[-1] != horizon:
        frame_steps.append(horizon)
    frame_steps.extend([horizon] * 18)

    writer = imageio.get_writer(str(out_path), fps = fps, codec = 'libx264', quality = 8, macro_block_size = 1)
    try:
        for t in frame_steps:
            t_arr = np.arange(t + 1)
            title_text.set_text(f'Honest Deceptive MountainCar (Raw [x, v]) — Momentum Pumping via PC Selection (Step {t:3d} / {horizon})')
            for k in mode_order:
                r = rollouts[k]
                px = r['pos'][t]
                py = np.sin(3.0 * px)
                hist_x = r['pos'][: t + 1]

                trails[k].set_data(hist_x, np.sin(3.0 * hist_x))
                cars[k].set_data([px], [py])

                g_step = int(r['goal_step'])
                status = f'SOLVED (t={g_step})' if t >= g_step and g_step < horizon else f'x={px:+.2f}'
                status_texts[k].set_text(f'Bins: {int(r["bins"][t]):2d}/20 | {status}\nPC Lrn: {float(r["learned"][t]):.4f}')

                phase_lines[k].set_data(r['pos'][: t + 1], r['vel'][: t + 1])
                phase_dots[k].set_data([px], [r['vel'][t]])
                lrn_lines[k].set_data(t_arr, r['learned'][: t + 1])
                lrn_dots[k].set_data([t], [r['learned'][t]])

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)

    print(f'Saved MountainCar exploration video -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--artifact-dir',
        type = str,
        default = '/Users/philwang/.gemini/antigravity/brain/bd154348-9522-471c-a12b-2836a97765dd',
    )
    args = parser.parse_args()

    out_dir = Path('/Users/philwang/dl/populora')
    artifact_dir = Path(args.artifact_dir)

    fr_results, fr_rollouts = run_four_rooms_experiments(
        seeds = (42, 43, 44), num_generations = 28, pop_size = 32, horizon = 240
    )
    mc_results, mc_rollouts = run_mountain_car_experiments(
        seeds = (42, 43, 44), num_generations = 18, pop_size = 32, horizon = 250
    )

    plot_path = out_dir / 'baldwin_exploration_comparison.png'
    fr_video_path = out_dir / 'exploration_four_rooms.mp4'
    mc_video_path = out_dir / 'exploration_mountain_car.mp4'

    render_comparison_plot(fr_results, mc_results, plot_path)
    render_four_rooms_video(fr_rollouts, fr_video_path, fps = 25)
    render_mountain_car_video(mc_rollouts, mc_video_path, fps = 25)

    if artifact_dir.exists():
        for p in (plot_path, fr_video_path, mc_video_path):
            dest = artifact_dir / p.name
            shutil.copy2(p, dest)
            print(f'Copied {p.name} -> {dest}')


if __name__ == '__main__':
    main()
