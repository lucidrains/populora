# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "einops",
#   "numpy",
#   "fire",
#   "tqdm",
#   "gymnasium",
#   "mujoco",
#   "imageio",
#   "imageio-ffmpeg",
#   "populora",
# ]
# [tool.uv.sources]
# populora = { path = ".", editable = true }
# ///

# evolve a bipedal microduck walking gait from scratch (no Raibert or CPG priors)
# via PopuLoRA + AURORA unsupervised trajectory descriptors + Dominated Novelty Search

from __future__ import annotations

import os
import random
import subprocess
from functools import partial
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import fire
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import mujoco
import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential, Tanh
from einops import einsum, rearrange, repeat
from tqdm import tqdm

torch.set_num_threads(1)

from populora import (
    Aurora,
    Baldwin,
    CosineAnnealingSchedule,
    Memory,
    Population,
    interact_with_env,
    make_action,
    preserve_rng,
    quality_dominated_novelty,
    sketch_trajectories,
    vendi_score,
)
from populora._utils import default, exists, torch_save

# prerequisites

def ensure_deps():
    path = Path(__file__).resolve().parent.parent / 'microduck_rl'
    if path.exists():
        return

    url = 'https://github.com/pollen-robotics/microduck_rl'
    print(f'[deps] cloning {url} -> {path}')
    subprocess.run(['git', 'clone', '--depth', '1', url, str(path)], check = True)

ensure_deps()

# constants

MJCF = Path(__file__).resolve().parent.parent / 'microduck_rl' / 'src' / 'mjlab_microduck' / 'robot' / 'microduck' / 'scene_walk.xml'

CONTROL_HZ = 50
N_SUBSTEPS = 5
PHYSICS_DT = 1.0 / (CONTROL_HZ * N_SUBSTEPS)
CTRL_DT = PHYSICS_DT * N_SUBSTEPS

ACTION_SCALE = 0.5
POLICY_RANGE = (-1., 1.)
KP_MULT = 3.0
KV = 0.1

COMMAND_DIM = 2
COMMAND = (0.15, 0.0)

PROGRESS_WEIGHT = 350.0
CLEARANCE_WEIGHT = 900.0
CLEARANCE_TARGET = 0.008
FOOT_DRAG_WEIGHT = 160.0
ALT_STEP_BONUS = 400.0
SWAY_WEIGHT = 20.0
BODY_ANG_VEL_WEIGHT = 0.04
ACTION_RATE_WEIGHT = 0.10
FALL_PENALTY = 3500.0
NAN_PENALTY = -10.0

RESET_Z_RANGE = (0.0, 0.002)
RESET_JOINT_NOISE = 0.005
RESET_VEL_NOISE = 0.01
MIN_TRUNK_Z = 0.04
MIN_UPRIGHTNESS = 0.35

TRUNK_BODY = 1
FREE_QPOS_DIM = 7
FREE_QVEL_DIM = 6
SPATIAL_DIM = 3
BODY_VEL_DIM = 6
FEET = ('left', 'right')
LEG_JOINTS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
WORLD_UP = np.array([0., 0., 1.])
VIDEO_SIZE = (480, 640)

LORA_TARGETS = (
    'net.encoder.0',
    'net.encoder.2',
    'net.cell.linear_ih',
    'net.cell.linear_hh',
    'net.decoder.0',
    'net.decoder.2',
)

# environment

class MicroduckGaitEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': CONTROL_HZ}

    def __init__(
        self,
        mjcf_path: str | Path = MJCF,
        *,
        horizon: int = 400,
        action_scale: float = ACTION_SCALE,
        kp_mult: float = KP_MULT,
        kv: float = KV,
        fixed_command: tuple[float, float] | None = COMMAND,
        progress_weight: float = PROGRESS_WEIGHT,
        clearance_weight: float = CLEARANCE_WEIGHT,
        sway_weight: float = SWAY_WEIGHT,
        fall_penalty: float = FALL_PENALTY,
        seed: int = 0
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = PHYSICS_DT
        self.model.actuator_gainprm[:, 0] *= kp_mult
        self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
        self.model.actuator_biasprm[:, 2] = -kv
        self.data = mujoco.MjData(self.model)

        self.horizon = horizon
        self.action_scale = action_scale
        self.action_dim = self.model.nu
        self.obs_dim = 11 + 3 * self.action_dim + COMMAND_DIM
        self.fixed_command = fixed_command
        self.progress_weight = progress_weight
        self.clearance_weight = clearance_weight
        self.sway_weight = sway_weight
        self.fall_penalty = fall_penalty
        self.rng = np.random.default_rng(seed)

        stand = self.model.key('STAND')
        self.home_qpos = stand.qpos.copy()
        self.home_ctrl = stand.ctrl.copy()

        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self._foot_geoms = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{s}_foot_collision') for s in FEET])
        self._foot_sites = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_foot') for s in FEET])
        self._foot_bodies = np.array([self.model.geom_bodyid[g] for g in self._foot_geoms])
        self._non_foot_bodies = np.array([b for b in range(1, self.model.nbody) if b not in self._foot_bodies])

        self._command = np.array(default(fixed_command, COMMAND), dtype = np.float64)
        self._last_action = np.zeros(self.action_dim, dtype = np.float32)
        self._prev_foot_xy = np.zeros((2, 2), dtype = np.float64)
        self._max_foot_proj = np.zeros(2, dtype = np.float64)
        self._td_proj = np.zeros(2, dtype = np.float64)
        self._lifted_adv = np.zeros(2, dtype = np.float64)
        self._swing_steps = np.zeros(2, dtype = np.int32)
        self._swing_peak = np.zeros(2, dtype = np.float64)
        self._last_foot = -1
        self._max_deff = 0.0
        self._steps = 0
        self.ssup_steps = 0
        self.foot_clears_mm: list[tuple[float, float]] = []

        self.observation_space = Box(-np.inf, np.inf, shape = (self.obs_dim,), dtype = np.float32)
        self.action_space = Box(*POLICY_RANGE, shape = (self.action_dim,), dtype = np.float32)

    def _velocity(self, local: bool) -> np.ndarray:
        v = np.zeros(BODY_VEL_DIM)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY, v, flg_local = local)
        return v

    def _observe(self) -> np.ndarray:
        g = -self.data.xmat[TRUNK_BODY].reshape(SPATIAL_DIM, SPATIAL_DIM).T @ WORLD_UP
        v_local = self._velocity(local = True)
        v_world = self._velocity(local = False)

        return np.concatenate([
            g,
            v_local[SPATIAL_DIM:],
            v_local[:SPATIAL_DIM],
            v_world[SPATIAL_DIM:SPATIAL_DIM + 2],
            self.data.qpos[FREE_QPOS_DIM:] - self.home_qpos[FREE_QPOS_DIM:],
            self.data.qvel[FREE_QVEL_DIM:],
            self._last_action,
            self._command,
        ]).astype(np.float32)

    def _safe_obs(self) -> np.ndarray:
        return np.concatenate([
            -WORLD_UP,
            np.zeros(8 + 3 * self.action_dim, dtype = np.float32),
            self._command,
        ]).astype(np.float32)

    def foot_clearances(self) -> np.ndarray:
        return np.array([
            mujoco.mj_geomDistance(self.model, self.data, self._floor_geom, int(g), 0.10, None)
            for g in self._foot_geoms
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if exists(seed):
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[SPATIAL_DIM - 1] += self.rng.uniform(*RESET_Z_RANGE)
        self.data.qpos[FREE_QPOS_DIM:] += self.rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size = self.action_dim)
        self.data.qvel[:] = self.rng.uniform(-RESET_VEL_NOISE, RESET_VEL_NOISE, size = self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.start_xy = self.data.qpos[:2].copy()
        self._command[:] = default(self.fixed_command, COMMAND)
        cmd_norm = float(np.linalg.norm(self._command))
        init_dir = (self._command / cmd_norm) if cmd_norm > 0.01 else np.array([1.0, 0.0])

        self._prev_foot_xy[:] = self.data.site_xpos[self._foot_sites, :2]
        self._max_foot_proj[:] = (self._prev_foot_xy - self.start_xy) @ init_dir
        self._td_proj[:] = self._max_foot_proj
        self._lifted_adv.fill(0.0)
        self._swing_steps.fill(0)
        self._swing_peak.fill(0.0)
        self._last_foot = -1
        self._last_action.fill(0.0)
        self._max_deff = 0.0
        self._steps = 0
        self.ssup_steps = 0
        self.foot_clears_mm = []

        return self._observe(), {}

    def step(self, action: np.ndarray):
        action = np.nan_to_num(action, nan = 0.0, posinf = 0.0, neginf = 0.0)
        action_eff = np.clip(action, *POLICY_RANGE)
        action_rate = float(np.mean((action_eff - np.clip(self._last_action, *POLICY_RANGE)) ** 2))

        self.data.ctrl[:] = self.home_ctrl + self.action_scale * action_eff
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        self._last_action[:] = action_eff

        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return self._safe_obs(), NAN_PENALTY - self.fall_penalty, True, False, {}

        q = self.data.qpos
        xmat = self.data.xmat[TRUNK_BODY]
        uprightness = float(xmat[8])
        floor_clip = max(0.0, -float(self.data.xpos[self._non_foot_bodies, 2].min()))

        terminated = bool(uprightness < MIN_UPRIGHTNESS or float(q[2]) < MIN_TRUNK_Z or floor_clip > 0.015)
        truncated = self._steps >= self.horizon

        cmd_speed = float(np.linalg.norm(self._command))
        has_cmd = cmd_speed > 0.01
        cmd_dir = (self._command / cmd_speed) if has_cmd else np.array([1.0, 0.0])

        v_world_xy = self.data.cvel[TRUNK_BODY][3:5]
        disp_xy = q[:2] - self.start_xy

        v_proj = float(v_world_xy @ cmd_dir) if has_cmd else 0.0
        v_lat = float(np.linalg.norm(v_world_xy - v_proj * cmd_dir))
        forward_dist = float(disp_xy @ cmd_dir) if has_cmd else 0.0
        lat_drift = float(np.linalg.norm(disp_xy - forward_dist * cmd_dir))
        heading_cos = float(xmat[0] * cmd_dir[0] + xmat[3] * cmd_dir[1]) if has_cmd else 1.0

        d_eff = min(0.22, max(0.0, forward_dist - 0.03 - 0.55 * lat_drift))
        frontier_vel = max(0.0, d_eff - self._max_deff) / CTRL_DT
        self._max_deff = max(self._max_deff, d_eff)

        ang_vel = self._velocity(local = True)[:SPATIAL_DIM]
        roll_yaw_sq = float(ang_vel[0] ** 2 + ang_vel[2] ** 2)

        pose_gate = float(np.exp(
            -12.0 * max(0.0, 1.0 - uprightness)
            - 10.0 * max(0.0, 1.0 - heading_cos)
            - 55.0 * (lat_drift ** 2)
        ))
        sway_gate = float(np.exp(-10.0 * (v_lat ** 2) - 0.06 * roll_yaw_sq))

        reward = (
            pose_gate * (sway_gate ** 0.5) * (self.progress_weight * frontier_vel + 0.25 * self.progress_weight * min(v_proj, cmd_speed))
            - self.sway_weight * (v_lat ** 2 + 0.80 * lat_drift ** 2)
            - BODY_ANG_VEL_WEIGHT * roll_yaw_sq
            - ACTION_RATE_WEIGHT * action_rate
        )

        # bilateral single-support swing clearance, sole-drag penalty & alternating touchdown

        if has_cmd and self.clearance_weight > 0.0:
            c = self.foot_clearances()
            foot_xy = self.data.site_xpos[self._foot_sites, :2].copy()
            foot_speed = np.linalg.norm(foot_xy - self._prev_foot_xy, axis = 1) / CTRL_DT
            self._prev_foot_xy[:] = foot_xy

            fp = (foot_xy - self.start_xy) @ cmd_dir - 0.50 * lat_drift
            fvel = np.maximum(0.0, fp - self._max_foot_proj) / CTRL_DT
            self._max_foot_proj[:] = np.maximum(self._max_foot_proj, fp)

            u = np.clip((c + 0.0003) / CLEARANCE_TARGET, 0.0, 1.0)
            opp_g = np.clip(1.0 - np.maximum(0.0, c[::-1] - 0.0005) / 0.0015, 0.0, 1.0)
            swing_vel = u * opp_g * fvel

            prev_min = float(self._lifted_adv.min())
            self._lifted_adv += swing_vel * CTRL_DT
            sym_vel = (float(self._lifted_adv.min()) - prev_min) / CTRL_DT

            c_scale = self.clearance_weight / CLEARANCE_WEIGHT
            drag_ramp = float(np.clip((forward_dist - 0.16) / 0.12, 0.0, 1.0))
            reward += (
                pose_gate * self.clearance_weight * (float(swing_vel.sum()) + 1.8 * sym_vel)
                - c_scale * FOOT_DRAG_WEIGHT * drag_ramp * float(((1.0 - u) ** 2) @ foot_speed)
            )

            for i in range(2):
                if c[i] > 0.0010 and c[1 - i] <= 0.0008:
                    self._swing_steps[i] += 1
                    self._swing_peak[i] = max(float(self._swing_peak[i]), float(c[i]))
                elif c[i] <= 0.0006:
                    if self._swing_steps[i] >= 2 and self._swing_peak[i] >= 0.0015 and fp[i] > max(fp[1 - i] + 0.004, self._td_proj[i] + 0.008) and self._last_foot != i:
                        reward += pose_gate * c_scale * ALT_STEP_BONUS * min(self._swing_steps[i] / 5.0, 1.0) * min(self._swing_peak[i] / CLEARANCE_TARGET, 1.0)
                        self._td_proj[i], self._last_foot = fp[i], i
                    self._swing_steps[i], self._swing_peak[i] = 0, 0.0

            opp_grounded = (c <= 0.0008)[::-1]
            self.ssup_steps += int(((c > 0.0010) & opp_grounded).any())
            self.foot_clears_mm.append(tuple(float(v) for v in np.where(opp_grounded, np.maximum(0.0, c) * 1000.0, 0.0)))

        if terminated:
            reward -= self.fall_penalty

        return self._observe(), float(reward), terminated, truncated, {}

# policy

class ManualGRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = Linear(input_size, 3 * hidden_size)
        self.linear_hh = Linear(hidden_size, 3 * hidden_size, bias = False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        i_r, i_z, i_n = self.linear_ih(x).chunk(3, dim = -1)
        h_r, h_z, h_n = self.linear_hh(h).chunk(3, dim = -1)

        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()
        return (1.0 - z) * n + z * h

class GRUMLPPolicy(Module):
    def __init__(self, obs_dim: int = 55, action_dim: int = 14, hidden_dim: int = 128, head_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.encoder = Sequential(
            Linear(obs_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, hidden_dim),
            Tanh(),
        )
        self.cell = ManualGRUCell(hidden_dim, hidden_dim)
        self.decoder = Sequential(
            Linear(hidden_dim, head_dim),
            Tanh(),
            Linear(head_dim, action_dim),
        )
        self.log_std = Parameter(torch.zeros(action_dim))

        with torch.no_grad():
            self.cell.linear_hh.weight.mul_(0.08)
            self.cell.linear_ih.bias.zero_()
            self.cell.linear_ih.bias[:hidden_dim].fill_(0.8)
            self.cell.linear_ih.bias[hidden_dim:2 * hidden_dim].uniform_(0.3, 1.3)
            for m in (self.encoder[0], self.encoder[2], self.decoder[0], self.decoder[2]):
                m.bias.zero_()
            self.decoder[2].weight.zero_()

    def forward(self, mem: Tensor, obs: Tensor) -> tuple[Tensor, Tensor]:
        next_mem = self.cell(self.encoder(obs), mem)
        mean = self.decoder(next_mem)
        return torch.cat((mean, self.log_std.expand_as(mean)), dim = -1), next_mem

# helpers

@torch.no_grad()
def merged_state_dict(pop: Population, idx: int) -> dict:
    _, (w_down, w_up) = pop.individual_weights(idx)
    state = {k: v.detach().clone() for k, v in pop.model.state_dict().items()}
    for path in pop.lora_targets:
        key = path.replace('.', '_')
        delta = einsum(w_up[key].float(), w_down[key].float(), 'o r, i r -> o i')
        w = state[f'{path}.weight']
        w.add_(delta.to(w.dtype))
    return state

def load_model(state: dict, hidden_dim: int = 128, head_dim: int = 128) -> Memory:
    policy = GRUMLPPolicy(55, 14, hidden_dim, head_dim)
    model = Memory(policy, init_memory = state.get('init_memory', torch.zeros(1, hidden_dim)))
    model.load_state_dict(state, strict = False)
    return model

def _episode_stats(env: MicroduckGaitEnv, ret: float, rolls: list[float]) -> tuple:
    disp = env.data.qpos[:2] - env.start_xy
    tail = env.foot_clears_mm[min(20, max(0, len(env.foot_clears_mm) - 1)):]
    if tail:
        arr = np.array(tail)
        clear_mm = float(np.percentile(arr.max(axis = 1), 95))
        min_lr_mm = float(min(np.percentile(arr[:, 0], 95), np.percentile(arr[:, 1], 95)))
    else:
        clear_mm = min_lr_mm = 0.0
    ssup_pct = 100.0 * env.ssup_steps / max(1, env._steps)
    return (ret, env._steps, float(disp[0]), abs(float(disp[1])), float(np.ptp(rolls)), clear_mm, min_lr_mm, ssup_pct)

def _aggregate_stats(outs: list[tuple]) -> dict:
    cols = list(zip(*outs))
    keys = ('ret', 'steps', 'x', 'abs_y', 'roll', 'clear_mm', 'min_lr_mm', 'ssup_pct')
    res = {k: float(np.mean(c)) for k, c in zip(keys, cols)}
    res['min_steps'] = float(np.min(cols[1]))
    res['min_x'] = float(np.min(cols[2]))
    return res

@torch.no_grad()
def evaluate_candidates(
    envs: list[MicroduckGaitEnv],
    pop: Population,
    candidates: list[int],
    action_fn,
    seeds: tuple[int, ...] = (0, 1, 2),
    horizon: int = 400
) -> dict[int, dict]:
    n_cands, n_seeds = len(candidates), len(seeds)
    total = n_cands * n_seeds
    inds = repeat(torch.tensor(candidates, device = pop.device, dtype = torch.long), 'c -> (c s)', s = n_seeds)
    obs_buf = np.zeros((total, envs[0].obs_dim), dtype = np.float32)
    rets = np.zeros(total, dtype = np.float64)
    done = np.zeros(total, dtype = bool)
    rolls = [[] for _ in range(total)]

    for k in range(total):
        obs_buf[k], _ = envs[k].reset(seed = seeds[k % n_seeds])

    mem = repeat(pop.model.init_memory.to(pop.device), '1 d -> b d', b = total).clone()

    with pop.routing_cache():
        for _ in range(horizon):
            if done.all():
                break
            logits, mem = pop(mem, torch.from_numpy(obs_buf), individuals = inds, eval_and_no_grad = True)
            acts = action_fn(logits).clamp(*POLICY_RANGE).numpy()
            for k in range(total):
                if done[k]:
                    continue
                env = envs[k]
                obs_buf[k], r, term, trunc, _ = env.step(acts[k])
                rets[k] += r
                xmat = env.data.xmat[TRUNK_BODY]
                rolls[k].append(float(np.degrees(np.arctan2(xmat[7], xmat[8]))))
                if term or trunc:
                    done[k] = True

    return {
        idx: _aggregate_stats([_episode_stats(envs[ci * n_seeds + si], rets[ci * n_seeds + si], rolls[ci * n_seeds + si]) for si in range(n_seeds)])
        for ci, idx in enumerate(candidates)
    }

@torch.no_grad()
def render_gait(
    env: MicroduckGaitEnv,
    model: Memory,
    action_fn,
    video_path: str | Path,
    horizon: int = 400,
    seed: int = 0
):
    renderer = mujoco.Renderer(env.model, height = VIDEO_SIZE[0], width = VIDEO_SIZE[1])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = TRUNK_BODY
    camera.distance = 0.72
    camera.azimuth = 135.0
    camera.elevation = -18.0

    obs, _ = env.reset(seed = seed)
    mem = model.init_memory.clone()
    frames = []

    for _ in range(horizon):
        logits, mem = model(mem, torch.from_numpy(obs).unsqueeze(0))
        action = action_fn(logits).squeeze(0).clamp(*POLICY_RANGE).numpy()
        obs, _, term, trunc, _ = env.step(action)

        renderer.update_scene(env.data, camera = camera)
        frames.append(renderer.render().copy())
        if term or trunc:
            break

    out = Path(video_path)
    out.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(out, frames, fps = 25)
    renderer.close()

@torch.no_grad()
def extract_gait_trajectories(trajs: list[tuple[Tensor, Tensor]]) -> list[Tensor]:
    # planar trunk path, velocity, orientation, sway rates, and the 10 leg joints
    # (excluding the 4 neck/head DoFs so novelty cannot be gamed by head wagging)

    out = []
    for obs, _ in trajs:
        if obs.shape[0] == 0:
            out.append(torch.zeros((1, 20)))
            continue
        v_xy = obs[:, 9:11]
        disp_xy = (v_xy * CTRL_DT).cumsum(dim = 0)
        out.append(torch.cat((disp_xy, v_xy, obs[:, :3], obs[:, 6:9], obs[:, 11:25][:, LEG_JOINTS]), dim = -1))
    return out

@torch.no_grad()
def init_population_(pop: Population, has_warmstart: bool = False) -> dict:
    pop.repopulate_(save_champion = False, std_down = None, std_up = 0.04 if has_warmstart else 1.0)
    if has_warmstart:
        return pop.backup_individual(0)

    # orthogonalize recurrent low-rank basis and seed oscillatory skew-symmetric modes + decoder gain spectrum

    p, r = pop.pop_size, pop.low_rank
    q, _ = torch.linalg.qr(pop.weight_down['net_cell_linear_hh'].data)
    pop.weight_down['net_cell_linear_hh'].data.copy_(q)

    g = torch.randn(1, r, r) + 0.35 * torch.randn(p, r, r)
    w = g - g.transpose(1, 2)
    w = w / w.norm(dim = (-2, -1), keepdim = True).clamp(min = 1e-6) * rearrange(torch.linspace(3.6, 4.6, p), 'p -> p 1 1')
    s = w + rearrange(torch.linspace(1.38, 1.18, p), 'p -> p 1 1') * torch.eye(r)

    pop.weight_up['net_cell_linear_hh'].data[:, :256, :].mul_(0.10)
    pop.weight_up['net_cell_linear_hh'].data[:, 256:384, :].copy_(torch.bmm(q, s))

    for k, scale in (
        ('net_cell_linear_ih', 0.12),
        ('net_encoder_0', 0.14),
        ('net_encoder_2', 0.14),
        ('net_decoder_0', 0.50),
    ):
        pop.weight_up[k].data.mul_(scale)

    pop.weight_up['net_decoder_2'].data.mul_(rearrange(torch.linspace(0.18, 0.44, p), 'p -> p 1 1'))
    return pop.backup_individual(0)

# train

def train(
    pop_size: int = 64,
    low_rank: int = 8,
    num_generations: int = 140,
    horizon: int = 400,
    num_envs: int = 4,
    num_episodes: int = 1,
    seed: int = 1,
    eval_seeds: tuple[int, ...] = (0, 1, 2),
    progress_weight: float = PROGRESS_WEIGHT,
    clearance_weight: float = CLEARANCE_WEIGHT,
    sway_weight: float = SWAY_WEIGHT,
    fall_penalty: float = FALL_PENALTY,
    use_pso: bool = True,
    use_aurora_dns: bool = True,
    aurora_latent_dim: int = 8,
    aurora_hidden_dim: int = 64,
    aurora_steps: int = 8,
    dns_k: int = 5,
    quality_weight: float = 0.65,
    quality_quantile: float = 0.30,
    quality_sharpness: float = 2.0,
    niche_k: int = 8,
    predictive_coding: bool = True,
    render: bool = True,
    resume_checkpoint: str | None = None,
    output_checkpoint: str | None = None,
    video_path: str | None = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    suffix = ('' if use_pso else '_no_pso') + ('_aurora_dns' if use_aurora_dns else '')
    output_checkpoint = default(output_checkpoint, f'checkpoints-microduck/gru_mlp_from_scratch{suffix}.pt')
    video_path = default(video_path, f'videos-microduck/gru_mlp_from_scratch{suffix}.mp4')

    has_warmstart = exists(resume_checkpoint) and Path(resume_checkpoint).exists()
    policy = load_model(torch.load(resume_checkpoint, map_location = 'cpu', weights_only = False)) if has_warmstart else Memory(
        GRUMLPPolicy(obs_dim = 55, action_dim = 14, hidden_dim = 128, head_dim = 128),
        init_memory = 0.18 * torch.randn(1, 128)
    )

    wrapped_model = Baldwin(
        policy,
        latent_layer = 'net.encoder.2',
        lora_memory = ('net.encoder.0', 'net.encoder.2'),
        action_dim = 14,
        lora_lr = 5e-4,
        predictor_lr = 3e-3,
        update_every = 8,
        trust_region = 0.10,
        reward_weight = 1.0,
        predictor_learning_weight = 8.0,
        assimilate_tau = 0.05
    ) if predictive_coding else policy

    make_env = partial(
        MicroduckGaitEnv,
        horizon = horizon,
        progress_weight = progress_weight,
        clearance_weight = clearance_weight,
        sway_weight = sway_weight,
        fall_penalty = fall_penalty
    )
    eval_envs = [make_env() for _ in range((min(8, pop_size) + 5) * len(eval_seeds))]

    action_fn = make_action(
        'squashed_gaussian',
        sample = False,
        temperature = 1.0,
        bounds = POLICY_RANGE,
        min_log_std = -5.0,
        max_log_std = 0.5
    )

    interactor = interact_with_env(
        make_env,
        num_envs = num_envs,
        seed = seed,
        to_range = POLICY_RANGE,
        standardize = False
    )

    pop = Population(
        wrapped_model,
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = LORA_TARGETS,
        eval_seed = seed
    )

    aurora = Aurora(
        latent_dim = aurora_latent_dim,
        hidden_dim = aurora_hidden_dim,
        batch_size = min(64, pop_size),
        update_steps = aurora_steps,
        max_horizon = horizon
    ) if use_aurora_dns else None

    eps_schedule = CosineAnnealingSchedule(eps_min = 0.005, eps_max = 0.015, period = 35, decay = 0.82)

    champ_backup = init_population_(pop, has_warmstart)
    champ_state = merged_state_dict(pop, 0)
    champ_metrics = evaluate_candidates(eval_envs, pop, [0], action_fn, seeds = eval_seeds, horizon = horizon)[0]
    best_ret = float('-inf')
    prev_elites = [0] if has_warmstart else []

    print(
        f"[init] warmstart={has_warmstart} | pso={use_pso} | aurora_dns={use_aurora_dns} | pc={predictive_coding} | "
        f"ret={champ_metrics['ret']:.1f} | steps={champ_metrics['steps']:.1f} (min={champ_metrics['min_steps']:.0f}) | "
        f"x={champ_metrics['x']:+.3f}m | |y|={champ_metrics['abs_y']:.3f}m | roll={champ_metrics['roll']:.1f}deg",
        flush = True
    )

    pbar = tqdm(range(num_generations), desc = f'evolving microduck gait (pso={use_pso}, aurora_dns={use_aurora_dns})')

    for gen in pbar:
        fits, trajs = interactor.evaluate(
            pop,
            action = action_fn,
            horizon = horizon,
            num_episodes = num_episodes,
            seed = seed + gen,
            return_trajectories = True
        )
        gait_trajs = extract_gait_trajectories(trajs)

        if exists(aurora):
            with preserve_rng():
                aur_loss = aurora.fit(gait_trajs)
                descriptors = aurora.encode(gait_trajs).to(pop.device)
        else:
            aur_loss = 0.0
            descriptors = sketch_trajectories(gait_trajs, max_horizon = horizon, device = pop.device)

        vendi = vendi_score(descriptors)
        pop.restore_individual(0, champ_backup)

        candidates = sorted(set(fits.topk(min(8, pop_size)).indices.tolist()) | set(prev_elites) | {0})
        cand_map = evaluate_candidates(eval_envs, pop, candidates, action_fn, seeds = eval_seeds, horizon = horizon)
        for idx, cm in cand_map.items():
            fits[idx] = cm['ret']

        ranked = sorted(((cm['ret'], idx, cm) for idx, cm in cand_map.items()), key = lambda x: x[0], reverse = True)
        prev_elites = [idx for _, idx, _ in ranked[:4]]
        _, best_idx, m = ranked[0]

        viable_count = int((fits >= 0.0).sum().item())
        improved = (m['min_steps'] >= horizon) and (m['ret'] > best_ret)
        if improved:
            best_ret = m['ret']
            champ_state = merged_state_dict(pop, best_idx)
            champ_metrics = m
            champ_backup = pop.backup_individual(best_idx)
            torch_save(champ_state, output_checkpoint)

        aur_str = f' | aur={aur_loss:.3f}' if use_aurora_dns else ''
        pbar.write(
            f"gen {gen:03d} | ret={m['ret']:7.1f} (best={best_ret:7.1f}) | "
            f"x={m['x']:+.3f}m (min={m['min_x']:+.3f}m) | |y|={m['abs_y']:.3f}m | "
            f"clear={m['clear_mm']:4.1f}mm (minLR={m['min_lr_mm']:4.1f}) | ssup={m['ssup_pct']:3.0f}% | "
            f"roll={m['roll']:.1f}deg | viable={viable_count:2d}/{pop_size} | vendi={vendi:4.1f}{aur_str}"
            + (' *' if improved else '')
        )

        if gen < num_generations - 1:
            comp_fits = quality_dominated_novelty(
                fits,
                descriptors,
                k = dns_k,
                weight = quality_weight,
                quantile = quality_quantile,
                sharpness = quality_sharpness
            ) if use_aurora_dns else None

            pop.evolve_(
                fits,
                competition_fitnesses = comp_fits,
                descriptors = descriptors if use_aurora_dns else None,
                raw_elites = min(4, pop_size // 2) if use_aurora_dns else None,
                survive_frac = 0.70,
                elite_frac = 0.15,
                num_elites = min(4, pop_size // 2),
                selection_type = 'deterministic',
                parent_selection_type = 'assortative' if use_aurora_dns else 'tournament',
                tournament_size = 4,
                niche_k = niche_k,
                crossover_type = 'extrapolative',
                eta_min = 0.20,
                eta_max = 1.25,
                mutation_type = 'full_gaussian',
                epsilon = eps_schedule,
                weight_decay = 1e-4,
                consolidate_alpha = 0.0,
                swarm_type = 'firefly' if use_pso else None,
                swarm_bottom_frac = 0.60 if use_aurora_dns else 1.0,
                swarm_kwargs = dict(
                    beta0 = 0.50,
                    gamma = 1.0,
                    alpha = 0.035,
                    align_lora_sign = True,
                    align_lora_procrustes = True,
                    preserve_norm = True,
                    normalize_weights = True,
                    preserve_distance_decay = True,
                    fitness_weighted = True,
                    scale_invariant_distance = True
                )
            )
            pop.restore_individual(0, champ_backup)

    torch_save(champ_state, output_checkpoint)

    if render:
        with preserve_rng():
            render_gait(eval_envs[0], load_model(champ_state), action_fn, video_path = video_path, horizon = horizon, seed = 0)
        print(f'saved video to {video_path}')

    interactor.close()
    for e in eval_envs:
        e.close()
    return champ_metrics

if __name__ == '__main__':
    fire.Fire(train)
