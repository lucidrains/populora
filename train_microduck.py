# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
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

from __future__ import annotations

import copy
import subprocess
from functools import partial
from pathlib import Path

import fire
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import mujoco
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from populora import Baldwin, Memory, Population, interact_with_env, make_action
from populora._utils import default, exists
from populora.population import _iter_adapters, _lora_delta

# prerequisites: clone the robot model repo next to this repo if not already present

def ensure_deps():
    root = Path(__file__).resolve().parent.parent
    path = root / 'microduck_rl'

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
SWAY_WEIGHT = 15.0
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
WORLD_UP = np.array([0., 0., 1.])

VIDEO_SIZE = (480, 640)

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
        sway_weight: float = SWAY_WEIGHT,
        fall_penalty: float = FALL_PENALTY,
        seed: int = 0,
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
        self.sway_weight = sway_weight
        self.fall_penalty = fall_penalty
        self.rng = np.random.default_rng(seed)

        stand = self.model.key('STAND')
        self.home_qpos = stand.qpos.copy()
        self.home_ctrl = stand.ctrl.copy()

        self._foot_geoms = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_foot_collision')
            for side in FEET
        ])
        self._foot_bodies = np.array([self.model.geom_bodyid[g] for g in self._foot_geoms])
        self._non_foot_bodies = np.array([b for b in range(1, self.model.nbody) if b not in self._foot_bodies])

        self._command = np.array(default(fixed_command, COMMAND), dtype = np.float64)
        self._last_action = np.zeros(self.action_dim, dtype = np.float32)
        self._max_deff = 0.0
        self._steps = 0

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
        self._last_action.fill(0.0)
        self._max_deff = 0.0
        self._steps = 0

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
        uprightness = float(self.data.xmat[TRUNK_BODY][8])
        trunk_z = float(q[2])
        v_world_xy = self.data.cvel[TRUNK_BODY][3:5]

        body_z = self.data.xpos[self._non_foot_bodies, 2]
        floor_clip_depth = max(0.0, -float(body_z.min()))

        terminated = bool(uprightness < MIN_UPRIGHTNESS or trunk_z < MIN_TRUNK_Z or floor_clip_depth > 0.015)
        truncated = self._steps >= self.horizon

        cmd_speed = float(np.linalg.norm(self._command))
        disp_xy = q[:2] - self.start_xy
        xmat = self.data.xmat[TRUNK_BODY]

        if cmd_speed > 0.01:
            cmd_dir = self._command / cmd_speed
            v_proj = float(np.dot(v_world_xy, cmd_dir))
            v_lat = float(np.linalg.norm(v_world_xy - v_proj * cmd_dir))
            forward_dist = float(np.dot(disp_xy, cmd_dir))
            lat_drift = float(np.linalg.norm(disp_xy - forward_dist * cmd_dir))
            heading_cos = float(xmat[0] * cmd_dir[0] + xmat[3] * cmd_dir[1])
        else:
            v_proj, v_lat = 0.0, float(np.linalg.norm(v_world_xy))
            forward_dist, lat_drift = 0.0, float(np.linalg.norm(disp_xy))
            heading_cos = 1.0

        d_eff = max(0.0, forward_dist - 0.03 - 0.35 * lat_drift)
        frontier_vel = max(0.0, d_eff - self._max_deff) / CTRL_DT
        self._max_deff = max(self._max_deff, d_eff)

        ang_vel = self._velocity(local = True)[:SPATIAL_DIM]
        roll_yaw_sq = float(ang_vel[0] ** 2 + ang_vel[2] ** 2)

        pose_gate = float(np.exp(
            -12.0 * max(0.0, 1.0 - uprightness)
            - 10.0 * max(0.0, 1.0 - heading_cos)
            - 20.0 * (lat_drift ** 2)
        ))
        sway_gate = float(np.exp(-15.0 * (v_lat ** 2) - 0.06 * roll_yaw_sq))

        reward = (
            pose_gate * (sway_gate ** 0.5) * (self.progress_weight * frontier_vel + 0.35 * self.progress_weight * v_proj)
            + pose_gate * sway_gate * (150.0 * d_eff * (1.0 + 2.0 * d_eff))
            - self.sway_weight * (v_lat ** 2 + 0.25 * lat_drift ** 2)
            - BODY_ANG_VEL_WEIGHT * roll_yaw_sq
            - ACTION_RATE_WEIGHT * action_rate
        )

        if terminated:
            reward -= self.fall_penalty

        return self._observe(), float(reward), terminated, truncated, {}

# policy

class ManualGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size, bias = False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        i_r, i_z, i_n = self.linear_ih(x).chunk(3, dim = -1)
        h_r, h_z, h_n = self.linear_hh(h).chunk(3, dim = -1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

class GRUMLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 55, action_dim: int = 14, hidden_dim: int = 128, head_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cell = ManualGRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.Tanh(),
            nn.Linear(head_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        with torch.no_grad():
            self.cell.linear_hh.weight.mul_(0.08)
            self.cell.linear_ih.bias.zero_()
            self.cell.linear_ih.bias[:hidden_dim].fill_(0.8)
            self.cell.linear_ih.bias[hidden_dim:2 * hidden_dim].uniform_(0.3, 1.3)
            for m in (self.encoder[0], self.encoder[2], self.decoder[0], self.decoder[2]):
                m.bias.zero_()
            self.decoder[2].weight.zero_()

    def forward(self, mem: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(obs)
        next_mem = self.cell(x, mem)
        mean = self.decoder(next_mem)
        return torch.cat([mean, self.log_std.expand_as(mean)], dim = -1), next_mem

# evaluation & rendering

def export_champion_state_dict(pop: Population, individual_idx: int) -> dict:
    champion = copy.deepcopy(pop.model)
    for path, _, w_down, w_up in _iter_adapters(pop):
        linear = champion.get_submodule(path)
        delta = _lora_delta(w_down[individual_idx], w_up[individual_idx])
        linear.weight.data.add_(delta.to(linear.weight.dtype))
    return champion.state_dict()

def load_model(state: dict, hidden_dim: int = 128, head_dim: int = 128) -> Memory:
    policy = GRUMLPPolicy(55, 14, hidden_dim, head_dim)
    model = Memory(policy, init_memory = state.get('init_memory', torch.zeros(1, hidden_dim)))
    model.load_state_dict(state, strict = False)
    return model

@torch.no_grad()
def evaluate_gait(
    env: MicroduckGaitEnv,
    model: Memory,
    action_fn,
    seeds: tuple[int, ...] = (0, 1, 2),
    horizon: int = 400,
) -> dict:
    outs = []
    for seed in seeds:
        obs, _ = env.reset(seed = seed)
        mem = model.init_memory.clone()
        start_xy = env.data.qpos[:2].copy()
        ret, steps, lat_sway = 0.0, 0, 0.0
        rolls = []

        for _ in range(horizon):
            logits, mem = model(mem, torch.tensor(obs, dtype = torch.float32).unsqueeze(0))
            action = action_fn(logits).squeeze(0).clamp(*POLICY_RANGE).numpy()
            obs, r, term, trunc, _ = env.step(action)
            ret += float(r)
            steps += 1
            lat_sway += abs(float(env.data.cvel[TRUNK_BODY][4])) * CTRL_DT
            xmat = env.data.xmat[TRUNK_BODY]
            rolls.append(float(np.degrees(np.arctan2(xmat[7], xmat[8]))))
            if term or trunc:
                break

        dx = float(env.data.qpos[0] - start_xy[0])
        dy = float(env.data.qpos[1] - start_xy[1])
        outs.append((ret, steps, dx, dy, lat_sway, float(np.ptp(rolls))))

    return dict(
        ret = float(np.mean([o[0] for o in outs])),
        steps = float(np.mean([o[1] for o in outs])),
        min_steps = float(np.min([o[1] for o in outs])),
        x = float(np.mean([o[2] for o in outs])),
        min_x = float(np.min([o[2] for o in outs])),
        abs_y = float(np.mean([abs(o[3]) for o in outs])),
        sway = float(np.mean([o[4] for o in outs])),
        roll = float(np.mean([o[5] for o in outs])),
    )

@torch.no_grad()
def render_gait(
    env: MicroduckGaitEnv,
    model: Memory,
    action_fn,
    video_path: str | Path,
    horizon: int = 400,
    seed: int = 0,
):
    renderer = mujoco.Renderer(env.model, height = VIDEO_SIZE[0], width = VIDEO_SIZE[1])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = TRUNK_BODY
    camera.distance = 0.80
    camera.azimuth = 135.0
    camera.elevation = -22.0

    obs, _ = env.reset(seed = seed)
    mem = model.init_memory.clone()
    frames = []

    for _ in range(horizon):
        logits, mem = model(mem, torch.tensor(obs, dtype = torch.float32).unsqueeze(0))
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

# train

@torch.no_grad()
def snapshot_lora(pop: Population, idx: int = 0) -> dict:
    return {k: (pop.weight_down[k].data[idx].clone(), pop.weight_up[k].data[idx].clone()) for k in pop.weight_down}

@torch.no_grad()
def restore_lora_(pop: Population, lora: dict, idx: int = 0):
    for k, (wd, wu) in lora.items():
        pop.weight_down[k].data[idx].copy_(wd)
        pop.weight_up[k].data[idx].copy_(wu)

@torch.no_grad()
def init_population_(pop: Population, state: dict, has_resume: bool = False) -> dict:
    pop.model.load_state_dict(state, strict = False)
    if hasattr(pop, 'baldwin') and exists(pop.baldwin._frozen_params):
        pop.baldwin._frozen_params = [(p, p.data.detach().clone()) for p in pop.model.parameters()]

    pop.repopulate_(save_champion = False, std_down = None, std_up = 0.04 if has_resume else 1.0)
    if has_resume:
        return snapshot_lora(pop, 0)

    p, r = pop.pop_size, pop.low_rank
    q, _ = torch.linalg.qr(pop.weight_down['net_cell_linear_hh'].data)
    pop.weight_down['net_cell_linear_hh'].data.copy_(q)

    g = torch.randn(1, r, r) + 0.35 * torch.randn(p, r, r)
    w = g - g.transpose(1, 2)
    w = w / w.norm(dim = (-2, -1), keepdim = True).clamp(min = 1e-6) * torch.linspace(3.6, 4.6, p).view(-1, 1, 1)
    s = w + torch.linspace(1.38, 1.18, p).view(-1, 1, 1) * torch.eye(r)

    pop.weight_up['net_cell_linear_hh'].data[:, :256, :].mul_(0.10)
    pop.weight_up['net_cell_linear_hh'].data[:, 256:384, :].copy_(torch.bmm(q, s))

    for k, scale in (
        ('net_cell_linear_ih', 0.12),
        ('net_encoder_0', 0.14),
        ('net_encoder_2', 0.14),
        ('net_decoder_0', 0.50),
        ('net_decoder_2', 0.44),
    ):
        pop.weight_up[k].data.mul_(scale)

    return snapshot_lora(pop, 0)

def train(
    pop_size: int = 64,
    low_rank: int = 8,
    num_generations: int = 75,
    horizon: int = 400,
    num_envs: int = 16,
    num_episodes: int = 1,
    seed: int = 0,
    eval_seeds: tuple[int, ...] = (0, 1, 2),
    progress_weight: float = PROGRESS_WEIGHT,
    sway_weight: float = SWAY_WEIGHT,
    fall_penalty: float = FALL_PENALTY,
    use_pso: bool = True,
    predictive_coding: bool = True,
    render: bool = True,
    resume_checkpoint: str | None = None,
    output_checkpoint: str | None = None,
    video_path: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    suffix = '' if use_pso else '_no_pso'
    output_checkpoint = default(output_checkpoint, f'checkpoints-microduck/gru_mlp_from_scratch{suffix}.pt')
    video_path = default(video_path, f'videos-microduck/gru_mlp_from_scratch{suffix}.mp4')

    has_resume = exists(resume_checkpoint) and Path(resume_checkpoint).exists()
    if has_resume:
        policy = load_model(torch.load(resume_checkpoint, map_location = 'cpu', weights_only = False))
    else:
        policy = Memory(
            GRUMLPPolicy(obs_dim = 55, action_dim = 14, hidden_dim = 128, head_dim = 128),
            init_memory = 0.18 * torch.randn(1, 128),
        )

    lora_targets = [
        'net.encoder.0',
        'net.encoder.2',
        'net.cell.linear_ih',
        'net.cell.linear_hh',
        'net.decoder.0',
        'net.decoder.2',
    ]

    wrapped_model = Baldwin(
        copy.deepcopy(policy),
        latent_layer = 'net.encoder.2',
        lora_memory = ('net.encoder.0', 'net.encoder.2'),
        action_dim = 14,
        lora_lr = 5e-4,
        predictor_lr = 3e-3,
        update_every = 8,
        trust_region = 0.10,
        reward_weight = 1.0,
        predictor_learning_weight = 8.0,
        assimilate_tau = 0.05,
    ) if predictive_coding else copy.deepcopy(policy)

    make_env = partial(
        MicroduckGaitEnv,
        horizon = horizon,
        progress_weight = progress_weight,
        sway_weight = sway_weight,
        fall_penalty = fall_penalty,
    )
    eval_env = make_env()

    action_fn = make_action(
        'squashed_gaussian',
        sample = False,
        temperature = 1.0,
        bounds = POLICY_RANGE,
        min_log_std = -5.0,
        max_log_std = 0.5,
    )

    interactor = interact_with_env(
        make_env,
        num_envs = num_envs,
        seed = seed,
        to_range = POLICY_RANGE,
        standardize = False,
    )

    pop = Population(
        wrapped_model,
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = lora_targets,
        eval_seed = seed,
    )

    champ_lora = init_population_(pop, copy.deepcopy(policy.state_dict()), has_resume)
    champ_state = export_champion_state_dict(pop, 0)
    champ_model = load_model(champ_state)
    champ_metrics = evaluate_gait(eval_env, champ_model, action_fn, seeds = eval_seeds, horizon = horizon)
    best_ret = float('-inf')

    print(
        f"[init] resume={has_resume} | pso={use_pso} | pc={predictive_coding} | "
        f"ret={champ_metrics['ret']:.1f} | steps={champ_metrics['steps']:.1f} (min={champ_metrics['min_steps']:.0f}) | "
        f"x={champ_metrics['x']:+.3f}m | |y|={champ_metrics['abs_y']:.3f}m | sway={champ_metrics['sway']:.3f}m | "
        f"roll={champ_metrics['roll']:.1f}deg",
        flush = True,
    )

    prev_elites = [0] if has_resume else []
    pbar = tqdm(range(num_generations), desc = f'evolving microduck gait (pso={use_pso})')

    for gen in pbar:
        fits = interactor.evaluate(
            pop,
            action = action_fn,
            horizon = horizon,
            num_episodes = num_episodes,
            seed = seed + gen,
        )

        pc_loss = float(pop.baldwin.last_loss) if (predictive_coding and hasattr(pop, 'baldwin') and exists(pop.baldwin.last_loss)) else 0.0
        restore_lora_(pop, champ_lora, 0)

        candidates = sorted(set(torch.topk(fits, k = min(8, pop_size)).indices.tolist()) | set(prev_elites) | {0})
        eval_cands = []
        for idx in candidates:
            s = export_champion_state_dict(pop, idx)
            mod = load_model(s)
            cm = evaluate_gait(eval_env, mod, action_fn, seeds = eval_seeds, horizon = horizon)
            fits[idx] = cm['ret']
            eval_cands.append((cm['ret'], idx, s, mod, cm))

        eval_cands.sort(key = lambda x: x[0], reverse = True)
        prev_elites = [idx for _, idx, _, _, _ in eval_cands[:4]]
        _, best_idx, cand_state, cand_model, m = eval_cands[0]

        improved = (m['min_steps'] >= horizon) and (m['ret'] > best_ret)
        if improved:
            best_ret = m['ret']
            champ_state = cand_state
            champ_model = cand_model
            champ_metrics = m
            champ_lora = snapshot_lora(pop, best_idx)
            Path(output_checkpoint).parent.mkdir(parents = True, exist_ok = True)
            torch.save(champ_state, output_checkpoint)

        pbar.write(
            f"gen {gen:03d} | ret={m['ret']:7.1f} (best={best_ret:7.1f}) | "
            f"steps={m['steps']:5.1f} (min={m['min_steps']:3.0f}) | "
            f"x={m['x']:+.3f}m (min={m['min_x']:+.3f}m) | |y|={m['abs_y']:.3f}m | sway={m['sway']:.3f}m | "
            f"roll={m['roll']:.1f}deg | pc_loss={pc_loss:.4f}"
            + (' *' if improved else '')
        )

        if gen < num_generations - 1:
            pop.evolve_(
                fits,
                survive_frac = 0.80,
                elite_frac = 0.15,
                num_elites = 4,
                selection_type = 'deterministic',
                parent_selection_type = 'tournament',
                crossover_type = 'extrapolative',
                eta_min = 0.0,
                eta_max = 1.2,
                mutation_type = 'full_gaussian',
                epsilon = 0.008,
                weight_decay = 1e-4,
                consolidate_alpha = 0.0,
                swarm_type = 'firefly' if use_pso else None,
                swarm_bottom_frac = 1.0,
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
                    scale_invariant_distance = True,
                ),
            )
            restore_lora_(pop, champ_lora, 0)

    Path(output_checkpoint).parent.mkdir(parents = True, exist_ok = True)
    torch.save(champ_state, output_checkpoint)

    if render:
        render_gait(eval_env, champ_model, action_fn, video_path = video_path, horizon = horizon, seed = 0)
        print(f'saved video to {video_path}')

    interactor.close()
    eval_env.close()
    return champ_metrics

if __name__ == '__main__':
    fire.Fire(train)
