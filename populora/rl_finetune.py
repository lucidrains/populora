from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from populora._utils import default, exists, is_tensor


def rollout(
    pop,
    env: Any,
    individual: int,
    *,
    seed: int,
    horizon: int,
    noise: float = 0.,
    action_clip: tuple[float, float] = (-1., 1.),
    rng: np.random.RandomState | None = None,
):
    obs, _ = env.reset(seed = seed)
    states, actions = [], []
    total_reward = 0.

    for _ in range(horizon):
        obs_t = torch.as_tensor(obs, dtype = torch.float32, device = pop.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            action = pop(obs_t, individual = individual).squeeze(0).cpu().numpy()

        if noise > 0.:
            rng = default(rng, np.random)
            action = action + rng.normal(0., noise, size = action.shape).astype(np.float32)

        if exists(action_clip):
            action = np.clip(action, *action_clip)

        states.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    return total_reward, states, actions


def eval_seeds(pop, env, individual, seeds, horizon, action_clip = (-1., 1.)):
    total = sum(rollout(pop, env, individual, seed = s, horizon = horizon, action_clip = action_clip)[0] for s in seeds)
    return total / max(len(seeds), 1)


def rl_finetune_elites_(
    pop,
    fitnesses: Tensor | np.ndarray,
    env: Any,
    *,
    num_elites: int = 2,
    rollouts: int = 4,
    noise: float = 0.08,
    lr: float = 0.05,
    horizon: int = 400,
    seeds: Sequence[int] | None = None,
    max_weight: float | None = 8.0,
    action_clip: tuple[float, float] = (-1.0, 1.0),
    gen: int = 0,
) -> tuple[Tensor, int, float]:
    """Monotonic Policy Improvement for elites via Reward-Weighted Regression / REINFORCE."""
    if is_tensor(fitnesses):
        fitnesses_t = fitnesses
        fits_np = fitnesses.detach().cpu().numpy()
    else:
        fits_np = np.asarray(fitnesses, dtype = np.float32)
        fitnesses_t = torch.as_tensor(fitnesses, device = pop.device)

    seeds = default(seeds, [gen * 1000 + r for r in range(2)])
    elite_idxs = np.argsort(fits_np)[::-1][:num_elites]

    n_updated = 0
    total_gain = 0.0

    for idx in elite_idxs:
        idx = int(idx)
        baseline_fit = eval_seeds(pop, env, idx, seeds, horizon, action_clip)
        backup = pop.backup_individual(idx)
        best_trajs = []

        for rep_i, seed in enumerate(seeds):
            base_ret, _, _ = rollout(pop, env, idx, seed = seed, horizon = horizon, action_clip = action_clip)
            best_rep_adv = 0.0
            best_rep_traj = None

            for r in range(rollouts):
                seed_noise = (int(gen) * 100003 + idx * 7919 + rep_i * 104729 + r * 1301) % (2 ** 31 - 1)
                rng = np.random.RandomState(seed_noise)

                ep_ret, states, actions = rollout(
                    pop,
                    env,
                    idx,
                    seed = seed,
                    horizon = horizon,
                    noise = noise,
                    action_clip = action_clip,
                    rng = rng,
                )

                adv = ep_ret - base_ret
                if adv > best_rep_adv:
                    best_rep_adv = adv
                    best_rep_traj = (states, actions)

            if exists(best_rep_traj) and best_rep_adv > 0.:
                best_trajs.append(best_rep_traj)

        if len(best_trajs) == 0:
            continue

        all_states = np.concatenate([np.stack(t[0]) for t in best_trajs], axis = 0)
        all_actions = np.concatenate([np.stack(t[1]) for t in best_trajs], axis = 0)

        S = torch.as_tensor(all_states, dtype = torch.float32, device = pop.device)
        A = torch.as_tensor(all_actions, dtype = torch.float32, device = pop.device)

        step_lr = lr / max(len(all_states), 1)

        for p in pop.model.parameters():
            p.requires_grad = False
        for p in pop.weight_down.parameters():
            p.requires_grad = True
        for p in pop.weight_up.parameters():
            p.requires_grad = True

        pop.zero_grad(set_to_none = True)
        pred = pop(S, individual = idx)
        loss = 0.5 * F.mse_loss(pred, A, reduction = 'sum')
        loss.backward()

        with torch.no_grad():
            for k in pop.weight_down:
                if exists(pop.weight_down[k].grad):
                    pop.weight_down[k].data[idx].sub_(step_lr * pop.weight_down[k].grad[idx])
                if exists(pop.weight_up[k].grad):
                    pop.weight_up[k].data[idx].sub_(step_lr * pop.weight_up[k].grad[idx])
                if exists(max_weight):
                    pop.weight_down[k].data[idx].clamp_(-max_weight, max_weight)
                    pop.weight_up[k].data[idx].clamp_(-max_weight, max_weight)

            pop.zero_grad(set_to_none = True)

        # Monotonic acceptance check - strictly better or revert
        new_fit = eval_seeds(pop, env, idx, seeds, horizon, action_clip)

        if new_fit > baseline_fit:
            fitnesses_t[idx] = new_fit
            n_updated += 1
            total_gain += (new_fit - baseline_fit)
        else:
            pop.restore_individual(idx, backup)

    mean_gain = total_gain / max(n_updated, 1)
    return fitnesses_t, n_updated, mean_gain


rl_finetune_elites = rl_finetune_elites_
