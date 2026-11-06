# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "fire",
#   "numpy",
#   "matplotlib",
#   "torch-einops-utils",
#   "x-mlps-pytorch>=0.6.2",
#   "gymnasium[box2d]",
#   "populora",
# ]
# [tool.uv.sources]
# populora = { path = "." }
# ///

# baldwinian selection with live self-predictive (SPR) test-time training (TTT)
# on windy / turbulent LunarLanderContinuous-v3. compares baseline evolution
# (no TTT) against baldwinian evolution with per-episode reset
# (carry_across_episodes = False) and cross-episode persistence
# (carry_across_episodes = True).

from __future__ import annotations

import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import torch
from x_mlps_pytorch import WeightOnlyMLP

from populora import Baldwin, Population, interact_with_env, make_action
from populora._utils import default, exists

# per-episode evaluation helper - measures episode 0 vs episode 1 returns and
# counterfactual test-time training gain (TTT ON vs TTT OFF) on held-out seeds

@torch.no_grad()
def eval_per_episode_returns(
    interactor,
    population: Population,
    *,
    action_fn,
    baldwin_cfg: dict | None = None,
    horizon: int = 450,
    seed: int = 9000,
):
    env = interactor.envs[0]
    device = population.device
    pop_size = population.pop_size
    assert env.num_envs == pop_size

    baldwin_learner = interactor._resolve_baldwin(population, baldwin_cfg) if exists(baldwin_cfg) else None
    individuals = torch.arange(pop_size, device = device, dtype = torch.long)

    ep_returns = []

    if exists(baldwin_learner):
        baldwin_learner.start_rollout(individuals)

    try:
        for ep in range(2):
            obs, _ = env.reset(seed = seed + ep * 100)
            done = np.zeros(pop_size, dtype = bool)
            ret = np.zeros(pop_size, dtype = np.float64)

            for _ in range(horizon):
                if done.all():
                    break

                obs_t = obs.to(device) if torch.is_tensor(obs) else torch.as_tensor(obs, device = device)
                policy_out = population(obs_t, individuals = individuals, eval_and_no_grad = True)
                raw_actions = action_fn(policy_out)
                actions = interactor._rescale_actions(raw_actions, action_fn)

                next_obs, reward, terminated, truncated, _ = env.step(actions)

                rew_np = reward.cpu().numpy() if torch.is_tensor(reward) else np.asarray(reward)
                term_np = terminated.cpu().numpy() if torch.is_tensor(terminated) else np.asarray(terminated)
                trunc_np = truncated.cpu().numpy() if torch.is_tensor(truncated) else np.asarray(truncated)
                step_done = term_np | trunc_np

                alive = ~done
                ret[alive] += rew_np[alive]
                done = done | step_done

                if exists(baldwin_learner):
                    dones_t = torch.as_tensor(done, device = device, dtype = torch.bool)
                    baldwin_learner.step(
                        obs_t,
                        raw_actions.detach(),
                        next_obs,
                        individuals = individuals,
                        dones = dones_t,
                    )

                obs = next_obs

            ep_returns.append(ret)

            if exists(baldwin_learner) and ep == 0:
                baldwin_learner.reset_slots(
                    list(range(pop_size)),
                    list(range(pop_size)),
                    episode_boundary_only = True,
                )
    finally:
        if exists(baldwin_learner):
            baldwin_learner.end_rollout()

    return ep_returns[0], ep_returns[1]

# plotting helper

def save_comparison_plot(results: dict, modes: dict, num_generations: int, plot_path: str | Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed - skipping plot generation')
        return

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize = (14.5, 10.2), dpi = 160)
    gens = np.arange(1, num_generations + 1)

    palette = {
        'baseline': '#64748b',
        'baldwin_reset': '#0ea5e9',
        'baldwin_carry': '#10b981',
    }

    # Panel A: Peak fitness across generations
    ax = axes[0, 0]
    for k, cfg in modes.items():
        arr = np.asarray(results[k]['best_curves'])
        mean = arr.mean(axis = 0)
        sem = arr.std(axis = 0) / np.sqrt(len(arr))
        color = palette.get(k, '#3b82f6')
        ax.plot(gens, mean, lw = 2.4, color = color, label = cfg['label'], marker = 'o', markersize = 4)
        ax.fill_between(gens, mean - sem, mean + sem, color = color, alpha = 0.18)
    ax.set_title('A. Peak Fitness vs. Generation (Turbulent LunarLander-v3)', fontsize = 12, fontweight = 'bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Return (2-Episode Mean)')
    ax.legend(loc = 'lower right', frameon = True, fontsize = 9.5)
    ax.grid(True, alpha = 0.3)

    # Panel B: Top-quartile elite fitness across generations
    ax = axes[0, 1]
    for k, cfg in modes.items():
        arr = np.asarray(results[k]['topq_curves'])
        mean = arr.mean(axis = 0)
        sem = arr.std(axis = 0) / np.sqrt(len(arr))
        color = palette.get(k, '#3b82f6')
        ax.plot(gens, mean, lw = 2.4, color = color, label = cfg['label'], marker = 's', markersize = 4)
        ax.fill_between(gens, mean - sem, mean + sem, color = color, alpha = 0.18)
    ax.set_title('B. Top-Quartile Elite Fitness vs. Generation', fontsize = 12, fontweight = 'bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Top-25% Elite Mean Return')
    ax.legend(loc = 'lower right', frameon = True, fontsize = 9.5)
    ax.grid(True, alpha = 0.3)

    # Panel C: Online SPR cosine prediction loss across generations
    ax = axes[1, 0]
    for k, cfg in modes.items():
        if not results[k]['spr_loss_by_gen']:
            continue
        arr = np.asarray(results[k]['spr_loss_by_gen'])
        mean = arr.mean(axis = 0)
        sem = arr.std(axis = 0) / np.sqrt(len(arr))
        color = palette.get(k, '#3b82f6')
        ax.plot(gens, mean, lw = 2.4, color = color, label = cfg['label'], marker = '^', markersize = 4.5)
        ax.fill_between(gens, mean - sem, mean + sem, color = color, alpha = 0.18)
    ax.set_title('C. Live Next-Latent Prediction Loss (SPR Cosine Loss)', fontsize = 12, fontweight = 'bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Online SPR Cosine Loss (Lower = More Predictable)')
    ax.legend(loc = 'upper right', frameon = True, fontsize = 9.5)
    ax.grid(True, alpha = 0.3)

    # Panel D: Cross-episode adaptation (Ep 0 -> Ep 1) & net TTT gain
    ax = axes[1, 1]
    mode_keys = list(modes.keys())
    x = np.arange(len(mode_keys))
    width = 0.26

    ep0_means = [float(np.mean(results[k]['ep0_returns'])) for k in mode_keys]
    ep1_means = [float(np.mean(results[k]['ep1_returns'])) for k in mode_keys]
    ttt_gains = [
        float(np.mean(np.asarray(results[k]['ttt_ablation_on']) - np.asarray(results[k]['ttt_ablation_off'])))
        for k in mode_keys
    ]

    bars0 = ax.bar(x - width, ep0_means, width, label = 'Episode 0 Return', color = '#94a3b8', edgecolor = 'black', linewidth = 0.6)
    bars1 = ax.bar(x, ep1_means, width, label = 'Episode 1 Return (After Ep 0)', color = '#38bdf8', edgecolor = 'black', linewidth = 0.6)
    bars_g = ax.bar(x + width, ttt_gains, width, label = 'Net TTT Gain (TTT ON - TTT OFF)', color = '#10b981', edgecolor = 'black', linewidth = 0.6)

    for bars in (bars0, bars1, bars_g):
        for b in bars:
            h = b.get_height()
            va = 'bottom' if h >= 0 else 'top'
            offset = 2.0 if h >= 0 else -8.0
            ax.annotate(
                f'{h:+.1f}' if bars is bars_g else f'{h:.1f}',
                xy = (b.get_x() + b.get_width() / 2, h),
                xytext = (0, offset),
                textcoords = 'offset points',
                ha = 'center',
                va = va,
                fontsize = 8.5,
                fontweight = 'bold',
            )

    ax.axhline(0, color = 'black', lw = 0.8, alpha = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([modes[k]['short_label'] for k in mode_keys], fontsize = 10)
    ax.set_title('D. Cross-Episode Adaptation & Net Test-Time Training Gain', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Return / Delta Return')
    ax.legend(loc = 'best', frameon = True, fontsize = 9.5)
    ax.grid(True, alpha = 0.3, axis = 'y')

    fig.suptitle(
        'PopuLoRA: Baldwinian Selection with Live SPR Next-Latent Test-Time Training (TTT)',
        fontsize = 14,
        fontweight = 'bold',
        y = 0.99,
    )
    fig.tight_layout(rect = [0, 0, 1, 0.965])
    out_path = Path(plot_path)
    fig.savefig(out_path, dpi = 160)
    plt.close(fig)
    print(f'\nsaved comparison plot to {out_path}')

# main training & benchmark entrypoint

def train_baldwinian(
    pop_size: int = 48,
    low_rank: int = 4,
    num_generations: int = 18,
    horizon: int = 450,
    num_episodes: int = 2,
    distribution: str = 'beta',
    sample_actions: bool = False,
    enable_wind: bool = True,
    wind_power: float = 10.0,
    turbulence_power: float = 1.0,
    carry_across_episodes: bool | None = None,
    lamarckian: bool = False,
    assimilate_tau: float = 0.0,
    lora_lr: float = 1.2e-3,
    predictor_lr: float = 4e-3,
    lora_warmup_steps: int = 16,
    update_every: int = 4,
    k_steps: int = 1,
    trust_region: float = 0.20,
    anchor_decay: float = 0.03,
    target_ema_decay: float = 1.0,
    survive_frac: float = 0.5,
    elite_frac: float = 0.25,
    num_elites: int = 2,
    epsilon: float = 0.15,
    weight_decay: float = 1e-3,
    seed: int | None = None,
    seeds: tuple[int, ...] = (42, 43, 44),
    plot_path: str | None = 'baldwin_ttt_comparison.png',
    return_results: bool = False,
):
    run_seeds = (int(seed),) if exists(seed) else tuple(int(s) for s in seeds)
    action_fn = make_action(distribution, sample = sample_actions)
    topq_count = max(1, pop_size // 4)

    def _baldwin_cfg(carry: bool) -> dict:
        return dict(
            carry_across_episodes = carry,
            lamarckian = lamarckian,
            assimilate_tau = assimilate_tau,
            lora_lr = lora_lr,
            predictor_lr = predictor_lr,
            lora_warmup_steps = lora_warmup_steps,
            update_every = update_every,
            k_steps = k_steps,
            trust_region = trust_region,
            anchor_decay = anchor_decay,
            target_ema_decay = target_ema_decay,
        )

    modes = dict(
        baseline = dict(
            label = 'Baseline Evolution (No TTT)',
            short_label = 'Baseline\n(No TTT)',
            baldwin = None,
        ),
    )

    if carry_across_episodes in (None, False):
        modes['baldwin_reset'] = dict(
            label = 'Baldwin + SPR TTT (carry_across_episodes=False)',
            short_label = 'Baldwin TTT\n(carry=False)',
            baldwin = _baldwin_cfg(False),
        )

    if carry_across_episodes in (None, True):
        modes['baldwin_carry'] = dict(
            label = 'Baldwin + SPR TTT (carry_across_episodes=True)',
            short_label = 'Baldwin TTT\n(carry=True)',
            baldwin = _baldwin_cfg(True),
        )

    results = {
        k: dict(
            best_curves = [],
            topq_curves = [],
            mean_curves = [],
            spr_loss_by_gen = [],
            ep0_returns = [],
            ep1_returns = [],
            ttt_ablation_on = [],
            ttt_ablation_off = [],
        )
        for k in modes
    }

    print(
        f'[PopuLoRA Baldwinian TTT] pop_size: {pop_size} | low_rank: {low_rank} | '
        f'generations: {num_generations} | episodes: {num_episodes} | horizon: {horizon}'
    )
    print(
        f'  env: LunarLanderContinuous-v3 (enable_wind={enable_wind}, wind_power={wind_power}, '
        f'turbulence_power={turbulence_power}) | seeds: {run_seeds}'
    )
    print(
        f'  ttt: lora_lr={lora_lr}, predictor_lr={predictor_lr}, warmup={lora_warmup_steps}, '
        f'update_every={update_every}, k_steps={k_steps}, trust_region={trust_region}, anchor_decay={anchor_decay}\n'
    )

    t0 = time.time()

    for s in run_seeds:
        for mode_key, cfg in modes.items():
            torch.manual_seed(s)
            np.random.seed(s)

            vec_env = gym.make_vec(
                'LunarLanderContinuous-v3',
                num_envs = pop_size,
                enable_wind = enable_wind,
                wind_power = wind_power,
                turbulence_power = turbulence_power,
            )
            interactor = interact_with_env(vec_env, seed = s)

            try:
                mlp = WeightOnlyMLP(8, 64, 64, 4)

                if exists(cfg['baldwin']):
                    model = Baldwin(
                        mlp,
                        latent_layer = mlp.layers[1].linear,
                        lora_memory = mlp.layers[1].linear,
                        **cfg['baldwin'],
                    )
                else:
                    model = mlp

                pop = Population(
                    model,
                    pop_size = pop_size,
                    low_rank = low_rank,
                    lora_targets = ['layers.*.linear'],
                    eval_seed = s,
                    adaptive_epsilon = True,
                    epsilon_init = epsilon,
                    sigma_granularity = 'weight',
                )
                # break initial symmetry at gen 0 so each individual starts with a non-zero LoRA subspace
                pop.mutate_('full_gaussian', all_individuals = True, epsilon = epsilon)

                best_curve, topq_curve, mean_curve, spr_gen_curve = [], [], [], []

                for gen in range(num_generations):
                    prev_loss_idx = len(pop.baldwin.loss_history) if hasattr(pop, 'baldwin') else 0

                    fits = interactor.evaluate(
                        pop,
                        action = action_fn,
                        horizon = horizon,
                        num_episodes = num_episodes,
                        seed = s,
                    )

                    fits_np = fits.detach().cpu().numpy()
                    best_curve.append(float(fits_np.max()))
                    topq_curve.append(float(np.sort(fits_np)[-topq_count:].mean()))
                    mean_curve.append(float(fits_np.mean()))

                    if hasattr(pop, 'baldwin') and len(pop.baldwin.loss_history) > prev_loss_idx:
                        gen_losses = pop.baldwin.loss_history[prev_loss_idx:]
                        spr_gen_curve.append(float(np.mean(gen_losses)))

                    if gen < num_generations - 1:
                        pop.evolve_(
                            fits,
                            survive_frac = survive_frac,
                            elite_frac = elite_frac,
                            num_elites = num_elites,
                            crossover_type = 'extrapolative',
                            mutation_type = 'full_gaussian',
                            epsilon = epsilon,
                            weight_decay = weight_decay,
                        )

                eval_baldwin_cfg = default(cfg['baldwin'], _baldwin_cfg(True))
                heldout_seed = 9000 + s * 10

                ep0_off, ep1_off = eval_per_episode_returns(
                    interactor,
                    pop,
                    action_fn = action_fn,
                    baldwin_cfg = None,
                    horizon = horizon,
                    seed = heldout_seed,
                )
                ep0_on, ep1_on = eval_per_episode_returns(
                    interactor,
                    pop,
                    action_fn = action_fn,
                    baldwin_cfg = eval_baldwin_cfg,
                    horizon = horizon,
                    seed = heldout_seed,
                )

                elite_idx = np.argsort(fits_np)[-topq_count:]
                results[mode_key]['best_curves'].append(best_curve)
                results[mode_key]['topq_curves'].append(topq_curve)
                results[mode_key]['mean_curves'].append(mean_curve)
                if spr_gen_curve:
                    results[mode_key]['spr_loss_by_gen'].append(spr_gen_curve)

                used_ep0 = ep0_on[elite_idx] if exists(cfg['baldwin']) else ep0_off[elite_idx]
                used_ep1 = ep1_on[elite_idx] if exists(cfg['baldwin']) else ep1_off[elite_idx]
                results[mode_key]['ep0_returns'].append(float(used_ep0.mean()))
                results[mode_key]['ep1_returns'].append(float(used_ep1.mean()))

                results[mode_key]['ttt_ablation_off'].append(float(0.5 * (ep0_off[elite_idx].mean() + ep1_off[elite_idx].mean())))
                results[mode_key]['ttt_ablation_on'].append(float(0.5 * (ep0_on[elite_idx].mean() + ep1_on[elite_idx].mean())))

                ttt_gain = results[mode_key]['ttt_ablation_on'][-1] - results[mode_key]['ttt_ablation_off'][-1]
                print(
                    f'[seed={s} | {mode_key:13s}] '
                    f'final_best={best_curve[-1]:7.2f} | '
                    f'final_topQ={topq_curve[-1]:7.2f} | '
                    f'ep0={results[mode_key]["ep0_returns"][-1]:7.2f} -> ep1={results[mode_key]["ep1_returns"][-1]:7.2f} | '
                    f'TTT_gain={ttt_gain:+6.2f}',
                    flush = True,
                )
            finally:
                interactor.close()

    elapsed = time.time() - t0
    print(f'\ncompleted in {elapsed:.1f}s — summary across {len(run_seeds)} seed(s):')
    for mode_key, cfg in modes.items():
        best_m = float(np.mean([c[-1] for c in results[mode_key]['best_curves']]))
        topq_m = float(np.mean([c[-1] for c in results[mode_key]['topq_curves']]))
        ep0_m = float(np.mean(results[mode_key]['ep0_returns']))
        ep1_m = float(np.mean(results[mode_key]['ep1_returns']))
        gain_m = float(np.mean(np.asarray(results[mode_key]['ttt_ablation_on']) - np.asarray(results[mode_key]['ttt_ablation_off'])))
        print(
            f'  {mode_key:13s} | best: {best_m:7.2f} | topQ: {topq_m:7.2f} | '
            f'heldout ep0->ep1: {ep0_m:7.2f} -> {ep1_m:7.2f} | net TTT gain: {gain_m:+6.2f}'
        )

    if exists(plot_path) and str(plot_path).strip():
        save_comparison_plot(results, modes, num_generations, plot_path)

    if return_results:
        return results

if __name__ == '__main__':
    fire.Fire(train_baldwinian)
