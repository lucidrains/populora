from __future__ import annotations

import inspect
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import atleast_1d, is_tensor, nn

from env_ssl_wrapper import AutoBatchedWrapper, DoneTrackerWrapper, StandardizeWrapper
from env_ssl_wrapper.helpers import env_autoresets, first_existing, get_attr
from env_ssl_wrapper.utils import parse_wrapper
from torch_einops_utils import temp_eval

from populora._utils import cast_tensor, default, exists, rescale_from_range_to_range, torch_save
from populora.distributed import (
    broadcast_object,
    distributed_device,
    evaluate_population_distributed,
    is_main_rank,
    preserve_rng,
)
from populora.memory import Memory, init_memory_tensor
from populora.policies import make_categorical_action
from populora.population import Population, _generation_loop
from populora.spaces import action_space_bounds, action_space_is_discrete
from populora.vector import MultiprocessingVecEnv, action_dim_of

# helpers

def _safe_close(env):
    if not exists(env):
        return

    try:
        env.close()
    except (AttributeError, TypeError):
        pass

def _flatten_batch(t):
    if is_tensor(t):
        return t.reshape(-1)
    return np.asarray(t).reshape(-1)

def _env_autoresets(env) -> bool:
    # whether a vector env resets terminated slots on its own. the rollout
    # engine relies on autoresetting vector envs for its per-slot tour
    # bookkeeping - only non-autoresetting envs may be force-reset mid-run.
    # detection is the canonical one from env-ssl-wrapper (autoreset_mode /
    # autoresets markers, or the gymnasium 1.x VectorEnv base class).

    return env_autoresets(env)

def _gather_slot_rewards(reward_arr, locs, device):
    # per-slot rewards for a step, tensorized - `None` entries (some envs emit
    # them on terminal transitions) count as zero reward

    r = reward_arr[locs]

    if is_tensor(r):
        return r.to(device).float()

    r = np.asarray(r)

    if r.dtype == object:
        r = np.array([0. if v is None else v for v in r], dtype = np.float32)

    return torch.from_numpy(r.astype(np.float32)).to(device)

# seeding - deterministic per (evaluation, slot, episode), so episodes are
# reproducible across runs and distinct across slots / individuals

MASK64 = (1 << 64) - 1

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return (z ^ (z >> 31)) & MASK64

def _mix_seed(*parts):
    x = 0

    for part in parts:
        x = (x ^ (int(part) + 0x9E3779B97F4A7C15 + (x << 6) + (x >> 2))) & MASK64

    return _splitmix64(x) & 0xFFFFFFFF

# composition - any env, idempotently wrapped with the canonical stack.
# wrappers already present in the env chain are never applied twice, so a
# researcher's own compose_env output is accepted as-is

def _has_wrapper(env, cls):
    curr = env

    while curr is not None:
        if isinstance(curr, cls):
            return True
        curr = getattr(curr, 'env', None)

    return False

def _compose_env(env, wrappers):
    if not _has_wrapper(env, StandardizeWrapper):
        env = StandardizeWrapper(env)

    parsed = [parse_wrapper(spec) for spec in wrappers]

    # auto_batch gives single envs their leading batch dim, which the done
    # tracker (and the rollout) rely on - for vector envs it is a passthrough

    if not _has_wrapper(env, AutoBatchedWrapper) and not any(cls is AutoBatchedWrapper for _, cls in parsed):
        parsed.insert(0, parse_wrapper('auto_batch'))

    for func, cls in parsed:
        if _has_wrapper(env, cls):
            continue

        env = func(env)

    if not _has_wrapper(env, DoneTrackerWrapper):
        # autoresetting vector envs self-reset terminated slots and skip tracker;
        # single or non-autoresetting envs keep it for needs_reset tracking
        if get_attr(env, 'num_envs', 1) <= 1 or not env_autoresets(env):
            env = DoneTrackerWrapper(env)

    return env

def _try_seed(env, seed):
    # some envs have no seedable rng - evaluate them with their own stream

    try:
        env.seed(seed)
    except ValueError:
        pass

def _fitness_mode(fn):
    # infer a custom fitness function's mode: (pop, individuals/indices) -> batch;
    # (pop, idx/individual) -> per-index; (pop) -> all-at-once. a **kwargs
    # signature falls through to all-at-once

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return 'all'

    required = {
        name for name, param in params.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and param.default is inspect.Parameter.empty
    }

    if 'individuals' in required or 'indices' in required:
        return 'batch'

    if required.intersection(('idx', 'individual', 'individual_idx')):
        return 'per_index'

    return 'all'

class EnvInteractor:
    def __init__(
        self,
        envs,
        *,
        num_envs: int | None = None,
        device: torch.device | str | None = None,
        wrappers: Sequence | None = None,
        # policy outputs live on from_range (-1, 1) by default - an action fn
        # from make_action carries its own range, e.g. a beta with
        # beta_rescale_neg_one_one = False emits on (0, 1) instead

        seed: int = 0,
        to_range = None,
        from_range = (-1., 1.)
    ):
        self.seed = seed
        self.device = torch.device(default(device, distributed_device()))
        self.to_range = to_range
        self.from_range = from_range

        envs = self._maybe_vectorize(envs, num_envs)
        envs = self._normalize_envs(envs)

        if isinstance(wrappers, str):
            wrappers = (wrappers,)

        wrappers = default(wrappers, ('auto_batch', 'flatten_obs', ('tensor', dict(device = self.device))))

        self.envs = [
            _compose_env(env, wrappers)
            for env in envs
        ]

        self.num_envs = sum(env.num_envs for env in self.envs)

        self.obs_dim, self.action_dim, self.action_space = self._probe()

        # policies emit on (-1, 1) - map them onto a Box action space for
        # free; discrete and unclassifiable spaces are left untouched
        self.to_range = default(to_range, action_space_bounds(self.action_space))

    def __repr__(self):
        return f'{self.__class__.__name__}(num_envs = {self.num_envs}, device = {self.device})'

    def close(self):
        for env in self.envs:
            _safe_close(env)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # a factory / class plus num_envs becomes a parallel vector env; an env
    # instance cannot be cloned, and lists / vector envs pass through as-is

    def _maybe_vectorize(self, envs, num_envs):
        if not exists(num_envs) or num_envs <= 1 or isinstance(envs, (list, dict)):
            return envs

        if not isinstance(envs, (type, str)) and callable(get_attr(envs, 'step')):
            if get_attr(envs, 'num_envs', 1) > 1:
                return envs

            spec_id = getattr(get_attr(envs, 'spec'), 'id', None)
            if exists(spec_id):
                spec_kwargs = getattr(get_attr(envs, 'spec'), 'kwargs', {}) or {}
                return MultiprocessingVecEnv(lambda: __import__('gymnasium').make(spec_id, **spec_kwargs), num_envs, seed = self.seed)

            raise ValueError('num_envs > 1 needs a factory or class - a single env instance cannot be cloned')

        return MultiprocessingVecEnv(envs, num_envs, seed = self.seed)

    # probe the first env once for observation / action dimensions

    def _probe(self):
        env = self.envs[0]
        obs, _ = env.reset()

        if isinstance(obs, dict):
            obs_dim = None
        else:
            shape = obs.shape if is_tensor(obs) else np.asarray(obs).shape
            obs_dim = int(shape[1]) if len(shape) > 1 else int(shape[0])

        action_space = first_existing(env, 'single_action_space', 'action_space')
        return obs_dim, action_dim_of(env), action_space

    def _rescale_actions(self, actions, action = None):
        if not exists(self.to_range):
            return actions

        # an action fn from make_action carries its own output range, which
        # beats the interactor-level default - e.g. a beta with
        # beta_rescale_neg_one_one = False emits on (0, 1), not (-1, 1)

        from_range = default(get_attr(action, 'from_range'), self.from_range)
        return rescale_from_range_to_range(actions, from_range = from_range, to_range = self.to_range)

    # environment normalization - a single env, a list / dict of envs, a
    # vectorized env, or a factory returning any of the above

    @staticmethod
    def _normalize_envs(envs):
        if envs is None:
            raise ValueError('interact_with_env requires an environment')

        if isinstance(envs, str):
            import gymnasium as gym
            return EnvInteractor._normalize_envs(gym.make(envs))

        if isinstance(envs, type):
            envs = envs()
            return EnvInteractor._normalize_envs(envs)

        if hasattr(envs, 'step'):
            return [envs]

        if callable(envs):
            envs = envs()
            return EnvInteractor._normalize_envs(envs)

        if isinstance(envs, dict):
            envs = envs.values()

        envs = list(envs)

        # any env factories nested in the list / dict are called too

        envs = [
            env() if callable(env) and not hasattr(env, 'step') else env
            for env in envs
        ]

        # and any lists of envs returned by those factories are flattened

        flattened = []

        for env in envs:
            if isinstance(env, (list, tuple)):
                flattened.extend(env)
            else:
                flattened.append(env)

        envs = flattened

        assert len(envs) > 0, 'interact_with_env requires at least one environment'

        return envs

    # population construction - model = None builds a default MLP from the
    # probed dims (logits for discrete action spaces, tanh for continuous)

    def _default_backbone(self) -> nn.Module:
        if not exists(self.obs_dim) or not exists(self.action_dim):
            raise ValueError('cannot auto-build a policy - pass a model explicitly')

        layers = [nn.Linear(self.obs_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, self.action_dim)]

        if not action_space_is_discrete(self.action_space):
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def population(
        self,
        model: nn.Module | None = None,
        *,
        pop_size: int,
        low_rank: int,
        lora_targets: Sequence[str] | None = None,
        seed: int | None = None,
        eval_seed: int | None = 0,
        **kwargs
    ):
        return Population(
            default(model, self._default_backbone()),
            pop_size = pop_size,
            low_rank = low_rank,
            lora_targets = lora_targets,
            device = self.device,
            seed = seed,
            eval_seed = eval_seed,
            **kwargs
        )

    # rollout engine - every slot plays its assigned individuals, one routed
    # forward per timestep; fitness = mean episode return over slots. slots tile
    # the individuals cyclically (slots >= pop) or play their share in turn

    @torch.no_grad()
    def _rollout_fitness(
        self,
        population,
        individuals,
        *,
        action = None,
        horizon = 1000,
        num_episodes = 1,
        seed = None,
        agg = 'mean'
    ):
        envs = self.envs
        device = population.device
        individuals = [int(i) for i in individuals]
        num_individuals = len(individuals)

        assert horizon >= 1, 'horizon must be at least 1'
        assert num_episodes >= 1, 'num_episodes must be at least 1'

        if num_individuals == 0:
            return torch.zeros(0, device = device)

        base_seed = default(seed, default(population.eval_seed, self.seed))

        # seed and reset every environment once per evaluation call

        last_obs = []

        for j, env in enumerate(envs):
            _try_seed(env, _mix_seed(base_seed, j, 0))
            obs, _ = env.reset()
            last_obs.append(obs)

        num_slots = sum(env.num_envs for env in envs)
        assert num_slots > 0, 'interact_with_env requires at least one environment'

        # memory threading - each slot carries the last emitted memory, fed back
        # next step, initialized to the researcher's init and reset on episode end

        memory_wrapped = isinstance(population.model, Memory)
        memories = None

        if memory_wrapped:
            init_memories = init_memory_tensor(population.model.init_memory, num_slots, device = device)
            memories = init_memories.clone()

        slot_to_env = []
        env_slot_offsets = []

        for e, env in enumerate(envs):
            env_slot_offsets.append(len(slot_to_env))
            slot_to_env.extend([e] * env.num_envs)

        # slot tours - which individuals each slot plays, in order

        if num_slots >= num_individuals:
            tours = [[i % num_individuals] for i in range(num_slots)]
        else:
            tours = [list(range(i, num_individuals, num_slots)) for i in range(num_slots)]

        # route ids - kept current as tours advance, so the per-step route
        # tensor is a cached index instead of a rebuild

        route_ids = torch.tensor([tour[0] for tour in tours], device = device)

        single_env = len(envs) == 1

        budgets = [horizon * num_episodes * len(tour) for tour in tours]
        cursor = [0] * num_slots
        episodes_done = [0] * num_slots
        steps_used = [0] * num_slots
        episode_counter = [0] * num_slots

        returns = torch.zeros(num_individuals, device = device)
        current_return = torch.zeros(num_individuals, device = device)
        episode_counts = torch.zeros(num_individuals, dtype = torch.long, device = device)
        env_reset_count = [0] * len(envs)

        with temp_eval(population):
            while True:
                active_slots = [
                    k for k in range(num_slots)
                    if cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]
                ]

                if not active_slots:
                    break

                all_live = len(active_slots) == num_slots

                # one routed forward over the batch of current observations; when
                # every slot of a single env is live, the obs and routes are already
                # batched, so the per-step assembly is skipped entirely

                if single_env and all_live:
                    obs_batch = last_obs[0]
                    obs_batch = obs_batch.to(device) if is_tensor(obs_batch) else torch.as_tensor(obs_batch).to(device)
                    individuals_batch = route_ids
                else:
                    positions = {k: i for i, k in enumerate(active_slots)}

                    active_obs = []

                    for k in active_slots:
                        e = slot_to_env[k]
                        local = k - env_slot_offsets[e]
                        active_obs.append(last_obs[e][local])

                    obs_batch = torch.stack([
                        obs if is_tensor(obs) else torch.as_tensor(obs)
                        for obs in active_obs
                    ]).to(device)
                    individuals_batch = route_ids[active_slots]

                if memory_wrapped:
                    active_mem = memories[active_slots]
                    outputs = population(active_mem, obs_batch, individuals = individuals_batch, eval_and_no_grad = True)
                    policy_out, mem_next = outputs

                    mem_next = cast_tensor(mem_next, device = device)
                    mem_next = atleast_1d(mem_next)

                    assert mem_next.shape == memories[active_slots].shape, f'network emitted memory {tuple(mem_next.shape)} does not match the carried memory {tuple(memories[active_slots].shape)}'
                    memories[active_slots] = mem_next
                else:
                    outputs = population(obs_batch, individuals = individuals_batch, eval_and_no_grad = True)
                    policy_out = outputs

                actions = action(policy_out) if exists(action) else policy_out
                actions = actions if is_tensor(actions) else torch.as_tensor(actions)
                actions = atleast_1d(actions)
                actions = self._rescale_actions(actions, action)

                # step every environment with active slots; inactive sub-envs of a
                # vector env still have to step along, so they get zero actions

                for e, env in enumerate(envs):
                    # when every slot of the single env is live, the actions are
                    # the whole env batch - no zero-pad scatter

                    if single_env and all_live:
                        e_slots = active_slots
                        e_actions = actions
                        num_env_slots = env.num_envs
                    else:
                        e_slots = [k for k in active_slots if slot_to_env[k] == e]

                        if not e_slots:
                            continue

                        num_env_slots = env.num_envs
                        e_actions = torch.zeros(
                            (num_env_slots, *actions.shape[1:]),
                            dtype = actions.dtype,
                            device = actions.device
                        )

                        for k in e_slots:
                            e_actions[k - env_slot_offsets[e]] = actions[positions[k]]

                    obs, reward, terminated, truncated, info = env.step(e_actions)
                    last_obs[e] = obs

                    reward_arr = _flatten_batch(reward)
                    done_arr = _flatten_batch(terminated) | _flatten_batch(truncated)

                    # envs with no reward on terminal / reset transitions count zero
                    # reward; rewards accumulate in one tensor op per env

                    locs = [k - env_slot_offsets[e] for k in e_slots]
                    current_return.index_add_(0, route_ids[e_slots], _gather_slot_rewards(reward_arr, locs, device))

                    for k in e_slots:
                        steps_used[k] += 1

                    if done_arr.any():
                        for k in e_slots:
                            if not bool(done_arr[k - env_slot_offsets[e]]):
                                continue

                            idx = route_ids[k]

                            returns[idx] = torch.maximum(returns[idx], current_return[idx]) if agg == 'max' else returns[idx] + current_return[idx]
                            current_return[idx] = 0.
                            episode_counts[idx] += 1
                            episodes_done[k] += 1
                            episode_counter[k] += 1

                            if memory_wrapped:
                                memories[k] = init_memories[k]

                            if episodes_done[k] >= num_episodes:
                                cursor[k] += 1
                                episodes_done[k] = 0

                                if cursor[k] < len(tours[k]):
                                    route_ids[k] = tours[k][cursor[k]]

                            # single envs advance to the next episode with a fresh seed;
                            # vector envs autoreset on their own

                            if num_env_slots == 1 and cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]:
                                _try_seed(env, _mix_seed(base_seed, k, episode_counter[k]))
                                last_obs[e], _ = env.reset()

                    # non-autoresetting vector envs reset on full completion;
                    # autoresetting envs advance slots independently and are left untouched
                    if num_env_slots > 1 and not _env_autoresets(env) and env.needs_reset and any(
                        cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]
                        for k in e_slots
                    ):
                        for k in e_slots:
                            if cursor[k] < len(tours[k]):
                                current_return[route_ids[k]] = 0.

                                if memory_wrapped:
                                    memories[k] = init_memories[k]

                        _try_seed(env, _mix_seed(base_seed, e, env_reset_count[e]))
                        env_reset_count[e] += 1
                        last_obs[e], _ = env.reset()

        # mean return over completed episodes, falling back to the partial
        # return of an in-progress episode when the budget ran out

        completed = episode_counts > 0
        fitness = torch.zeros(num_individuals, device = device)

        # 'mean' averages episodes; 'max' keeps the best episode (optimistic)
        if agg == 'max':
            fitness[completed] = returns[completed]
            fitness[~completed] = torch.maximum(returns[~completed], current_return[~completed])
        else:
            fitness[completed] = returns[completed] / episode_counts[completed]
            fitness[~completed] = current_return[~completed]

        return fitness

    # built-in rollout evaluator, distributed-friendly: (pop, individuals)
    # evaluates exactly those - what evaluate_population_distributed partitions

    def fitness(
        self,
        population,
        *,
        action = None,
        horizon = 1000,
        num_episodes = 1,
        seed = None,
        agg = 'mean'
    ):
        def fitness_fn(population, individuals = None):
            individuals = default(individuals, range(population.pop_size))
            return self._rollout_fitness(
                population,
                individuals,
                action = action,
                horizon = horizon,
                num_episodes = num_episodes,
                seed = seed,
                agg = agg
            )

        fitness_fn.batch_eval = True
        return fitness_fn

    # evaluation - distributed across ranks whenever torchrun is used, and
    # through every slot of every environment at full batch parallelism

    def evaluate(
        self,
        population,
        *,
        action = None,
        fitness = None,
        horizon = 1000,
        num_episodes = 1,
        seed = None,
        agg = 'mean',
        contiguous = False,
        sync_base_model = False
    ):
        # discrete action spaces sample categorically when no action fn is given
        action = default(action, self._auto_action())

        if not exists(fitness):
            fitness = self.fitness(population, action = action, horizon = horizon, num_episodes = num_episodes, seed = seed, agg = agg)

        mode = 'batch' if getattr(fitness, 'batch_eval', False) else _fitness_mode(fitness)

        if mode != 'all':
            fitnesses = evaluate_population_distributed(
                population,
                fitness,
                batch_eval = mode == 'batch',
                contiguous = contiguous,
                sync_base_model = sync_base_model
            )
        else:
            # all-at-once fitness - evaluated on the main rank and broadcast,
            # so every rank derives the same fitnesses and evolves in lockstep

            with preserve_rng():
                res = fitness(population) if is_main_rank() else None

                if exists(res):
                    dd = dict(device = population.device, dtype = torch.float32)
                    res = cast_tensor(res, **dd)
                    assert res.shape[0] == population.pop_size, f'all-at-once fitness must return one fitness per individual, got {tuple(res.shape)}'

            fitnesses = broadcast_object(res, src = 0)

        # a NaN fitness would poison selection - rank it dead last instead
        if not torch.isfinite(fitnesses).all():
            fitnesses = torch.nan_to_num(fitnesses, nan = torch.finfo(torch.float32).min)

        return fitnesses

    def _auto_action(self):
        if action_space_is_discrete(self.action_space):
            return make_categorical_action()

        return None

    # high-level evolve - a bare model (population built around it) or an
    # existing population in, merged best policy out; target_fitness (+ patience)
    # stops early, return_history gives per-generation best / mean

    def evolve(
        self,
        model: nn.Module | None = None,
        *,
        pop_size: int | None = None,
        low_rank: int | None = None,
        lora_targets: Sequence[str] | None = None,
        action = None,
        fitness = None,
        num_generations: int = 25,
        horizon: int = 1000,
        num_episodes: int = 1,
        seed: int = 0,
        eval_seed: int = 0,
        progress: bool = False,
        target_fitness: float | None = None,
        patience: int = 1,
        return_history: bool = False,
        evolve_kwargs: dict | None = None,
        evaluate_kwargs: dict | None = None,
        on_generation = None,
        adaptive_epsilon: bool = False,
        target_success_rate: float = 0.20,
        epsilon_factor: float = 1.15,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        resume: bool = False,
        exact_resume: bool = False
    ):
        if isinstance(model, Population):
            population = model
        else:
            population = self.population(
                model,
                pop_size = pop_size,
                low_rank = low_rank,
                lora_targets = lora_targets,
                seed = seed,
                eval_seed = eval_seed
            )

        evolve_kwargs = default(evolve_kwargs, dict())
        evaluate_kwargs = default(evaluate_kwargs, dict())

        checkpoint_dir = Path(checkpoint_dir) if exists(checkpoint_dir) else None

        if exists(checkpoint_dir):
            checkpoint_every = default(checkpoint_every, 1)

        start_generation = 0
        initial_state = None

        # resume from the latest checkpoint, so a killed run picks up where it
        # left off - with exact_resume, the rng state is restored too, making
        # the resumed run identical to an uninterrupted one

        if exists(checkpoint_dir) and resume:
            resumed = self._resume_checkpoint(population, checkpoint_dir, exact_resume)

            if exists(resumed):
                start_generation, best_fitness, best_index, history = resumed
                initial_state = (best_fitness, best_index, history)

        def evaluate_gen():
            return self.evaluate(
                population,
                action = action,
                fitness = fitness,
                horizon = horizon,
                num_episodes = num_episodes,
                **evaluate_kwargs
            )

        def _checkpoint(generation, best_fitness, best_index, history, is_best):
            if exists(checkpoint_dir) and ((generation + 1) % checkpoint_every == 0 or generation == num_generations - 1):
                self._save_checkpoint(
                    population,
                    checkpoint_dir,
                    generation = generation,
                    best_fitness = best_fitness,
                    best_index = best_index,
                    history = history,
                    exact_resume = exact_resume,
                    as_best = is_best
                )

            if exists(on_generation):
                on_generation(generation, best_fitness, best_index, history, is_best)

        _, best_index, history = _generation_loop(
            population,
            evaluate_gen,
            num_generations = num_generations,
            target_fitness = target_fitness,
            patience = patience,
            progress = progress,
            start_generation = start_generation,
            initial_state = initial_state,
            on_generation = _checkpoint,
            adaptive_epsilon = adaptive_epsilon,
            target_success_rate = target_success_rate,
            epsilon_factor = epsilon_factor,
            **evolve_kwargs
        )

        policy = population.merge_(best_index)

        if return_history:
            return policy, history

        return policy

    def _resume_checkpoint(self, population, checkpoint_dir, exact_resume):
        # the loop state to pick up from - (generation, best_fitness, best_index,
        # history) - or None when there is nothing to resume

        path = Path(checkpoint_dir) / 'latest.pt'

        if not path.exists():
            return None

        pkg = torch.load(path, map_location = population.device, weights_only = False)
        population.load(pkg['population'])

        if 'eval_seed' in pkg:
            population._eval_seed = pkg['eval_seed']

        if exact_resume and 'rng_state' in pkg:
            torch.random.set_rng_state(pkg['rng_state'].to('cpu'))

        return pkg['generation'] + 1, pkg['best_fitness'], pkg['best_index'], pkg['history']

    def _save_checkpoint(
        self,
        population,
        checkpoint_dir,
        *,
        generation,
        best_fitness,
        best_index,
        history,
        exact_resume,
        as_best
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents = True, exist_ok = True)

        pkg = dict(
            population = population.state_dict_pkg(save_base_model = False),
            generation = generation,
            best_fitness = best_fitness,
            best_index = best_index,
            history = history,
            eval_seed = population.eval_seed
        )

        if exact_resume:
            pkg['rng_state'] = torch.random.get_rng_state().clone()

        torch_save(pkg, checkpoint_dir / 'latest.pt')

        if as_best:
            torch_save(pkg, checkpoint_dir / 'best.pt')

        return self

    # evaluate a final policy (merged model / LoRA / plain module) - mean
    # episode return over every environment, used for post-evolution reporting

    @torch.no_grad()
    def evaluate_policy(
        self,
        policy: nn.Module,
        *,
        action = None,
        num_episodes: int = 5,
        horizon: int = 1000,
        seed: int = 0
    ):
        device = self.device
        returns = []

        policy = policy.to(device)
        memory_wrapped = isinstance(policy, Memory)

        # the episode seeding consumes the env rng, so evaluation is kept off
        # the global rng - a side-effect-free policy eval never perturbs an
        # ongoing evolution

        with temp_eval(policy), preserve_rng():
            for j, env in enumerate(self.envs):
                num_env_slots = env.num_envs

                for episode in range(num_episodes):
                    _try_seed(env, _mix_seed(seed, j, episode))
                    obs, _ = env.reset()

                    if memory_wrapped:
                        init_memories = init_memory_tensor(policy.init_memory, num_env_slots, device = device)
                        mem = init_memories.clone()

                    episode_returns = np.zeros(num_env_slots)
                    completed = np.zeros(num_env_slots, dtype = bool)
                    steps = 0

                    while steps < horizon and not completed.all():
                        obs_t = obs.to(device) if is_tensor(obs) else torch.as_tensor(obs).to(device)

                        if memory_wrapped:
                            policy_out, mem = policy(mem, obs_t)

                            mem = cast_tensor(mem, device = device)
                            mem = atleast_1d(mem)

                            assert mem.shape == init_memories.shape, f'network emitted memory {tuple(mem.shape)} does not match the carried memory {tuple(init_memories.shape)}'
                        else:
                            policy_out = policy(obs_t)

                        actions = action(policy_out) if exists(action) else policy_out
                        actions = actions if is_tensor(actions) else torch.as_tensor(actions)
                        actions = atleast_1d(actions)
                        actions = self._rescale_actions(actions, action)

                        obs, reward, terminated, truncated, info = env.step(actions)

                        reward = _flatten_batch(reward)
                        done = _flatten_batch(terminated) | _flatten_batch(truncated)

                        for k in range(num_env_slots):
                            if not completed[k]:
                                reward_k = reward[k]
                                episode_returns[k] += float(reward_k) if reward_k is not None else 0.
                                completed[k] = bool(done[k])

                                if completed[k] and memory_wrapped:
                                    mem[k] = init_memories[k]

                        steps += 1

                    returns.append(episode_returns.mean())

        return float(np.mean(returns))

# entry point - wrap any environment (or environments) from any simulator into
# a populora evolution loop

def interact_with_env(
    envs,
    *,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    wrappers: Sequence | None = None,
    seed: int = 0,
    to_range = None,
    from_range = (-1., 1.)
):
    return EnvInteractor(
        envs,
        num_envs = num_envs,
        device = device,
        wrappers = wrappers,
        seed = seed,
        to_range = to_range,
        from_range = from_range
    )

# convenience - evolve a model against an environment (or environments) in one
# call, wrapping interact_with_env and returning the merged best policy

def evolve_with_env(
    envs,
    model: nn.Module | None = None,
    *,
    num_envs: int | None = None,
    pop_size: int = 16,
    low_rank: int = 16,
    lora_targets: Sequence[str] | None = None,
    action = None,
    fitness = None,
    num_generations: int = 25,
    horizon: int = 1000,
    num_episodes: int = 1,
    seed: int = 0,
    eval_seed: int = 0,
    device: torch.device | str | None = None,
    wrappers: Sequence | None = None,
    progress: bool = False,
    target_fitness: float | None = None,
    patience: int = 1,
    return_history: bool = False,
    evolve_kwargs: dict | None = None,
    evaluate_kwargs: dict | None = None,
    on_generation = None,
    adaptive_epsilon: bool = False,
    target_success_rate: float = 0.20,
    epsilon_factor: float = 1.15,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
    exact_resume: bool = False,
    to_range = None,
    from_range = (-1., 1.)
):
    interactor = interact_with_env(
        envs,
        num_envs = num_envs,
        device = device,
        wrappers = wrappers,
        seed = seed,
        to_range = to_range,
        from_range = from_range
    )

    population = interactor.population(
        model,
        pop_size = pop_size,
        low_rank = low_rank,
        lora_targets = lora_targets,
        seed = seed,
        eval_seed = eval_seed
    )

    return interactor.evolve(
        population,
        action = action,
        fitness = fitness,
        num_generations = num_generations,
        horizon = horizon,
        num_episodes = num_episodes,
        progress = progress,
        target_fitness = target_fitness,
        patience = patience,
        return_history = return_history,
        evolve_kwargs = evolve_kwargs,
        evaluate_kwargs = evaluate_kwargs,
        on_generation = on_generation,
        adaptive_epsilon = adaptive_epsilon,
        target_success_rate = target_success_rate,
        epsilon_factor = epsilon_factor,
        checkpoint_dir = checkpoint_dir,
        checkpoint_every = checkpoint_every,
        resume = resume,
        exact_resume = exact_resume
    )
