from __future__ import annotations

import inspect
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import atleast_1d, is_tensor, nn

from env_ssl_wrapper import AutoBatchedWrapper, DoneTrackerWrapper, StandardizeWrapper
from env_ssl_wrapper.utils import parse_wrapper
from torch_einops_utils import temp_eval

from populora._utils import cast_tensor, default, exists
from populora.distributed import (
    broadcast_object,
    distributed_device,
    evaluate_population_distributed,
    is_main_rank,
    preserve_rng,
)
from populora.memory import Memory, init_memory_tensor
from populora.population import Population

# helpers

def linear_layer_paths(model: nn.Module) -> list[str]:
    """module paths of every Linear layer, used as `lora_targets` for the population"""
    return [
        path for path, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

def _flatten_batch(t):
    if is_tensor(t):
        return t.reshape(-1)
    return np.asarray(t).reshape(-1)

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
        env = DoneTrackerWrapper(env)

    return env

def _try_seed(env, seed):
    # some envs have no seedable rng - evaluate them with their own stream

    try:
        env.seed(seed)
    except ValueError:
        pass

def _fitness_mode(fn):
    # infer how a researcher's custom fitness function evaluates a population:
    #   (population, individuals = ...) or (population, indices = ...)  -> batch mode, distributed per rank
    #   (population, idx) or (population, individual)                  -> per-index mode, distributed per rank
    #   (population)                                                   -> all-at-once, main rank evaluates and broadcasts
    #
    # a **kwargs signature cannot receive the individuals positionally, so it
    # falls through to all-at-once, same as anything uninspectable

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

def _maybe_progress(iterable, enabled = False, desc = ''):
    if not enabled:
        return iterable

    try:
        from tqdm import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, desc = desc)

# main class

class EnvInteractor:
    def __init__(
        self,
        envs,
        *,
        device: torch.device | str | None = None,
        wrappers: Sequence | None = None,
        seed: int = 0
    ):
        self.seed = seed
        self.device = torch.device(default(device, distributed_device()))

        envs = self._normalize_envs(envs)

        if isinstance(wrappers, str):
            wrappers = (wrappers,)

        wrappers = default(wrappers, ('auto_batch', 'flatten_obs', ('tensor', dict(device = self.device))))

        self.envs = [
            _compose_env(env, wrappers)
            for env in envs
        ]

        self.num_envs = sum(env.num_envs for env in self.envs)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_envs = {self.num_envs}, device = {self.device})'

    # environment normalization - a single env, a list / dict of envs, a
    # vectorized env, or a factory returning any of the above

    @staticmethod
    def _normalize_envs(envs):
        if envs is None:
            raise ValueError('interact_with_env requires an environment')

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

    # population construction

    def population(
        self,
        model: nn.Module,
        *,
        pop_size: int,
        low_rank: int,
        lora_targets: Sequence[str] | None = None,
        seed: int | None = None,
        eval_seed: int | None = 0,
        **kwargs
    ):
        lora_targets = default(lora_targets, linear_layer_paths(model))

        assert len(lora_targets) > 0, 'model has no Linear layers to target - pass explicit lora_targets'

        return Population(
            model,
            pop_size = pop_size,
            low_rank = low_rank,
            lora_targets = lora_targets,
            device = self.device,
            seed = seed,
            eval_seed = eval_seed,
            **kwargs
        )

    # rollout engine - the heart of the interactor. every slot (sub-env) plays
    # the individuals assigned to it, one model forward per timestep over the
    # whole batch with per-sample routing, and each individual's fitness is its
    # mean episode return over all slots and episodes. with as many slots as
    # individuals, each slot plays exactly one individual - the classic setup;
    # otherwise the slots tile the individuals cyclically (num_slots >= pop) or
    # each slot plays its share in turn (num_slots < pop)

    @torch.no_grad()
    def _rollout_fitness(
        self,
        population,
        individuals,
        *,
        action = None,
        horizon = 1000,
        num_episodes = 1,
        seed = None
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

        # memory threading - when the policy is wrapped in Memory, each slot
        # carries the memory emitted by the previous timestep, fed back in on
        # the next, initialized to the researcher's init and reset to it
        # wherever episodes reset

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

        budgets = [horizon * num_episodes * len(tour) for tour in tours]
        cursor = [0] * num_slots
        episodes_done = [0] * num_slots
        steps_used = [0] * num_slots
        episode_counter = [0] * num_slots

        returns = torch.zeros(num_individuals, device = device)
        current_return = torch.zeros(num_individuals, device = device)
        episode_counts = torch.zeros(num_individuals, dtype = torch.long, device = device)
        env_reset_count = [0] * len(envs)

        while True:
            active_slots = [
                k for k in range(num_slots)
                if cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]
            ]

            if not active_slots:
                break

            positions = {k: i for i, k in enumerate(active_slots)}

            # one routed forward over the batch of current observations

            active_obs = []
            active_individuals = []

            for k in active_slots:
                e = slot_to_env[k]
                local = k - env_slot_offsets[e]
                active_obs.append(last_obs[e][local])
                active_individuals.append(tours[k][cursor[k]])

            obs_batch = torch.stack([
                obs if is_tensor(obs) else torch.as_tensor(obs)
                for obs in active_obs
            ]).to(device)
            individuals_batch = torch.tensor(active_individuals, device = device)

            if memory_wrapped:
                active_mem = memories[active_slots]
                outputs = population(active_mem, obs_batch, individuals = individuals_batch, eval_and_no_grad = True)
                policy_out, mem_next = outputs

                mem_next = cast_tensor(mem_next, device)
                mem_next = atleast_1d(mem_next)

                assert mem_next.shape == memories[active_slots].shape, f'network emitted memory {tuple(mem_next.shape)} does not match the carried memory {tuple(memories[active_slots].shape)}'
                memories[active_slots] = mem_next
            else:
                outputs = population(obs_batch, individuals = individuals_batch, eval_and_no_grad = True)
                policy_out = outputs

            actions = action(policy_out) if exists(action) else policy_out
            actions = actions if is_tensor(actions) else torch.as_tensor(actions)
            actions = atleast_1d(actions)

            # step every environment with active slots; inactive sub-envs of a
            # vector env still have to step along, so they get zero actions

            for e, env in enumerate(envs):
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

                # some envs return no reward on terminal / reset transitions
                # (e.g. dm_control) - treat those steps as zero reward. rewards
                # accumulate in one tensor op per env; the per-slot state machine
                # below only runs on episode endings

                locs = [k - env_slot_offsets[e] for k in e_slots]
                idxs = [tours[k][cursor[k]] for k in e_slots]

                idxs_t = torch.tensor(idxs, device = device)
                current_return.index_add_(0, idxs_t, _gather_slot_rewards(reward_arr, locs, device))

                for k in e_slots:
                    steps_used[k] += 1

                for k in e_slots:
                    if not bool(done_arr[k - env_slot_offsets[e]]):
                        continue

                    idx = tours[k][cursor[k]]

                    returns[idx] += current_return[idx]
                    current_return[idx] = 0.
                    episode_counts[idx] += 1
                    episodes_done[k] += 1
                    episode_counter[k] += 1

                    if memory_wrapped:
                        memories[k] = init_memories[k]

                    if episodes_done[k] >= num_episodes:
                        cursor[k] += 1
                        episodes_done[k] = 0

                    # single envs advance to the next episode with a fresh seed;
                    # vector envs autoreset on their own

                    if num_env_slots == 1 and cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]:
                        _try_seed(env, _mix_seed(base_seed, k, episode_counter[k]))
                        last_obs[e], _ = env.reset()

                # a vector env whose sub-envs have all had an episode end, while
                # individuals remain to be played, is reset onto a fresh seeded
                # episode. any in-progress (autoreset) episodes of the next
                # individuals are discarded - they never completed

                if num_env_slots > 1 and env.needs_reset and any(
                    cursor[k] < len(tours[k]) and steps_used[k] < budgets[k]
                    for k in e_slots
                ):
                    for k in e_slots:
                        if cursor[k] < len(tours[k]):
                            current_return[tours[k][cursor[k]]] = 0.

                            if memory_wrapped:
                                memories[k] = init_memories[k]

                    _try_seed(env, _mix_seed(base_seed, e, env_reset_count[e]))
                    env_reset_count[e] += 1
                    last_obs[e], _ = env.reset()

        # mean return over completed episodes, falling back to the partial
        # return of an in-progress episode when the budget ran out

        completed = episode_counts > 0
        fitness = torch.zeros(num_individuals, device = device)
        fitness[completed] = returns[completed] / episode_counts[completed]
        fitness[~completed] = current_return[~completed]

        return fitness

    # fitness function - the built-in rollout evaluator, distributed-friendly:
    # `fitness(population, individuals)` evaluates exactly those individuals,
    # which is what `evaluate_population_distributed` needs to partition the
    # population across ranks

    def fitness(
        self,
        population,
        *,
        action = None,
        horizon = 1000,
        num_episodes = 1,
        seed = None
    ):
        def fitness_fn(population, individuals = None):
            individuals = default(individuals, range(population.pop_size))
            return self._rollout_fitness(
                population,
                individuals,
                action = action,
                horizon = horizon,
                num_episodes = num_episodes,
                seed = seed
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
        contiguous = False,
        sync_base_model = False
    ):
        if not exists(fitness):
            fitness = self.fitness(population, action = action, horizon = horizon, num_episodes = num_episodes, seed = seed)

        mode = 'batch' if getattr(fitness, 'batch_eval', False) else _fitness_mode(fitness)

        if mode != 'all':
            return evaluate_population_distributed(
                population,
                fitness,
                batch_eval = mode == 'batch',
                contiguous = contiguous,
                sync_base_model = sync_base_model
            )

        # all-at-once fitness - evaluated on the main rank and broadcast, so
        # every rank derives the same fitnesses and evolves in lockstep

        with preserve_rng():
            res = fitness(population) if is_main_rank() else None

            if exists(res):
                res = cast_tensor(res, population.device).to(dtype = torch.float32)
                assert res.shape[0] == population.pop_size, f'all-at-once fitness must return one fitness per individual, got {tuple(res.shape)}'

        return broadcast_object(res, src = 0)

    # high level evolution loop - accepts a bare model (a population is built
    # around it) or an existing population, and returns the merged best policy.
    # pass `target_fitness` to stop early once it is reached, and
    # `return_history = True` to also get the per-generation best / mean

    def evolve(
        self,
        model: nn.Module,
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
        return_history: bool = False,
        evolve_kwargs: dict | None = None,
        evaluate_kwargs: dict | None = None,
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

        best_fitness = float('-inf')
        best_index = 0
        history = []
        start_generation = 0

        # resume from the latest checkpoint, so a killed run picks up where it
        # left off - with exact_resume, the rng state is restored too, making
        # the resumed run identical to an uninterrupted one

        if exists(checkpoint_dir) and resume:
            resumed = self._resume_checkpoint(population, checkpoint_dir, exact_resume)

            if exists(resumed):
                start_generation, best_fitness, best_index, history = resumed

        for generation in _maybe_progress(range(start_generation, num_generations), progress, desc = 'evolving'):
            fitnesses = self.evaluate(
                population,
                action = action,
                fitness = fitness,
                horizon = horizon,
                num_episodes = num_episodes,
                **evaluate_kwargs
            )

            gen_best_fitness = float(fitnesses.max())
            is_best = gen_best_fitness > best_fitness

            if is_best:
                best_fitness = gen_best_fitness
                best_index = int(fitnesses.argmax())

            history.append(dict(
                best_fitness = gen_best_fitness,
                mean_fitness = float(fitnesses.mean()),
            ))

            if exists(target_fitness) and gen_best_fitness >= target_fitness:
                break

            population.evolve_(fitnesses, **evolve_kwargs)

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

        if exact_resume:
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

        torch.save(pkg, checkpoint_dir / 'latest.pt')

        if as_best:
            torch.save(pkg, checkpoint_dir / 'best.pt')

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

        with temp_eval(policy):
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

                            mem = cast_tensor(mem, device)
                            mem = atleast_1d(mem)

                            assert mem.shape == init_memories.shape, f'network emitted memory {tuple(mem.shape)} does not match the carried memory {tuple(init_memories.shape)}'
                        else:
                            policy_out = policy(obs_t)

                        actions = action(policy_out) if exists(action) else policy_out
                        actions = actions if is_tensor(actions) else torch.as_tensor(actions)
                        actions = atleast_1d(actions)

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
    device: torch.device | str | None = None,
    wrappers: Sequence | None = None,
    seed: int = 0
):
    return EnvInteractor(
        envs,
        device = device,
        wrappers = wrappers,
        seed = seed
    )

# convenience - evolve a model against an environment (or environments) in one
# call, wrapping interact_with_env and returning the merged best policy

def evolve_with_env(
    envs,
    model: nn.Module,
    *,
    pop_size: int,
    low_rank: int,
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
    return_history: bool = False,
    evolve_kwargs: dict | None = None,
    evaluate_kwargs: dict | None = None
):
    interactor = interact_with_env(
        envs,
        device = device,
        wrappers = wrappers,
        seed = seed
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
        return_history = return_history,
        evolve_kwargs = evolve_kwargs,
        evaluate_kwargs = evaluate_kwargs
    )
