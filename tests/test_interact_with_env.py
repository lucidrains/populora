from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.distributed as dist
from env_ssl_wrapper import DoneTrackerWrapper, compose_env
from env_ssl_wrapper.mocks import (
    AutoresetVectorMockEnv,
    GymnasiumMockEnv,
    IsaacMockEnv,
    ManiSkillMockEnv,
)
from torch import allclose, nn

from populora import (
    EnvInteractor,
    evolve_with_env,
    interact_with_env,
    linear_layer_paths,
)
from populora.distributed import (
    distributed_world_size,
    is_distributed,
)

# an environment whose reward discriminates policies - the optimal action is a
# fixed constant, so evolution has to find it to maximize episode return

class ActionAimMockEnv:
    obs_dim = 4
    action_dim = 1
    max_steps = 60
    num_envs = 1
    target = 0.5
    is_vector = False

    def __init__(self, seed = 0):
        self.rng = np.random.default_rng(seed)
        self.unwrapped = self

    def seed(self, seed = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.t = 0
        return self.obs(), {}

    def step(self, action):
        action = float(np.asarray(action).reshape(-1)[0])
        reward = 1.0 - abs(action - self.target)
        self.t += 1
        return self.obs(), reward, self.t >= self.max_steps, False, {}

    def obs(self):
        return self.rng.standard_normal(self.obs_dim)

class StaggeredVectorMockEnv:
    # vector env whose sub-envs end episodes at different times - exercises the
    # whole-env reset path when slots have individuals left to play

    obs_dim = 4
    action_dim = 2
    num_envs = 3
    is_vector = True
    max_steps_per_env = np.array([10, 20, 30])

    def __init__(self, seed = 0):
        self.rng = np.random.default_rng(seed)
        self.unwrapped = self
        self.t = np.zeros(self.num_envs, dtype = int)
        self.state = np.zeros((self.num_envs, self.obs_dim))

    def seed(self, seed = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.t = np.zeros(self.num_envs, dtype = int)
        self.state = np.zeros((self.num_envs, self.obs_dim))
        return self.obs(), {}

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (self.num_envs, self.action_dim)

        self.state[..., :self.action_dim] += action
        self.t += 1

        dones = self.t >= self.max_steps_per_env

        for ind in np.where(dones)[0]:
            self.t[ind] = 0
            self.state[ind] = 0

        return self.obs(), np.ones(self.num_envs), dones, np.zeros(self.num_envs, dtype = bool), {}

    def obs(self):
        return self.state + self.rng.standard_normal((self.num_envs, self.obs_dim))

class NoSeedMockEnv(GymnasiumMockEnv):
    # an env whose rng cannot be seeded - the interactor must fall back to the
    # env's own stream

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.unwrapped = self

    def seed(self, seed = 0):
        raise ValueError('cannot seed this environment')

def make_policy(obs_dim = 4, out_dim = 1, zero_last = False):
    model = nn.Sequential(
        nn.Linear(obs_dim, 16),
        nn.ReLU(),
        nn.Linear(16, out_dim)
    )

    if zero_last:
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()

    return model

def test_linear_layer_paths():
    model = make_policy(out_dim = 2)
    assert linear_layer_paths(model) == ['0', '2']

def test_interact_single_env():
    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    assert isinstance(interactor, EnvInteractor)
    assert interactor.num_envs == 1
    assert len(interactor.envs) == 1

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 6, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert fitnesses.shape == (6,)
    assert torch.equal(fitnesses, torch.full((6,), 40.))  # constant reward, full episodes

def test_interact_list_of_envs():
    interactor = interact_with_env([GymnasiumMockEnv() for _ in range(4)], seed = 0)
    assert interactor.num_envs == 4

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert fitnesses.shape == (4,)
    assert torch.equal(fitnesses, torch.full((4,), 40.))

def test_interact_vector_env():
    # more individuals than slots - each slot plays its share in turn

    interactor = interact_with_env(AutoresetVectorMockEnv(), seed = 0)
    assert interactor.num_envs == 4

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 6, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 50)
    assert fitnesses.shape == (6,)
    assert torch.equal(fitnesses, torch.full((6,), 40.))

    # more slots than individuals - the individuals tile across the slots

    interactor = interact_with_env([AutoresetVectorMockEnv(), AutoresetVectorMockEnv()], seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 50)
    assert fitnesses.shape == (4,)
    assert torch.equal(fitnesses, torch.full((4,), 40.))

    # multiple episodes per individual on a vector env

    interactor = interact_with_env(AutoresetVectorMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 50, num_episodes = 2)
    assert fitnesses.shape == (4,)
    assert torch.equal(fitnesses, torch.full((4,), 40.))

    # maniskill-style env - batched observations at num_envs = 1

    interactor = interact_with_env(ManiSkillMockEnv(), seed = 0)
    assert interactor.num_envs == 1

    pop = interactor.population(make_policy(obs_dim = 16, out_dim = 8), pop_size = 3, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert fitnesses.shape == (3,)

def test_interact_already_composed():
    # a researcher's own compose_env output must be accepted as-is - no
    # double-wrapping

    composed = compose_env(
        AutoresetVectorMockEnv(),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    interactor = interact_with_env(composed, seed = 0)
    assert interactor.num_envs == 4

    chain = interactor.envs[0]
    num_done_trackers = 0

    while chain is not None:
        num_done_trackers += isinstance(chain, DoneTrackerWrapper)
        chain = getattr(chain, 'env', None)

    assert num_done_trackers == 1

def test_interact_env_factory():
    interactor = interact_with_env(lambda: GymnasiumMockEnv(), seed = 0)
    assert interactor.num_envs == 1

    # factories nested in a list or dict are called too

    interactor = interact_with_env([lambda: GymnasiumMockEnv(), GymnasiumMockEnv()], seed = 0)
    assert interactor.num_envs == 2

    interactor = interact_with_env(dict(a = lambda: GymnasiumMockEnv()), seed = 0)
    assert interactor.num_envs == 1

    # a factory returning a list of envs is flattened

    interactor = interact_with_env(lambda: [GymnasiumMockEnv(), GymnasiumMockEnv()], seed = 0)
    assert interactor.num_envs == 2

def test_interact_custom_fitness_modes():
    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 6, low_rank = 2)

    def fitness_batch(population, individuals):
        return torch.randn(len(individuals))

    def fitness_per_index(population, idx):
        return population(torch.randn(1, 4), individual = idx).abs().mean()

    def fitness_all(population):
        return torch.randn(population.pop_size)

    assert interactor.evaluate(pop, fitness = fitness_batch).shape == (6,)
    assert interactor.evaluate(pop, fitness = fitness_per_index).shape == (6,)
    assert interactor.evaluate(pop, fitness = fitness_all).shape == (6,)

@pytest.mark.parametrize('num_episodes', [1, 2])
def test_interact_evolve_improves(num_episodes):
    # distributed evaluation reassigns which seeded episodes each individual
    # plays, so the improvement trajectory only holds in a single process

    if is_distributed():
        return

    interactor = interact_with_env(ActionAimMockEnv(), seed = 0)
    pop = interactor.population(make_policy(zero_last = True), pop_size = 16, low_rank = 4, seed = 0)

    action = lambda logits: torch.tanh(logits)
    initial = interactor.evaluate(pop, action = action, horizon = 60, num_episodes = num_episodes)

    best_fitness = float('-inf')

    for _ in range(10):
        fitnesses = interactor.evaluate(pop, action = action, horizon = 60, num_episodes = num_episodes)
        best_fitness = max(best_fitness, float(fitnesses.max()))
        pop.evolve_(
            fitnesses,
            survive_frac = 0.5,
            elite_frac = 0.2,
            mutation_type = 'full_gaussian',
            epsilon = 0.2
        )

    assert best_fitness > float(initial.max()) + 1.

def test_interact_evolve_high_level():
    if is_distributed():
        return

    interactor = interact_with_env(ActionAimMockEnv(), seed = 0)

    policy = interactor.evolve(
        make_policy(zero_last = True),
        pop_size = 16,
        low_rank = 4,
        num_generations = 10,
        horizon = 60,
        action = lambda logits: torch.tanh(logits),
        seed = 0,
        evolve_kwargs = dict(
            survive_frac = 0.5,
            elite_frac = 0.2,
            mutation_type = 'full_gaussian',
            epsilon = 0.2
        )
    )

    assert isinstance(policy, nn.Module)
    assert interactor.evaluate_policy(policy, action = lambda logits: torch.tanh(logits), num_episodes = 5) > 40.

def test_interact_evolve_checkpoint_resume(tmp_path):
    # a killed run resumes from its checkpoint and, with exact_resume, finishes
    # bit-identically to an uninterrupted run - latest and best checkpoints are
    # both written

    if is_distributed():
        return

    def run(num_generations, checkpoint_dir = None, resume = False, return_history = True):
        # the base model's init consumes the process rng, so it is seeded here
        # to keep every run identical

        torch.manual_seed(0)
        interactor = interact_with_env(ActionAimMockEnv(), seed = 0)

        return interactor.evolve(
            make_policy(zero_last = True),
            pop_size = 16,
            low_rank = 4,
            num_generations = num_generations,
            horizon = 60,
            action = lambda logits: torch.tanh(logits),
            seed = 0,
            return_history = return_history,
            evolve_kwargs = dict(
                survive_frac = 0.5,
                elite_frac = 0.2,
                mutation_type = 'full_gaussian',
                epsilon = 0.2
            ),
            checkpoint_dir = checkpoint_dir,
            checkpoint_every = 1,
            resume = resume,
            exact_resume = True
        )

    checkpoint_dir = tmp_path / 'checkpoints'

    _, partial_history = run(5, checkpoint_dir = checkpoint_dir)
    assert len(partial_history) == 5
    assert (checkpoint_dir / 'latest.pt').exists()
    assert (checkpoint_dir / 'best.pt').exists()

    resumed_policy, resumed_history = run(10, checkpoint_dir = checkpoint_dir, resume = True)
    _, full_history = run(10)

    assert len(resumed_history) == 10
    assert resumed_history == full_history

    full_policy = run(10, return_history = False)
    assert allclose(resumed_policy.state_dict()['2.weight'], full_policy.state_dict()['2.weight'])

def test_evolve_with_env():
    # the one-call wrapper - env + model in, merged best policy out, with a
    # per-generation best / mean history. like the high-level evolve, the
    # improvement trajectory only holds in a single process

    if is_distributed():
        return

    policy, history = evolve_with_env(
        ActionAimMockEnv(),
        make_policy(zero_last = True),
        pop_size = 16,
        low_rank = 4,
        num_generations = 10,
        horizon = 60,
        action = lambda logits: torch.tanh(logits),
        seed = 0,
        return_history = True,
        evolve_kwargs = dict(
            survive_frac = 0.5,
            elite_frac = 0.2,
            mutation_type = 'full_gaussian',
            epsilon = 0.2
        )
    )

    assert isinstance(policy, nn.Module)
    assert len(history) == 10
    assert history[-1]['best_fitness'] > history[0]['best_fitness']
    assert 0. <= history[0]['mean_fitness'] <= history[0]['best_fitness'] <= 60.

def test_evolve_with_env_policy_only():
    # without return_history, just the merged policy comes back

    if is_distributed():
        return

    policy = evolve_with_env(
        ActionAimMockEnv(),
        make_policy(zero_last = True),
        pop_size = 8,
        low_rank = 2,
        num_generations = 2,
        horizon = 60,
        action = lambda logits: torch.tanh(logits),
        seed = 0,
    )

    assert isinstance(policy, nn.Module)

def test_evolve_with_env_target_fitness():
    # the constant action scores a full episode immediately, so the loop stops
    # after the first generation

    if is_distributed():
        return

    _, history = evolve_with_env(
        ActionAimMockEnv(),
        make_policy(),
        pop_size = 8,
        low_rank = 2,
        num_generations = 10,
        horizon = 60,
        action = lambda logits: 0.5,
        target_fitness = 60.,
        seed = 0,
        return_history = True,
    )

    assert len(history) == 1
    assert history[0]['best_fitness'] == 60.

def test_interact_deterministic():
    model = make_policy(out_dim = 2)

    def run():
        interactor = interact_with_env([GymnasiumMockEnv() for _ in range(3)], seed = 7)
        pop = interactor.population(model, pop_size = 6, low_rank = 2)
        return interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)

    assert torch.equal(run(), run())

def test_interact_no_envs():
    # intentionally raising - run single-process only

    if is_distributed():
        return

    with pytest.raises(ValueError):
        interact_with_env(None)

    with pytest.raises(AssertionError):
        interact_with_env([])

    with pytest.raises(AssertionError):
        interact_with_env(list)

def test_interact_dict_envs():
    interactor = interact_with_env(dict(a = GymnasiumMockEnv(), b = GymnasiumMockEnv()), seed = 0)
    assert interactor.num_envs == 2

def test_interact_wrappers_string():
    # a bare wrapper name instead of a list

    interactor = interact_with_env(GymnasiumMockEnv(), wrappers = 'tensor', seed = 0)
    assert interactor.num_envs == 1

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 2, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert torch.equal(fitnesses, torch.full((2,), 40.))

def test_interact_wrappers_without_auto_batch():
    # auto_batch is re-added when missing - otherwise single envs report
    # num_envs == obs_dim through the done tracker

    interactor = interact_with_env(GymnasiumMockEnv(), wrappers = ['tensor', 'done_tracker'], seed = 0)
    assert interactor.num_envs == 1

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 2, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert torch.equal(fitnesses, torch.full((2,), 40.))

def test_interact_population_requires_linear():
    if is_distributed():
        return

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)

    with pytest.raises(AssertionError):
        interactor.population(nn.Sequential(nn.ReLU()), pop_size = 2, low_rank = 2)

def test_interact_invalid_horizon():
    if is_distributed():
        return

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 2, low_rank = 2)

    with pytest.raises(AssertionError):
        interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 0)

    with pytest.raises(AssertionError):
        interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), num_episodes = 0)

def test_interact_no_seed_env():
    interactor = interact_with_env(NoSeedMockEnv(), seed = 0)
    assert interactor.num_envs == 1

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert torch.equal(fitnesses, torch.full((4,), 40.))

    assert interactor.evaluate_policy(pop.merge_(0), action = lambda logits: torch.tanh(logits), num_episodes = 2) == 40.

def test_interact_partial_credit():
    # 40-step episodes with a budget that only covers the first two individuals -
    # the second is cut short and credited its partial return, the third never
    # plays. exact per-individual budgets are rank-dependent, so single-process only

    if is_distributed():
        return

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 3, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 20)
    assert fitnesses.tolist() == [40., 20., 0.]

def test_interact_mixed_env_list():
    # a vector env and a single env in the same list - different slot widths

    interactor = interact_with_env([AutoresetVectorMockEnv(), GymnasiumMockEnv()], seed = 0)
    assert interactor.num_envs == 5

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 5, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 50)
    assert fitnesses.shape == (5,)
    assert torch.equal(fitnesses, torch.full((5,), 40.))

def test_interact_staggered_vector():
    # sub-envs end episodes at different times - the whole-env reset path kicks
    # in while the slots of the first two sub-envs still have individuals to
    # play, discarding their cut-off autoreset episodes. the exact slot tours
    # are rank-dependent, so single-process only

    if is_distributed():
        return

    interactor = interact_with_env(StaggeredVectorMockEnv(), seed = 0)
    assert interactor.num_envs == 3

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 5, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 50)

    assert fitnesses.tolist() == [10., 20., 30., 10., 20.]

    # evaluate_policy tracks each sub-env's episode independently

    policy = pop.merge_(0)
    score = interactor.evaluate_policy(policy, action = lambda logits: torch.tanh(logits), num_episodes = 2)
    assert score == 20.

def test_interact_single_individual():
    # pop_size = 1 tiles the individual across every slot

    interactor = interact_with_env([GymnasiumMockEnv(), GymnasiumMockEnv()], seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 1, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert fitnesses.shape == (1,)
    assert torch.equal(fitnesses, torch.tensor([40.]))

def test_interact_empty_individuals():
    # the built-in fitness handles an empty assignment (e.g. a rank with no share)

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)

    fitness_fn = interactor.fitness(pop, action = lambda logits: torch.tanh(logits), horizon = 10)

    assert fitness_fn(pop, individuals = []).shape == (0,)
    assert fitness_fn(pop).shape == (4,)
    assert fitness_fn(pop, individuals = [0, 2]).shape == (2,)

def test_interact_custom_fitness_indices_name():
    # the batch signature is also detected under the name `indices`

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)

    def fitness_indices(population, indices):
        return torch.full((len(indices),), 7.)

    fitnesses = interactor.evaluate(pop, fitness = fitness_indices)
    assert fitnesses.shape == (4,)
    assert torch.equal(fitnesses, torch.full((4,), 7.))

def test_interact_custom_fitness_kwargs():
    # **kwargs cannot receive the individuals positionally, so it is treated as
    # all-at-once rather than batch

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)

    fitnesses = interactor.evaluate(pop, fitness = lambda population, **kwargs: torch.randn(population.pop_size))
    assert fitnesses.shape == (4,)

def test_interact_custom_fitness_wrong_length():
    if is_distributed():
        return

    interactor = interact_with_env(GymnasiumMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 4, low_rank = 2)

    with pytest.raises(AssertionError):
        interactor.evaluate(pop, fitness = lambda population: torch.randn(3))

def test_interact_vector_num_episodes_more_individuals():
    # num_episodes > 1 with more individuals than slots - the whole-env reset
    # fires between individuals' episodes

    interactor = interact_with_env(AutoresetVectorMockEnv(), seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 6, low_rank = 2)

    fitnesses = interactor.evaluate(
        pop,
        action = lambda logits: torch.tanh(logits),
        horizon = 50,
        num_episodes = 2
    )

    assert fitnesses.shape == (6,)
    assert torch.equal(fitnesses, torch.full((6,), 40.))

def test_interact_dict_obs_vector_env():
    # vector env with nested dict observations - flattened into one vector per
    # sub-env, rewards are random so only shape and finiteness are checked

    interactor = interact_with_env(IsaacMockEnv(), seed = 0)
    assert interactor.num_envs == 4

    pop = interactor.population(make_policy(obs_dim = 196, out_dim = 2), pop_size = 4, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)

    assert fitnesses.shape == (4,)
    assert torch.isfinite(fitnesses).all()

def test_interact_real_gymnasium_vector():
    # a real gymnasium vector env with autoreset and pop_size > num_envs

    gym = pytest.importorskip('gymnasium')

    interactor = interact_with_env(gym.make_vec('CartPole-v1', num_envs = 4), seed = 0)
    assert interactor.num_envs == 4

    pop = interactor.population(make_policy(out_dim = 1), pop_size = 6, low_rank = 2)
    fitnesses = interactor.evaluate(
        pop,
        action = lambda logits: (logits > 0.).long().squeeze(-1),
        horizon = 100
    )

    assert fitnesses.shape == (6,)
    assert torch.isfinite(fitnesses).all()

def test_interact_scalar_action():
    # an action fn returning a plain scalar is expanded to the batch dim

    interactor = interact_with_env(ActionAimMockEnv(), seed = 0)
    pop = interactor.population(make_policy(zero_last = True), pop_size = 4, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: 0.5, horizon = 60)
    assert torch.equal(fitnesses, torch.full((4,), 60.))

def test_distributed_interact():
    # every rank evaluates its share of the population and the fitnesses are
    # gathered, so all ranks evolve in lockstep - run under torchrun

    if not is_distributed():
        return

    interactor = interact_with_env([GymnasiumMockEnv() for _ in range(4)], seed = 0)
    pop = interactor.population(make_policy(out_dim = 2), pop_size = 8, low_rank = 2)

    for _ in range(3):
        fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)

        local_total = fitnesses.sum()
        dist.all_reduce(local_total, op = dist.ReduceOp.SUM)
        assert allclose(local_total / distributed_world_size(), fitnesses.sum())

        pop.evolve_(fitnesses, survive_frac = 0.5, elite_frac = 0.2, mutation_type = 'full_gaussian', epsilon = 0.2)

    lora_stat = sum(w.pow(2).sum() for w in (*pop.weight_down.values(), *pop.weight_up.values()))
    local_total = lora_stat.clone()
    dist.all_reduce(local_total, op = dist.ReduceOp.SUM)
    assert allclose(local_total / distributed_world_size(), lora_stat)

    # pop_size smaller than the world size - one rank evaluates nothing

    pop = interactor.population(make_policy(out_dim = 2), pop_size = 1, low_rank = 2)
    fitnesses = interactor.evaluate(pop, action = lambda logits: torch.tanh(logits), horizon = 40)
    assert fitnesses.shape == (1,)

if __name__ == '__main__':
    test_linear_layer_paths()
    test_interact_single_env()
    test_interact_list_of_envs()
    test_interact_vector_env()
    test_interact_already_composed()
    test_interact_env_factory()
    test_interact_custom_fitness_modes()
    test_interact_evolve_improves()
    test_interact_evolve_high_level()
    test_evolve_with_env()
    test_evolve_with_env_policy_only()
    test_evolve_with_env_target_fitness()
    test_interact_deterministic()
    test_interact_no_envs()
    test_interact_dict_envs()
    test_interact_wrappers_string()
    test_interact_wrappers_without_auto_batch()
    test_interact_population_requires_linear()
    test_interact_invalid_horizon()
    test_interact_no_seed_env()
    test_interact_partial_credit()
    test_interact_mixed_env_list()
    test_interact_staggered_vector()
    test_interact_single_individual()
    test_interact_empty_individuals()
    test_interact_custom_fitness_indices_name()
    test_interact_custom_fitness_kwargs()
    test_interact_custom_fitness_wrong_length()
    test_interact_vector_num_episodes_more_individuals()
    test_interact_dict_obs_vector_env()
    test_interact_scalar_action()
    test_interact_real_gymnasium_vector()
    test_distributed_interact()
    print('interact tests passed')
