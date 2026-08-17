from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from populora import Memory, interact_with_env
from populora.memory import init_memory_tensor

# the counting policy emits the carried memory value as its action, and
# increments the memory every step - so an episode's return is the triangular
# number of its length, and any failure to thread / reset the memory changes
# the returns in a predictable way. the dummy projection exists only so the
# population has a Linear layer to attach lora to

class CountingPolicy(nn.Module):
    def __init__(self, obs_dim = 4):
        super().__init__()
        self.proj = nn.Linear(obs_dim, 1)

    def forward(self, mem, obs):
        return mem.float().unsqueeze(-1), mem + 1

class MemoryCountEnv:
    obs_dim = 4
    action_dim = 1
    num_envs = 1
    is_vector = False
    max_steps = 4

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
        self.t += 1
        return self.obs(), action, self.t >= self.max_steps, False, {}

    def obs(self):
        return self.rng.standard_normal(self.obs_dim)

class MemoryCountVectorEnv:
    # autoreset vector env whose sub-envs end at different times - exercises
    # per-slot memory and the whole-env reset path between individuals

    obs_dim = 4
    action_dim = 1
    num_envs = 3
    is_vector = True
    max_steps_per_env = np.array([2, 4, 6])

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

        self.state += action
        self.t += 1

        dones = self.t >= self.max_steps_per_env
        reward = action[:, 0].copy()

        for ind in np.where(dones)[0]:
            self.t[ind] = 0
            self.state[ind] = 0

        return self.obs(), reward, dones, np.zeros(self.num_envs, dtype = bool), {}

    def obs(self):
        return self.state + self.rng.standard_normal((self.num_envs, self.obs_dim))

# wrapper contract

def test_memory_wrapper_first_arg():
    # the memory is passed through as the 1st positional arg, and the tuple
    # output is returned untouched

    seen = {}

    class Net(nn.Module):
        def forward(self, mem, obs):
            seen['mem'] = mem
            seen['obs'] = obs
            return torch.zeros(obs.shape[0], 2), mem

    net = Net()
    wrapped = Memory(net)

    mem = torch.arange(4).float()
    obs = torch.randn(4, 4)
    policy, mem_next = wrapped(mem, obs)

    assert seen['mem'] is mem
    assert seen['obs'] is obs
    assert torch.equal(policy, torch.zeros(4, 2))
    assert mem_next is mem

def test_memory_wrapper_kwarg():
    # with memory_kwarg the memory is dispatched under that name, obs first

    seen = {}

    class Net(nn.Module):
        def forward(self, obs, hx = None):
            seen['obs'] = obs
            seen['hx'] = hx
            return torch.zeros(obs.shape[0], 2), hx + 1

    wrapped = Memory(Net(), memory_kwarg = 'hx')

    mem = torch.zeros(4, 8)
    obs = torch.randn(4, 4)
    _, mem_next = wrapped(mem, obs)

    assert seen['obs'] is obs
    assert seen['hx'] is mem
    assert torch.equal(mem_next, mem + 1)

def test_memory_wrapper_requires_tuple():
    class Net(nn.Module):
        def forward(self, mem, obs):
            return torch.zeros(obs.shape[0], 2)

    wrapped = Memory(Net())

    with pytest.raises(AssertionError):
        wrapped(torch.zeros(4), torch.randn(4, 4))

def test_memory_wrapper_tuple_length():
    class Net(nn.Module):
        def forward(self, mem, obs):
            return torch.zeros(obs.shape[0], 2), mem, mem

    wrapped = Memory(Net())

    with pytest.raises(AssertionError):
        wrapped(torch.zeros(4), torch.randn(4, 4))

def test_init_memory_tensor_normalization():
    # scalar ints and 0-dim tensors broadcast to every slot

    assert torch.equal(init_memory_tensor(0, 4), torch.zeros(4, dtype = torch.long))
    assert torch.equal(init_memory_tensor(torch.tensor(3.), 4), torch.full((4,), 3.))

    # a (1, ...) tensor is expanded along the batch

    assert init_memory_tensor(torch.zeros(1, 8), 4).shape == (4, 8)

    # a tensor already matching the slots is used as-is (per-slot control)

    per_slot = torch.tensor([10., 20., 30.])
    assert torch.equal(init_memory_tensor(per_slot, 3), per_slot)

    # anything else is rejected

    with pytest.raises(AssertionError):
        init_memory_tensor(torch.zeros(3, 8), 4)

# rollout threading

def test_interact_memory_carries():
    # a 4-step episode of the counting policy returns 0 + 1 + 2 + 3 = 6 - the
    # memory must be threaded step to step for that to hold

    interactor = interact_with_env(MemoryCountEnv(), seed = 0)
    pop = interactor.population(Memory(CountingPolicy()), pop_size = 1, low_rank = 2, seed = 0)
    fitnesses = interactor.evaluate(pop, action = lambda out: out, horizon = 20)

    assert torch.equal(fitnesses, torch.tensor([6.]))

def test_interact_memory_custom_init():
    # with the memory starting at 5, the episode returns 5 + 6 + 7 + 8 = 26

    interactor = interact_with_env(MemoryCountEnv(), seed = 0)
    pop = interactor.population(Memory(CountingPolicy(), init_memory = 5), pop_size = 1, low_rank = 2, seed = 0)
    fitnesses = interactor.evaluate(pop, action = lambda out: out, horizon = 20)

    assert torch.equal(fitnesses, torch.tensor([26.]))

def test_interact_memory_resets_per_episode():
    # without a reset the second episode would start at memory 4 and return
    # 4 + 5 + 6 + 7 = 22, dragging the mean up to 14

    interactor = interact_with_env(MemoryCountEnv(), seed = 0)
    pop = interactor.population(Memory(CountingPolicy()), pop_size = 1, low_rank = 2, seed = 0)
    fitnesses = interactor.evaluate(pop, action = lambda out: out, horizon = 20, num_episodes = 2)

    assert torch.equal(fitnesses, torch.tensor([6.]))

def test_interact_memory_resets_per_individual():
    # one slot, two individuals - each must start its own episode from the
    # initial memory, or the second individual would return 22

    interactor = interact_with_env(MemoryCountEnv(), seed = 0)
    pop = interactor.population(Memory(CountingPolicy()), pop_size = 2, low_rank = 2, seed = 0)
    fitnesses = interactor.evaluate(pop, action = lambda out: out, horizon = 20)

    assert torch.equal(fitnesses, torch.tensor([6., 6.]))

def test_interact_memory_vector_per_slot_init():
    # per-slot initial memories on a vector env, with autoreset and the
    # whole-env reset between individuals - slot returns are the runs
    # [10, 11], [20, 21, 22, 23], [30, 31, 32, 33, 34, 35], replayed per individual

    interactor = interact_with_env(MemoryCountVectorEnv(), seed = 0)
    init = torch.tensor([10., 20., 30.])
    pop = interactor.population(Memory(CountingPolicy(), init_memory = init), pop_size = 6, low_rank = 2, seed = 0)
    fitnesses = interactor.evaluate(pop, action = lambda out: out, horizon = 20)

    assert torch.equal(fitnesses, torch.tensor([21., 86., 195., 21., 86., 195.]))

def test_evaluate_policy_with_memory():
    # the final-policy evaluator threads and resets memory too - with a reset
    # every episode returns 6, without one the second episode returns 22

    interactor = interact_with_env(MemoryCountEnv(), seed = 0)
    score = interactor.evaluate_policy(
        Memory(CountingPolicy()),
        action = lambda out: out,
        num_episodes = 2,
        horizon = 20,
        seed = 0
    )

    assert score == 6.

def test_interact_memory_real_gymnasium_vector():
    # a real batch-first GRU policy wrapped in Memory against a vectorized
    # cartpole - the memory is the hidden state, carried step to step

    gym = pytest.importorskip('gymnasium')

    class GRUMemoryPolicy(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden = 16):
            super().__init__()
            self.gru = nn.GRU(obs_dim, hidden, batch_first = True)
            self.head = nn.Linear(hidden, act_dim)

        def forward(self, mem, obs):
            _, mem = self.gru(obs.unsqueeze(1), mem.unsqueeze(0))
            mem = mem.squeeze(0)
            return self.head(mem), mem

    interactor = interact_with_env(gym.make_vec('CartPole-v1', num_envs = 4), seed = 0)
    policy = Memory(GRUMemoryPolicy(4, 2, hidden = 16), init_memory = torch.zeros(1, 16))
    pop = interactor.population(policy, pop_size = 4, low_rank = 2)

    fitnesses = interactor.evaluate(pop, action = lambda logits: logits.argmax(-1), horizon = 100)
    assert fitnesses.shape == (4,)
    assert torch.isfinite(fitnesses).all()

    # the final-policy evaluator on a vector env with the same policy

    score = interactor.evaluate_policy(
        policy,
        action = lambda logits: logits.argmax(-1),
        num_episodes = 2,
        horizon = 100,
        seed = 0
    )
    assert np.isfinite(score)
