import numpy as np
import torch
from torch import nn

from populora import Population, rl_finetune_elites_

# test environments and models

class SimpleEnv:
    def __init__(self, target = 0.5):
        self.target = target
        self.t = 0
        self.horizon = 10

    def reset(self, seed = None):
        self.t = 0
        if seed is not None:
            np.random.seed(seed)
        self.obs = np.array([0.1, 0.2, 0.3, 0.4], dtype = np.float32)
        return self.obs.copy(), {}

    def step(self, action):
        self.t += 1
        a = float(np.asarray(action).reshape(-1)[0])
        reward = float(1.0 - (a - self.target) ** 2)
        done = self.t >= self.horizon
        return self.obs.copy(), reward, done, False, {}

class MultiDimEnv:
    def __init__(self, target = (0.5, -0.3, 0.2)):
        self.target = np.array(target, dtype = np.float32)
        self.t = 0
        self.horizon = 10

    def reset(self, seed = None):
        self.t = 0
        if seed is not None:
            np.random.seed(seed)
        self.obs = np.array([0.1, 0.2, 0.3, 0.4], dtype = np.float32)
        return self.obs.copy(), {}

    def step(self, action):
        self.t += 1
        a = np.asarray(action, dtype = np.float32).reshape(-1)
        reward = float(1.0 - np.mean((a - self.target) ** 2))
        done = self.t >= self.horizon
        return self.obs.copy(), reward, done, False, {}

class SimpleNet(nn.Module):
    def __init__(self, in_dim = 4, out_dim = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))

# tests

def test_backup_and_restore_individual():
    torch.manual_seed(42)
    net = SimpleNet()
    pop = Population(net, pop_size = 4, low_rank = 2)

    backup = pop.backup_individual(1)

    for k in pop.weight_down:
        pop.weight_down[k].data[1].add_(10.0)
        pop.weight_up[k].data[1].add_(10.0)

    for k in pop.weight_down:
        assert not torch.allclose(pop.weight_down[k].data[1], backup['down'][k])

    pop.restore_individual(1, backup)

    for k in pop.weight_down:
        assert torch.allclose(pop.weight_down[k].data[1], backup['down'][k])
        assert torch.allclose(pop.weight_up[k].data[1], backup['up'][k])

def test_monotonic_rl_finetune_elites():
    torch.manual_seed(42)
    net = SimpleNet()
    pop = Population(net, pop_size = 6, low_rank = 2)
    env = SimpleEnv(target = 0.8)

    fitnesses = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 4.0], dtype = torch.float32)

    # snapshot non-elites to ensure they are untouched
    non_elite_down = {k: pop.weight_down[k].data[0].clone() for k in pop.weight_down}
    non_elite_up = {k: pop.weight_up[k].data[0].clone() for k in pop.weight_up}

    # run RL fine tuning with 2 elites
    updated_fits, n_up, mean_gain = rl_finetune_elites_(
        pop,
        fitnesses,
        env,
        num_elites = 2,
        rollouts = 8,
        noise = 0.2,
        lr = 0.1,
        horizon = 10,
        seeds = [123, 456]
    )

    # monotonic guarantee: updated fitnesses for all individuals must be >= initial fitnesses
    assert (updated_fits >= fitnesses).all()

    # non-elites must be untouched
    for k in pop.weight_down:
        assert torch.allclose(pop.weight_down[k].data[0], non_elite_down[k])
        assert torch.allclose(pop.weight_up[k].data[0], non_elite_up[k])

def test_revert_on_no_improvement():
    torch.manual_seed(42)
    net = SimpleNet()
    pop = Population(net, pop_size = 4, low_rank = 2)

    env = SimpleEnv(target = 0.0)
    fitnesses = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype = torch.float32)

    weights_before_down = {k: pop.weight_down[k].data.clone() for k in pop.weight_down}
    weights_before_up = {k: pop.weight_up[k].data.clone() for k in pop.weight_up}

    # if rollouts find no advantage, nothing changes and weights remain identical
    updated_fits, n_up, mean_gain = rl_finetune_elites_(
        pop,
        fitnesses,
        env,
        num_elites = 2,
        rollouts = 2,
        noise = 0.0,
        lr = 0.1,
        horizon = 5,
        seeds = [1]
    )

    assert n_up == 0
    assert mean_gain == 0.0
    for k in pop.weight_down:
        assert torch.allclose(pop.weight_down[k].data, weights_before_down[k])
        assert torch.allclose(pop.weight_up[k].data, weights_before_up[k])

def test_population_method_and_lamarckian_crossover():
    torch.manual_seed(42)
    net = SimpleNet(in_dim = 4, out_dim = 3)
    pop = Population(net, pop_size = 8, low_rank = 2)
    env = MultiDimEnv(target = (0.6, -0.4, 0.3))

    fitnesses = torch.linspace(1.0, 8.0, 8)

    # use the pop.rl_finetune_elites_ method
    updated_fits, n_up, mean_gain = pop.rl_finetune_elites_(
        fitnesses,
        env,
        num_elites = 2,
        rollouts = 8,
        noise = 0.15,
        lr = 0.1,
        horizon = 10,
        seeds = [42, 43]
    )

    assert (updated_fits >= fitnesses).all()

    # evolve to next generation with elitism - Lamarckian transmission
    champ_idx = updated_fits.argmax().item()
    champ_down = {k: pop.weight_down[k][champ_idx].clone() for k in pop.weight_down}
    champ_up = {k: pop.weight_up[k][champ_idx].clone() for k in pop.weight_up}

    pop.evolve_(
        updated_fits,
        survive_frac = 0.5,
        num_elites = 1,
        crossover_type = 'extrapolative',
        epsilon = 0.1
    )

    # elite 0 in next gen should preserve champion's fine-tuned weights
    for k in pop.weight_down:
        assert torch.allclose(pop.weight_down[k][champ_idx], champ_down[k])
        assert torch.allclose(pop.weight_up[k][champ_idx], champ_up[k])
