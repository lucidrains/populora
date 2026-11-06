import torch
from torch import nn
import gymnasium as gym

from populora import (
    Population,
    Baldwin,
    LatentPredictor,
    Memory,
    PREDICTOR_LOSS_REGISTRY,
    register_predictor_loss,
    interact_with_env,
)


def _make_mlp(in_dim = 8, hidden = 16, out_dim = 2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def test_env_loras_shape_and_independent_slot_gradients():
    model = _make_mlp(8, 16, 2)
    pop = Population(model, pop_size = 6, low_rank = 4)

    # route 4 environment slots to individuals [0, 2, 4, 1]
    individuals = [0, 2, 4, 1]
    env_down, env_up = pop.init_env_loras_(individuals, targets = ('2',), requires_grad = True)

    assert len(env_down) == 1
    key = next(iter(env_down.keys()))
    assert env_down[key].shape == (4, 16, 4)
    assert env_up[key].shape == (4, 16, 4)

    # forward on all 4 slots, backprop only through slot 1
    x = torch.randn(4, 8)
    out = pop(x, individuals = individuals, eval_and_no_grad = False)
    loss = out[1].sum()
    loss.backward()

    # only slot 1 should receive non-zero gradient
    grad_d = env_down[key].grad
    assert grad_d[1].abs().sum() > 0
    assert grad_d[0].abs().sum() == 0
    assert grad_d[2].abs().sum() == 0
    assert grad_d[3].abs().sum() == 0

    # subset routing via env_indices
    sub_out = pop(x[:2], individuals = individuals[:2], env_indices = [0, 1], eval_and_no_grad = True)
    assert sub_out.shape == (2, 2)

    pop.clear_env_loras_()
    assert len(pop._env_weight_down) == 0


def test_latent_predictor_1step_and_kstep_rollout():
    pred_cont = LatentPredictor(latent_dim = 16, action_dim = 3, discrete_actions = False)
    z = torch.randn(5, 16)
    a = torch.randn(5, 3)
    out_1 = pred_cont(z, a)
    assert out_1.shape == (5, 16)

    a_seq = torch.randn(5, 4, 3)
    out_k = pred_cont.rollout(z, a_seq)
    assert out_k.shape == (5, 4, 16)

    pred_disc = LatentPredictor(latent_dim = 12, action_dim = 4, discrete_actions = True)
    z_d = torch.randn(5, 12)
    a_d = torch.tensor([0, 1, 2, 3, 1])
    assert pred_disc(z_d, a_d).shape == (5, 12)

    a_d_seq = torch.randint(0, 4, (5, 3))
    assert pred_disc.rollout(z_d, a_d_seq).shape == (5, 3, 12)


def test_baldwin_online_ttt_and_carry_across_episodes():
    torch.manual_seed(0)
    model = _make_mlp(8, 16, 2)
    pop = Population(model, pop_size = 4, low_rank = 4)

    # verify carry_across_episodes = False resets plastic loras at episode boundaries
    baldwin_reset = Baldwin(
        pop,
        action_dim = 2,
        lora_lr = 5e-2,
        predictor_lr = 1e-2,
        carry_across_episodes = False,
    )

    inds = [0, 1, 2, 3]
    baldwin_reset.start_rollout(inds)
    key = baldwin_reset.plastic_keys[0]
    init_down = pop._env_weight_down[key].detach().clone()

    obs = torch.randn(4, 8)
    act = torch.randn(4, 2)
    next_obs = torch.randn(4, 8)

    loss = baldwin_reset.step(obs, act, next_obs, individuals = inds)
    assert loss is not None
    # plastic lora must have updated away from init_down
    assert not torch.allclose(pop._env_weight_down[key], init_down)

    # episode boundary reset with carry_across_episodes = False restores genotype
    baldwin_reset.reset_slots([0, 1], [0, 1], episode_boundary_only = True)
    assert torch.allclose(pop._env_weight_down[key][:2], init_down[:2])
    assert not torch.allclose(pop._env_weight_down[key][2:], init_down[2:])
    baldwin_reset.end_rollout()

    # verify carry_across_episodes = True preserves plastic loras across episodes
    baldwin_carry = Baldwin(
        pop,
        action_dim = 2,
        lora_lr = 5e-2,
        predictor_lr = 1e-2,
        carry_across_episodes = True,
    )
    baldwin_carry.start_rollout(inds)
    baldwin_carry.step(obs, act, next_obs, individuals = inds)
    adapted_down = pop._env_weight_down[key].detach().clone()

    # episode boundary with carry_across_episodes = True keeps adapted weights
    baldwin_carry.reset_slots([0, 1], [0, 1], episode_boundary_only = True)
    assert torch.allclose(pop._env_weight_down[key], adapted_down)

    # tour boundary (episode_boundary_only = False) always resets to next individual
    baldwin_carry.reset_slots([0], [2], completed_individuals = [0], episode_boundary_only = False)
    assert torch.allclose(pop._env_weight_down[key][0], pop.weight_down[key][2])
    baldwin_carry.end_rollout()


def test_baldwin_pure_vs_lamarckian_assimilation():
    torch.manual_seed(42)
    model = _make_mlp(6, 16, 2)
    pop = Population(model, pop_size = 4, low_rank = 4)
    key = '2'
    genome_before = pop.weight_down[key].detach().clone()

    # pure baldwinian (lamarckian = False): genome untouched after rollout
    baldwin_pure = Baldwin(pop, action_dim = 2, lora_lr = 5e-2, lamarckian = False)
    with baldwin_pure.session([0, 1, 2, 3]):
        baldwin_pure.step(torch.randn(4, 6), torch.randn(4, 2), torch.randn(4, 6), individuals = [0, 1, 2, 3])
        baldwin_pure.reset_slots([0], [0], completed_individuals = [0], episode_boundary_only = False)

    assert torch.allclose(pop.weight_down[key], genome_before)

    # lamarckian (assimilate_tau = 0.5): genome blends toward plastic lora
    baldwin_lam = Baldwin(pop, action_dim = 2, lora_lr = 5e-2, assimilate_tau = 0.5)
    with baldwin_lam.session([0, 1, 2, 3]):
        baldwin_lam.step(torch.randn(4, 6), torch.randn(4, 2), torch.randn(4, 6), individuals = [0, 1, 2, 3])
        baldwin_lam.reset_slots([0], [0], completed_individuals = [0], episode_boundary_only = False)

    assert not torch.allclose(pop.weight_down[key][0], genome_before[0])
    assert torch.allclose(pop.weight_down[key][1:], genome_before[1:])


def test_baldwin_model_wrapper_latent_layer_and_lora_memory():
    torch.manual_seed(11)

    # 1. wrap by submodule references (including tapping an activation layer `net[3]` for latent_layer)
    net = _make_mlp(8, 16, 2)
    wrapped = Baldwin(
        net,
        latent_layer = net[3],       # SiLU activation module right after Linear '2'
        lora_memory = [net[0], net[2]],
        lora_lr = 1e-2,
    )
    # forward works directly on the wrapped model before Population is built
    assert wrapped(torch.randn(3, 8)).shape == (3, 2)

    pop = Population(wrapped, pop_size = 4, low_rank = 4)
    assert pop.model is net
    assert hasattr(pop, 'baldwin') and pop.baldwin is wrapped
    assert wrapped.plastic_targets == ('0', '2')
    assert wrapped.latent_target == '3'
    assert wrapped.latent_dim == 16
    assert wrapped.latent_activation is None  # net[3] is already SiLU

    with wrapped.session([0, 1, 2, 3]):
        loss = wrapped.step(torch.randn(4, 8), torch.randn(4, 2), torch.randn(4, 8), individuals = [0, 1, 2, 3])
        assert loss is not None

    # 2. wrap by integer indices (`latent_layer = 2, lora_memory = -2`)
    net2 = _make_mlp(8, 16, 2)
    wrapped2 = Baldwin(net2, latent_layer = 2, lora_memory = -2)
    pop2 = Population(wrapped2, pop_size = 4, low_rank = 4)
    assert pop2.baldwin.plastic_targets == ('2',)
    assert pop2.baldwin.latent_target == '2'
    assert pop2.baldwin.latent_activation is not None

    # 3. compose Baldwin with Memory in either order
    class RecurrentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(6, 12)
            self.act = nn.SiLU()
            self.head = nn.Linear(12, 2)

        def forward(self, mem, obs):
            h = self.act(self.enc(obs) + mem)
            return self.head(h), h

    rec = RecurrentNet()
    mem_wrapped = Memory(Baldwin(rec, latent_layer = rec.enc, lora_memory = rec.enc), init_memory = torch.zeros(1, 12))
    pop_mem = Population(mem_wrapped, pop_size = 4, low_rank = 2)
    assert isinstance(pop_mem.model, Memory)
    assert pop_mem.baldwin.plastic_targets == ('net.enc',)
    assert pop_mem.baldwin.latent_target == 'net.enc'


def test_baldwin_wrapped_model_in_interactor_evolve():
    torch.manual_seed(7)
    interactor = interact_with_env(gym.make_vec('CartPole-v1', num_envs = 2), seed = 0)
    try:
        net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )
        model = Baldwin(
            net,
            latent_layer = net[2],
            lora_memory = net[2],
            carry_across_episodes = True,
            k_steps = 2,
            lora_lr = 1e-2,
        )
        pop = interactor.population(model, pop_size = 4, low_rank = 4, seed = 0)

        # evaluate automatically uses the bound Baldwin wrapper
        fitnesses = interactor.evaluate(pop, horizon = 20, num_episodes = 2)
        assert fitnesses.shape == (4,)
        n_losses = len(pop.baldwin.loss_history)
        assert n_losses > 0

        # passing baldwin = False explicitly disables TTT for ablation
        interactor.evaluate(pop, horizon = 20, num_episodes = 1, baldwin = False)
        assert len(pop.baldwin.loss_history) == n_losses

        # evolve directly from a Baldwin-wrapped model
        net_evolve = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 2))
        wrapped_evolve = Baldwin(net_evolve, latent_layer = net_evolve[2], lora_memory = net_evolve[2])
        policy, history = interactor.evolve(
            wrapped_evolve,
            pop_size = 4,
            low_rank = 4,
            num_generations = 2,
            horizon = 15,
            num_episodes = 2,
            return_history = True,
        )
        assert len(history) == 2
        assert isinstance(policy, nn.Module)
    finally:
        interactor.close()


def test_baldwin_custom_loss_and_string_or_container_targets():
    @register_predictor_loss('custom_l2_test')
    def _custom_l2(pred, target):
        return (pred - target.detach()).pow(2).mean(dim = -1)

    assert 'custom_l2_test' in PREDICTOR_LOSS_REGISTRY

    class TwoBlockPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(6, 12), nn.SiLU(), nn.Linear(12, 12))
            self.head = nn.Linear(12, 2)

        def forward(self, x):
            return self.head(torch.relu(self.trunk(x)))

    policy = TwoBlockPolicy()
    # pass container module `policy.trunk` as lora_memory and string path 'trunk.2' as latent_layer
    wrapped = Baldwin(
        policy,
        latent_layer = 'trunk.2',
        lora_memory = policy.trunk,
        loss_fn = 'custom_l2_test',
        lora_lr = 1e-2,
    )
    pop = Population(wrapped, pop_size = 4, low_rank = 2)
    assert pop.baldwin.plastic_targets == ('trunk.0', 'trunk.2')
    assert pop.baldwin.latent_target == 'trunk.2'
    assert pop.baldwin.latent_dim == 12

    with pop.baldwin.session([0, 1, 2, 3]):
        loss = pop.baldwin.step(torch.randn(4, 6), torch.randn(4, 2), torch.randn(4, 6), individuals = [0, 1, 2, 3])
        assert loss is not None

