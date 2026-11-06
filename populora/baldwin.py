# baldwinian selection via online next-latent prediction (SPR, schwarzer et al.
# 2020; NextLat, teoh et al. 2025) and fast-weight decay (hinton & plaut 1987)

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, atleast_1d, cat, is_tensor, nn
from torch.nn import Linear, LayerNorm, Module, Sequential, SiLU

from populora._utils import cast_tensor, default, exists
from populora.distributed import preserve_rng
from populora.memory import Memory
from populora.population import Population, _adapter_key, resolve_lora_targets

# loss registry

PREDICTOR_LOSS_REGISTRY = dict()

def register_predictor_loss(name: str, fn: Callable | None = None):
    def _decorator(f: Callable):
        PREDICTOR_LOSS_REGISTRY[name] = f
        return f

    return _decorator(fn) if exists(fn) else _decorator

@register_predictor_loss('cosine')
def cosine_similarity_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_norm = F.normalize(pred.float(), dim = -1, eps = eps)
    target_norm = F.normalize(target.detach().float(), dim = -1, eps = eps)
    return 2. - 2. * (pred_norm * target_norm).sum(dim = -1)

@register_predictor_loss('mse')
def normalized_mse_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_norm = F.normalize(pred.float(), dim = -1, eps = eps)
    target_norm = F.normalize(target.detach().float(), dim = -1, eps = eps)
    return (pred_norm - target_norm).pow(2).sum(dim = -1)

@register_predictor_loss('smooth_l1')
def smooth_l1_latent_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(pred.float(), target.detach().float(), reduction = 'none').mean(dim = -1)

def _resolve_loss_fn(loss_fn: str | Callable) -> Callable:
    if callable(loss_fn):
        return loss_fn
    assert loss_fn in PREDICTOR_LOSS_REGISTRY, f'unknown predictor loss {loss_fn!r} - choose from {tuple(PREDICTOR_LOSS_REGISTRY)}'
    return PREDICTOR_LOSS_REGISTRY[loss_fn]

# action-conditioned residual next-latent predictor

class LatentPredictor(Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        hidden_dim: int | None = None,
        proj_dim: int | None = None,
        discrete_actions: bool = False,
        residual: bool = True,
        transition_net: Module | None = None,
        predictor_head: Module | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.residual = residual

        hidden_dim = default(hidden_dim, max(32, latent_dim))
        proj_dim = default(proj_dim, latent_dim)

        self.latent_norm = LayerNorm(latent_dim)

        self.transition_net = default(transition_net, Sequential(
            Linear(latent_dim + action_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, latent_dim),
        ))

        if exists(predictor_head):
            self.predictor_head = predictor_head
        elif exists(proj_dim) and proj_dim != latent_dim:
            self.predictor_head = Sequential(
                LayerNorm(latent_dim),
                Linear(latent_dim, hidden_dim),
                SiLU(),
                Linear(hidden_dim, proj_dim),
            )
        else:
            self.predictor_head = nn.Identity()

        # small residual init so initial predictions start near identity z_{t+1} ~ z_t

        if not exists(transition_net):
            nn.init.orthogonal_(self.transition_net[-1].weight, gain = 0.1)
            nn.init.zeros_(self.transition_net[-1].bias)

    def _encode_action(self, action: Tensor, batch_size: int, device: torch.device) -> Tensor:
        action = action.to(device)

        if self.discrete_actions or action.dtype in (torch.int32, torch.int64):
            idx = action.long().reshape(batch_size).clamp(0, max(1, self.action_dim - 1))
            return F.one_hot(idx, num_classes = self.action_dim).float()

        if action.ndim == 1:
            action = action.unsqueeze(-1)
        elif action.ndim > 2:
            action = action.reshape(batch_size, -1)

        return action.float()

    def transition(self, latent: Tensor, action: Tensor) -> Tensor:
        b = latent.shape[0]
        z = self.latent_norm(latent.float())
        a = self._encode_action(action, b, latent.device)
        delta = self.transition_net(cat((z, a), dim = -1))
        return (latent.float() + delta) if self.residual else delta

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        return self.predictor_head(self.transition(latent, action))

    def rollout(self, latent: Tensor, actions: Tensor) -> Tensor:
        # unroll k transition steps autoregressively: (b, d), (b, k, ...) -> (b, k, d)

        z = latent.float()
        preds = []

        for step in range(actions.shape[1]):
            z = self.transition(z, actions[:, step])
            preds.append(self.predictor_head(z))

        return torch.stack(preds, dim = 1)

# latent tap - short-circuits the forward at `latent_target` during online encoding

class _StopForward(Exception):
    def __init__(self, latent: Tensor):
        self.latent = latent

class _LatentTap:
    def __init__(self, module: Module, activation: Callable | None = None):
        self.module = module
        self.activation = activation
        self.short_circuit = False
        self.last_latent = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _, __, output):
        z = output[0] if isinstance(output, (tuple, list)) else output
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        if exists(self.activation):
            z = self.activation(z)
        self.last_latent = z
        if self.short_circuit:
            raise _StopForward(z)
        return output

    def remove(self):
        if exists(self._handle):
            self._handle.remove()
            self._handle = None

def _reset_optimizer_slots_(optimizer: torch.optim.Optimizer, params: Sequence[Tensor], env_idx: Tensor):
    # clear per-slot momentum buffers when a slot resets across episodes / individuals

    for p in params:
        state = optimizer.state.get(p, None)
        if not exists(state):
            continue
        for buf in state.values():
            if is_tensor(buf) and buf.ndim >= 1 and buf.shape[0] == p.shape[0]:
                buf.data[env_idx] = 0.

# target resolution - accepts module instances, int indices, or dotted paths / globs

def _module_to_path(root: Module, target: Module) -> str:
    for name, mod in root.named_modules():
        if mod is target:
            return name
    raise ValueError(f'module {target!r} is not a submodule of {type(root).__name__}')

def _strip_net_prefix(root: Module, path: str) -> str:
    try:
        root.get_submodule(path)
        return path
    except AttributeError:
        if path.startswith('net.'):
            stripped = path[4:]
            root.get_submodule(stripped)
            return stripped
        elif hasattr(root, 'net'):
            prefixed = f'net.{path}'
            root.get_submodule(prefixed)
            return prefixed
        raise

def _resolve_single_plastic_spec(pop: Population, spec: str | int | Module) -> tuple[str, ...]:
    model, lora_targets = pop.model, pop.lora_targets

    if isinstance(spec, Module):
        path = _module_to_path(model, spec)
        if isinstance(spec, Linear):
            assert path in lora_targets, f'Linear module at {path!r} is not in population.lora_targets {lora_targets}'
            return (path,)
        prefix = f'{path}.' if path else ''
        matched = tuple(t for t in lora_targets if t == path or t.startswith(prefix))
        assert len(matched) > 0, f'module at {path!r} contains no targeted Linear layers in {lora_targets}'
        return matched

    if isinstance(spec, int):
        container = model.net if (hasattr(model, 'net') and isinstance(model.net, nn.Sequential)) else model
        if spec >= 0 and isinstance(container, nn.Sequential) and spec < len(container):
            sub_path = _module_to_path(model, container[spec])
            if sub_path in lora_targets:
                return (sub_path,)
        return (lora_targets[spec],)

    if isinstance(spec, str):
        try:
            norm_spec = _strip_net_prefix(model, spec)
            sub = model.get_submodule(norm_spec)
            if not isinstance(sub, Linear):
                return _resolve_single_plastic_spec(pop, sub)
            if norm_spec in lora_targets:
                return (norm_spec,)
        except AttributeError:
            pass

        resolved = tuple(resolve_lora_targets(model, (spec,)))
        if len(resolved) == 0 and hasattr(model, 'net'):
            resolved = tuple(f'net.{p}' for p in resolve_lora_targets(model.net, (spec,)))
        assert len(resolved) > 0, f'lora_memory / plastic_targets {spec!r} matched no Linear layers'
        return resolved

    raise TypeError(f'unsupported lora_memory / plastic_targets spec: {spec!r}')

def _resolve_plastic_targets(pop: Population, spec) -> tuple[str, ...]:
    if not exists(spec):
        default_target = pop.lora_targets[-2] if len(pop.lora_targets) >= 2 else pop.lora_targets[0]
        return (default_target,)

    items = (spec,) if isinstance(spec, (str, int, Module)) else tuple(spec)
    out = []
    for item in items:
        for path in _resolve_single_plastic_spec(pop, item):
            if path not in out:
                out.append(path)
    assert len(out) > 0, 'lora_memory / plastic_targets resolved to no Linear modules'
    return tuple(out)

def _resolve_latent_target_and_dim(pop: Population, spec: str | int | Module | None, default_target: str) -> tuple[str, int, Module]:
    model, lora_targets = pop.model, pop.lora_targets

    if not exists(spec):
        path = default_target
    elif isinstance(spec, Module):
        path = _module_to_path(model, spec)
    elif isinstance(spec, int):
        container = model.net if (hasattr(model, 'net') and isinstance(model.net, nn.Sequential)) else model
        if spec >= 0 and isinstance(container, nn.Sequential) and spec < len(container):
            path = _module_to_path(model, container[spec])
        else:
            path = lora_targets[spec]
    elif isinstance(spec, str):
        try:
            path = _strip_net_prefix(model, spec)
        except AttributeError:
            matches = resolve_lora_targets(model, (spec,))
            assert len(matches) > 0, f'latent_layer {spec!r} matched no modules in model'
            path = matches[-1]
    else:
        raise TypeError(f'unsupported latent_layer / latent_target spec: {spec!r}')

    latent_mod = model.get_submodule(path)
    latent_dim = getattr(latent_mod, 'out_features', None)

    if not exists(latent_dim):
        for child in reversed(list(latent_mod.modules())):
            if hasattr(child, 'out_features'):
                latent_dim = child.out_features
                break

    if not exists(latent_dim):
        last_out = None
        for _, mod in model.named_modules():
            if hasattr(mod, 'out_features'):
                last_out = mod.out_features
            if mod is latent_mod:
                break
        latent_dim = last_out

    assert exists(latent_dim), f'could not infer latent_dim for latent_layer {path!r} ({type(latent_mod).__name__})'
    return path, int(latent_dim), latent_mod

# main class - wraps a policy `nn.Module` or a `Population` for baldwinian TTT

class Baldwin(Module):
    def __init__(
        self,
        model_or_pop: Module | Population,
        *,
        latent_layer: str | int | Module | None = None,
        lora_memory: str | int | Module | Sequence[str | int | Module] | None = None,
        plastic_targets: str | int | Module | Sequence[str | int | Module] | None = None,
        latent_target: str | int | Module | None = None,
        action_dim: int | None = None,
        discrete_actions: bool = False,
        latent_activation: Callable | str | None = 'auto',
        predictor: Module | None = None,
        predictor_hidden_dim: int | None = None,
        proj_dim: int | None = None,
        loss_fn: str | Callable = 'cosine',
        lora_lr: float = 1e-3,
        predictor_lr: float = 3e-3,
        lora_warmup_steps: int = 0,
        weight_decay: float = 0.0,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict | None = None,
        k_steps: int = 1,
        update_every: int = 1,
        grad_clip: float | None = 1.0,
        trust_region: float | None = 0.20,
        anchor_decay: float = 0.02,
        target_ema_decay: float | None = 1.0,
        carry_across_episodes: bool = False,
        lamarckian: bool = False,
        assimilate_tau: float = 0.0,
        carry_predictor: bool = True,
    ):
        super().__init__()
        assert k_steps >= 1, 'k_steps must be >= 1'
        assert update_every >= 1, 'update_every must be >= 1'

        self._lora_memory_spec = default(lora_memory, plastic_targets)
        self._latent_layer_spec = default(latent_layer, latent_target)
        self._custom_predictor = predictor
        self._latent_activation_spec = latent_activation

        self.lora_lr = lora_lr
        self.predictor_lr = predictor_lr
        self.lora_warmup_steps = lora_warmup_steps
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = default(optimizer_kwargs, dict())
        self.k_steps = k_steps
        self.update_every = update_every
        self.grad_clip = grad_clip
        self.trust_region = trust_region
        self.anchor_decay = anchor_decay
        self.target_ema_decay = target_ema_decay
        self.carry_across_episodes = carry_across_episodes
        self.assimilate_tau = 1.0 if (lamarckian and assimilate_tau <= 0.) else assimilate_tau
        self.carry_predictor = carry_predictor
        self.discrete_actions = discrete_actions
        self.loss_fn = _resolve_loss_fn(loss_fn)

        self.action_dim = action_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.proj_dim = proj_dim

        self._anchor_down = dict()
        self._anchor_up = dict()
        self._ema_down = dict()
        self._ema_up = dict()
        self._obs_window = []
        self._act_window = []
        self._done_window = []

        self.step_count = 0
        self.loss_history: list[float] = []
        self.last_loss: float | None = None

        self._lora_opt = None
        self._predictor_opt = None
        self._tap = None
        self._saved_requires_grad = None
        self._initial_predictor_state = None

        if isinstance(model_or_pop, Population):
            self.__dict__['net'] = model_or_pop.model
            self.__dict__['population'] = None
            self._bind_population(model_or_pop)
        else:
            self.net = model_or_pop
            self.__dict__['population'] = None
            self.__dict__['predictor'] = None
            self.plastic_targets = None
            self.plastic_keys = None
            self.latent_target = None
            self.latent_dim = None
            self.latent_activation = F.silu if latent_activation == 'auto' else latent_activation

    def _bind_population(self, population: Population):
        # keep population and predictor out of _modules so Population never sees
        # circular references or attaches evolutionary LoRAs onto predictor

        self.__dict__['population'] = population
        population.__dict__['baldwin'] = self

        resolved_plastic = _resolve_plastic_targets(population, self._lora_memory_spec)
        self.plastic_targets = resolved_plastic
        self.plastic_keys = tuple(_adapter_key(p) for p in resolved_plastic)

        latent_path, latent_dim, latent_mod = _resolve_latent_target_and_dim(
            population,
            self._latent_layer_spec,
            default_target = self.plastic_targets[-1],
        )
        self.latent_target = latent_path
        self.latent_dim = latent_dim

        if self._latent_activation_spec == 'auto':
            self.latent_activation = F.silu if isinstance(latent_mod, Linear) else None
        else:
            self.latent_activation = self._latent_activation_spec

        if not exists(self.action_dim):
            last_mod = population.model.get_submodule(population.lora_targets[-1])
            self.action_dim = int(last_mod.out_features)
        else:
            self.action_dim = int(self.action_dim)

        with preserve_rng():
            pred = default(self._custom_predictor, LatentPredictor(
                latent_dim = self.latent_dim,
                action_dim = self.action_dim,
                hidden_dim = self.predictor_hidden_dim,
                proj_dim = self.proj_dim,
                discrete_actions = self.discrete_actions,
            )).to(population.device)

        self.__dict__['predictor'] = pred
        self._initial_predictor_state = copy.deepcopy(pred.state_dict())
        self._predictor_opt = self._make_predictor_opt()
        return self

    def _apply(self, fn):
        super()._apply(fn)
        pred = self.__dict__.get('predictor', None)
        if exists(pred):
            pred._apply(fn)
        return self

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def _clear_window_(self):
        self._obs_window.clear()
        self._act_window.clear()
        self._done_window.clear()

    def _make_predictor_opt(self):
        return self.optimizer_cls(
            self.predictor.parameters(),
            lr = self.predictor_lr,
            weight_decay = self.weight_decay,
            **self.optimizer_kwargs,
        )

    def reset_predictor_(self):
        self.predictor.load_state_dict(copy.deepcopy(self._initial_predictor_state))
        self._predictor_opt = self._make_predictor_opt()
        return self

    def start_rollout(self, individuals: Sequence[int] | Tensor):
        pop = self.population
        assert exists(pop), 'bind Baldwin to a Population before calling start_rollout'

        if not pop._hooks_registered:
            pop.register_hooks()

        if not self.carry_predictor and self.step_count > 0:
            self.reset_predictor_()

        # freeze base model params so autograd only computes grads for plastic loras + predictor

        self._saved_requires_grad = [(p, p.requires_grad) for p in pop.model.parameters()]
        for p, _ in self._saved_requires_grad:
            p.requires_grad_(False)

        env_down, env_up = pop.init_env_loras_(individuals, targets = self.plastic_targets, requires_grad = True)

        self._anchor_down = {k: v.detach().clone() for k, v in env_down.items()}
        self._anchor_up = {k: v.detach().clone() for k, v in env_up.items()}

        if exists(self.target_ema_decay) and 0. < self.target_ema_decay < 1.:
            self._ema_down = {k: v.detach().clone() for k, v in env_down.items()}
            self._ema_up = {k: v.detach().clone() for k, v in env_up.items()}
        else:
            self._ema_down.clear()
            self._ema_up.clear()

        self._lora_opt = self.optimizer_cls(
            pop.env_lora_parameters(),
            lr = self.lora_lr,
            weight_decay = self.weight_decay,
            **self.optimizer_kwargs,
        )

        if exists(self._tap):
            self._tap.remove()
        self._tap = _LatentTap(pop.model.get_submodule(self.latent_target), activation = self.latent_activation)
        self._clear_window_()

        return self

    def end_rollout(self):
        if exists(self._tap):
            self._tap.remove()
            self._tap = None

        if exists(self._saved_requires_grad):
            for p, req in self._saved_requires_grad:
                p.requires_grad_(req)
            self._saved_requires_grad = None

        self.population.clear_env_loras_()
        self._anchor_down.clear()
        self._anchor_up.clear()
        self._ema_down.clear()
        self._ema_up.clear()
        self._clear_window_()
        self._lora_opt = None
        return self

    @contextmanager
    def session(self, individuals: Sequence[int] | Tensor):
        self.start_rollout(individuals)
        try:
            yield self
        finally:
            self.end_rollout()

    @torch.no_grad()
    def reset_slots(
        self,
        env_indices: int | Sequence[int] | Tensor,
        next_individuals: int | Sequence[int] | Tensor,
        *,
        completed_individuals: int | Sequence[int] | Tensor | None = None,
        episode_boundary_only: bool = False,
    ):
        pop = self.population
        if not pop._env_weight_down:
            return self

        env_idx = atleast_1d(cast_tensor(env_indices, device = pop.device, dtype = torch.long))
        next_ind = atleast_1d(cast_tensor(next_individuals, device = pop.device, dtype = torch.long))

        if episode_boundary_only and self.carry_across_episodes:
            self._clear_window_()
            return self

        if self.assimilate_tau > 0. and exists(completed_individuals):
            pop.assimilate_env_loras_(env_idx, completed_individuals, tau = self.assimilate_tau)

        pop.reset_env_loras_(env_idx, next_ind)

        for key in self.plastic_keys:
            self._anchor_down[key].data[env_idx] = pop._env_weight_down[key].data[env_idx]
            self._anchor_up[key].data[env_idx] = pop._env_weight_up[key].data[env_idx]

            if key in self._ema_down:
                self._ema_down[key].data[env_idx] = pop._env_weight_down[key].data[env_idx]
                self._ema_up[key].data[env_idx] = pop._env_weight_up[key].data[env_idx]

        if exists(self._lora_opt):
            _reset_optimizer_slots_(self._lora_opt, pop.env_lora_parameters(), env_idx)

        self._clear_window_()
        return self

    def encode(
        self,
        obs: Tensor,
        *,
        individuals: int | Sequence[int] | Tensor | None = None,
        env_indices: Sequence[int] | Tensor | None = None,
        memory: Tensor | None = None,
        use_ema_target: bool = False,
    ) -> Tensor:
        pop = self.population
        assert exists(self._tap), 'call start_rollout before encode'

        swapped = False
        if use_ema_target and exists(self.target_ema_decay) and self.target_ema_decay > 0.:
            live_down, live_up = dict(pop._env_weight_down), dict(pop._env_weight_up)
            if len(self._ema_down) > 0:
                pop._env_weight_down = self._ema_down
                pop._env_weight_up = self._ema_up
            else:
                pop._env_weight_down = self._anchor_down
                pop._env_weight_up = self._anchor_up
            swapped = True

        self._tap.short_circuit = True
        try:
            if isinstance(pop.model, Memory):
                mem = default(memory, pop.model.init_memory.expand(obs.shape[0], -1))
                pop(mem, obs, individuals = individuals, env_indices = env_indices, eval_and_no_grad = False)
            else:
                pop(obs, individuals = individuals, env_indices = env_indices, eval_and_no_grad = False)
        except _StopForward as stop:
            return stop.latent
        finally:
            self._tap.short_circuit = False
            if swapped:
                pop._env_weight_down = live_down
                pop._env_weight_up = live_up

        return self._tap.last_latent

    @torch.no_grad()
    def _project_trust_region_and_ema_(self, env_indices: Tensor | None = None):
        pop = self.population
        idx = env_indices if exists(env_indices) else slice(None)

        if self.anchor_decay > 0.:
            for key in self.plastic_keys:
                pop._env_weight_down[key].data[idx].lerp_(self._anchor_down[key].data[idx], self.anchor_decay)
                pop._env_weight_up[key].data[idx].lerp_(self._anchor_up[key].data[idx], self.anchor_decay)

        if exists(self.trust_region) and self.trust_region > 0.:
            rho = self.trust_region
            for key in self.plastic_keys:
                for live, anchor in (
                    (pop._env_weight_down[key], self._anchor_down[key]),
                    (pop._env_weight_up[key], self._anchor_up[key]),
                ):
                    w_live = live.data[idx]
                    w_anc = anchor.data[idx]
                    delta = w_live - w_anc
                    delta_norm = delta.norm(dim = (-2, -1), keepdim = True).clamp(min = 1e-8)
                    max_norm = rho * w_anc.norm(dim = (-2, -1), keepdim = True).clamp(min = 0.1)
                    scale = (max_norm / delta_norm).clamp(max = 1.0)
                    live.data[idx] = w_anc + delta * scale

        if exists(self.target_ema_decay) and 0. < self.target_ema_decay < 1. and len(self._ema_down) > 0:
            tau = self.target_ema_decay
            for key in self.plastic_keys:
                self._ema_down[key].data[idx].lerp_(pop._env_weight_down[key].data[idx], 1. - tau)
                self._ema_up[key].data[idx].lerp_(pop._env_weight_up[key].data[idx], 1. - tau)

    def step(
        self,
        obs: Tensor,
        action: Tensor,
        next_obs: Tensor,
        *,
        individuals: int | Sequence[int] | Tensor | None = None,
        env_indices: Sequence[int] | Tensor | None = None,
        dones: Tensor | None = None,
        memory: Tensor | None = None,
    ) -> float | None:
        pop = self.population
        device = pop.device
        self.step_count += 1

        obs = obs.to(device) if is_tensor(obs) else torch.as_tensor(obs, device = device)
        next_obs = next_obs.to(device) if is_tensor(next_obs) else torch.as_tensor(next_obs, device = device)
        action = action.to(device) if is_tensor(action) else torch.as_tensor(action, device = device)

        b = obs.shape[0]
        if exists(dones):
            dones = dones.to(device = device, dtype = torch.bool).reshape(b)
        else:
            dones = torch.zeros(b, device = device, dtype = torch.bool)

        if exists(env_indices) and not is_tensor(env_indices):
            env_indices = cast_tensor(env_indices, device = device, dtype = torch.long)

        if self.k_steps > 1 and not exists(env_indices):
            self._obs_window.append(obs.detach())
            self._act_window.append(action.detach())
            self._done_window.append(dones.detach())
            if len(self._obs_window) > self.k_steps:
                self._obs_window.pop(0)
                self._act_window.pop(0)
                self._done_window.pop(0)

        if self.step_count % self.update_every != 0:
            return None

        valid = ~dones
        if not valid.any():
            return None

        with torch.enable_grad():
            self._lora_opt.zero_grad(set_to_none = True)
            self._predictor_opt.zero_grad(set_to_none = True)

            if self.k_steps > 1 and len(self._obs_window) == self.k_steps and not exists(env_indices):
                z_start = self.encode(self._obs_window[0], individuals = individuals, memory = memory)
                actions_seq = torch.stack(self._act_window, dim = 1)
                preds_seq = self.predictor.rollout(z_start, actions_seq)

                target_list = []
                with torch.no_grad():
                    for k_i in range(1, self.k_steps):
                        target_list.append(self.encode(self._obs_window[k_i], individuals = individuals, use_ema_target = True))
                    target_list.append(self.encode(next_obs, individuals = individuals, use_ema_target = True))
                targets_seq = torch.stack(target_list, dim = 1)

                alive = torch.ones(b, device = device, dtype = torch.bool)
                step_losses = []
                for k_i in range(self.k_steps):
                    alive = alive & (~self._done_window[k_i])
                    if alive.any():
                        l_k = self.loss_fn(preds_seq[:, k_i], targets_seq[:, k_i])
                        step_losses.append(l_k[alive].sum())

                if not step_losses:
                    return None

                total_loss = torch.stack(step_losses).mean()
                n_valid = max(1, int(valid.sum().item()))
            else:
                z_t = self.encode(obs, individuals = individuals, env_indices = env_indices, memory = memory)
                pred_next_z = self.predictor(z_t, action)

                with torch.no_grad():
                    target_next_z = self.encode(
                        next_obs,
                        individuals = individuals,
                        env_indices = env_indices,
                        memory = memory,
                        use_ema_target = True,
                    )

                per_env_loss = self.loss_fn(pred_next_z, target_next_z)
                valid_losses = per_env_loss[valid]
                n_valid = max(1, int(valid_losses.numel()))

                # sum over envs so per-slot plastic lora grads are invariant to num_envs
                total_loss = valid_losses.sum()

            total_loss.backward()

            with torch.no_grad():
                for p in self.predictor.parameters():
                    if exists(p.grad):
                        p.grad.div_(n_valid)

                if exists(self.grad_clip):
                    nn.utils.clip_grad_norm_(self.predictor.parameters(), self.grad_clip)
                    for param in pop.env_lora_parameters():
                        if exists(param.grad):
                            g_norm = param.grad.norm(dim = (-2, -1), keepdim = True).clamp(min = 1e-8)
                            scale = (self.grad_clip / g_norm).clamp(max = 1.0)
                            param.grad.mul_(scale)

            if self.step_count > self.lora_warmup_steps:
                self._lora_opt.step()
            self._predictor_opt.step()

            if self.step_count > self.lora_warmup_steps:
                self._project_trust_region_and_ema_(env_indices = env_indices)

            mean_loss = float(total_loss.item() / n_valid)
            self.last_loss = mean_loss
            self.loss_history.append(mean_loss)
            return mean_loss
