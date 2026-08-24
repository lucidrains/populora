from __future__ import annotations

import inspect
import math

import torch
from torch.distributions import Beta as TorchBeta
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import Distribution, Normal, TanhTransform, TransformedDistribution
from torch.nn import Module
from torch.nn import functional as F

from populora._utils import default, exists

# policy distribution parametrizations - map network logits to env actions.
# every container is a uniform interface: an nn.Module whose forward maps
# logits to a torch Distribution, with an exact `mean`, a `log_prob` (one
# scalar per item, so RL losses just consume it), and a `from_range` telling
# the interactor what range its samples live in. every action factory returns
# an ActionFn wrapping that container - callable (logits -> env actions) and
# exposing the container through distribution / mean / log_prob. researchers
# can subclass ActionDist and pass their own instance / factory / registered
# name straight into make_action

class ActionDist(Module):
    from_range = None
    event_dim = 0

    def mean(self, params):
        raise NotImplementedError

    def distribution(self, params, temperature = 1.0):
        raise NotImplementedError

    def log_prob(
        self,
        params,
        action,
        sum_action_dim = True,
        eps = None
    ):
        # sum exactly the trailing event dims, never the batch dims - the
        # container's distribution may or may not reduce its own event dim
        # (categorical emits one scalar per item, beta / gaussian one per
        # action dim), so `event_dim` decides

        dist = params if isinstance(params, Distribution) else self.distribution(params)
        log_prob = dist.log_prob(action)

        if sum_action_dim and self.event_dim > 0 and log_prob.dim() >= self.event_dim:
            log_prob = log_prob.sum(dim = tuple(range(-self.event_dim, 0)))

        return log_prob

    def forward(self, params):
        return self.distribution(params)

# squashed gaussian - 2 * action_dim logits: mean then log std, clipped to a
# range, sampled and tanh squashed into (-1, 1). the distribution carries the
# tanh change-of-variables, so log_prob is exact. temperature 0 is the mean

class SquashedGaussian(ActionDist):
    from_range = (-1., 1.)
    event_dim = 1

    def __init__(
        self,
        min_log_std: float = -5.0,
        max_log_std: float = 0.5
    ):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def _params(self, params):
        mean, log_std = params.chunk(2, dim = -1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    def mean(self, params):
        return self._params(params)[0].tanh()

    def distribution(self, params, temperature = 1.0):
        mean, log_std = self._params(params)

        # scalar-base Normal + TanhTransform, matching the SB3 squashed
        # gaussian - log_prob comes back with the action dim intact and the
        # caller (or ActionDist.log_prob) sums it, exactly once

        base = Normal(mean, log_std.exp() * temperature)
        return TransformedDistribution(base, [TanhTransform(cache_size = 1)])

# categorical - one logit per action, softmax(logits / temperature) ->
# multinomial. discrete, so no from_range - the interactor never rescales it

class Categorical(ActionDist):
    def mean(self, params):
        return params.argmax(dim = -1)

    def distribution(self, params, temperature = 1.0):
        return TorchCategorical(logits = params / temperature)

# unimodal beta distribution - mean-concentration reparameterization. the
# first action_dim logits map through a sigmoid to the exact distribution
# mean, the second action_dim to a positive concentration, so the mean and the
# precision are independent knobs at reachable logit magnitudes

class Beta(ActionDist):
    from_range = (0., 1.)
    event_dim = 1

    def __init__(
        self,
        pos_fn = 'softplus',
        init_conc = 10.,
        min_conc = 0.,
        eps = 1e-5
    ):
        super().__init__()
        assert pos_fn in ('exp', 'softplus')
        assert init_conc > min_conc, 'init_conc must be greater than min_conc'

        self.pos_fn = pos_fn
        self.init_conc = init_conc
        self.min_conc = min_conc
        self.eps = eps

        # raw offset into the positive fn so the concentration at raw 0 is exactly init_conc

        self.raw_init_conc = math.log(math.expm1(init_conc - min_conc)) if pos_fn == 'softplus' else math.log(init_conc - min_conc)

    def concentration(
        self,
        raw_conc
    ):
        if self.pos_fn == 'softplus':
            return F.softplus(raw_conc + self.raw_init_conc) + self.min_conc
        elif self.pos_fn == 'exp':
            return (raw_conc + self.raw_init_conc).exp() + self.min_conc

    def mean(
        self,
        params
    ):
        raw_mean, _ = params.chunk(2, dim = -1)
        return raw_mean.sigmoid().clamp(min = self.eps, max = 1. - self.eps)

    def _concentrations(self, params):
        # mean via sigmoid, concentration via the positive fn, floored to keep
        # alpha, beta > 1 (unimodal) with exact mean = alpha / (alpha + beta)

        raw_mean, raw_conc = params.chunk(2, dim = -1)

        mean = raw_mean.sigmoid().clamp(min = self.eps, max = 1. - self.eps)
        min_mean = torch.minimum(mean, 1. - mean).clamp(min = self.eps)

        conc = self.concentration(raw_conc) + 1. / min_mean
        return mean, conc

    def distribution(self, params, temperature = 1.0):
        mean, conc = self._concentrations(params)

        if temperature != 1.0:
            # scaling both concentrations preserves the exact mean - smaller temperature sharpens

            conc = conc / temperature

        return TorchBeta(mean * conc, (1. - mean) * conc)

    def log_prob(
        self,
        params_or_dist,
        action,
        sum_action_dim = True,
        eps = None
    ):
        eps = default(eps, self.eps)
        action = action.clamp(min = eps, max = 1. - eps)
        return super().log_prob(params_or_dist, action, sum_action_dim = sum_action_dim)

# legacy alpha = 1 + softplus, beta = 1 + softplus - the mode is a ratio of
# the two concentrations, entangling the mean and the precision

class AlphaBeta(ActionDist):
    from_range = (0., 1.)
    event_dim = 1

    def _params(self, params):
        alpha_pre, beta_pre = params.chunk(2, dim = -1)
        alpha = 1.0 + F.softplus(alpha_pre)
        beta = 1.0 + F.softplus(beta_pre)
        return alpha, beta

    def mean(self, params):
        alpha, beta = self._params(params)
        return (alpha - 1.0) / (alpha + beta - 2.0)

    def distribution(self, params, temperature = 1.0):
        alpha, beta = self._params(params)

        if temperature != 1.0:
            # scaling keeps the mode fixed

            alpha = 1.0 + (alpha - 1.0) / temperature
            beta = 1.0 + (beta - 1.0) / temperature

        return TorchBeta(alpha, beta)

# the uniform wrapper every action factory returns - callable mapping logits
# to env actions, exposing the underlying distribution container:
#
#   action_fn(logits)                  -> actions (sampled or deterministic mean)
#   action_fn.distribution(logits)     -> the torch Distribution at these logits
#   action_fn.mean(logits)             -> deterministic actions, same space as the call
#   action_fn.log_prob(logits, action) -> log prob of the actions actually stepped,
#                                         one scalar per item - env-space actions
#                                         are mapped back to the dist's native domain
#   action_fn.container                -> the underlying ActionDist module
#   action_fn.from_range               -> range the emitted actions live in

class ActionFn:
    def __init__(
        self,
        container: ActionDist,
        *,
        sample: bool = True,
        temperature: float = 1.0,
        to_env_space = None,
        to_env_space_inv = None
    ):
        self.container = container
        self.sample = sample
        self.temperature = temperature

        # optional mapping from the container's native domain to the env
        # action space, e.g. beta's (0, 1) -> (-1, 1) rescale, and its
        # inverse - log_prob maps env-space actions back

        self.to_env_space = to_env_space
        self.to_env_space_inv = to_env_space_inv
        self.from_range = default(getattr(to_env_space, 'from_range', None), container.from_range)

    def distribution(self, params, temperature = None):
        # temperature 0 is the deterministic mean, handled in __call__ - the
        # dist itself clamps to a tiny positive floor so the beta's
        # concentration scaling stays finite

        temperature = default(temperature, self.temperature)
        return self.container.distribution(params, temperature = max(temperature, 1e-5))

    def mean(self, params):
        action = self.container.mean(params)
        return self.to_env_space(action) if exists(self.to_env_space) else action

    def log_prob(
        self,
        params,
        action,
        sum_action_dim = True,
        eps = None
    ):
        if exists(self.to_env_space_inv):
            action = self.to_env_space_inv(action)

        return self.container.log_prob(params, action, sum_action_dim = sum_action_dim, eps = eps)

    def __call__(self, params):
        if not self.sample or self.temperature == 0:
            return self.mean(params)

        action = self.distribution(params).sample()
        return self.to_env_space(action) if exists(self.to_env_space) else action

# helper - the affine maps carrying a container's native (0, 1) domain out to
# the (-1, 1) env action space and back, tagging the output range so the
# ActionFn picks it up as its from_range

def _unit_rescale(beta_rescale_neg_one_one):
    if not beta_rescale_neg_one_one:
        return None, None

    to_env = lambda action: 2.0 * action - 1.0
    to_env.from_range = (-1., 1.)

    return to_env, lambda action: (action + 1.0) / 2.0

# action factories - each returns an ActionFn (logits -> actions) carrying a
# `from_range` the interactor rescales from into the env's to_range

def make_categorical_action(
    *,
    sample: bool = True,
    temperature: float = 1.0
):
    # one logit per action - softmax(logits / temperature) -> multinomial,
    # temperature 0 is the argmax. discrete, so no from_range - the interactor
    # never rescales it

    return ActionFn(Categorical(), sample = sample, temperature = temperature)

def make_squashed_gaussian_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 0.5
):
    # 2 * action_dim logits - mean then log std, clipped to the range,
    # sampled and tanh squashed into (-1, 1). temperature 0 is the mean

    return ActionFn(
        SquashedGaussian(min_log_std = min_log_std, max_log_std = max_log_std),
        sample = sample,
        temperature = temperature
    )

def make_beta_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    beta_rescale_neg_one_one: bool = True,
    mean_concentration: bool = True,
    **kwargs
):
    # unimodal beta on (0, 1), rescaled to (-1, 1) by default. the
    # mean-concentration reparam (default) decouples the exact mean from the
    # precision; set mean_concentration = False for the legacy alpha/beta
    # mode parametrization

    factory = make_mean_concentration_beta_action if mean_concentration else make_alpha_beta_action
    return factory(sample = sample, temperature = temperature, beta_rescale_neg_one_one = beta_rescale_neg_one_one, **kwargs)

def make_mean_concentration_beta_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    beta_rescale_neg_one_one: bool = True,
    **kwargs
):
    # the mean is exact and precision is an independent knob, so evolution can
    # reach the sharp near-deterministic policies that balance tasks need

    to_env_space, to_env_space_inv = _unit_rescale(beta_rescale_neg_one_one)

    return ActionFn(
        Beta(**kwargs),
        sample = sample,
        temperature = temperature,
        to_env_space = to_env_space,
        to_env_space_inv = to_env_space_inv
    )

def make_alpha_beta_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    beta_rescale_neg_one_one: bool = True
):
    to_env_space, to_env_space_inv = _unit_rescale(beta_rescale_neg_one_one)

    return ActionFn(
        AlphaBeta(),
        sample = sample,
        temperature = temperature,
        to_env_space = to_env_space,
        to_env_space_inv = to_env_space_inv
    )

# custom distributions - researchers register their own factories by name,
# mirroring the mutation / selection / crossover registries. a registered name
# resolves in make_action alongside the builtins

ACTION_DIST_REGISTRY = dict()

def register_action_dist(name: str, factory: callable):
    ACTION_DIST_REGISTRY[name] = factory

_BUILTIN_ACTION_DISTS = None  # populated lazily, after the builtin factories are defined

def _action_dist_factories():
    global _BUILTIN_ACTION_DISTS

    if _BUILTIN_ACTION_DISTS is None:
        _BUILTIN_ACTION_DISTS = dict(
            categorical = make_categorical_action,
            squashed_gaussian = make_squashed_gaussian_action,
            beta = make_beta_action,
        )

    return {**_BUILTIN_ACTION_DISTS, **ACTION_DIST_REGISTRY}

def make_action(
    distribution: str | ActionDist | ActionFn | callable,
    *,
    sample: bool = True,
    temperature: float = 1.0,
    **kwargs
):
    # one entry point - 'categorical' for discrete, 'squashed_gaussian' /
    # 'beta' for continuous, or any name registered through
    # register_action_dist. a researcher can also bring their own:
    #
    #   - an ActionFn passes through as-is
    #   - an ActionDist instance (or subclass) is wrapped in an ActionFn
    #   - any other callable is invoked as a factory, receiving `sample` and
    #     `temperature` when accepted, plus the kwargs it accepts
    #
    # unknown keyword args on the string / factory paths are dropped rather
    # than erroring, so one config works across distributions

    if isinstance(distribution, ActionFn):
        return distribution

    if isinstance(distribution, ActionDist):
        return ActionFn(distribution, sample = sample, temperature = temperature)

    if isinstance(distribution, type) and issubclass(distribution, ActionDist):
        return ActionFn(distribution(), sample = sample, temperature = temperature)

    if callable(distribution):
        factory = distribution
    elif isinstance(distribution, str):
        factory = _action_dist_factories().get(distribution)
    else:
        factory = None

    if not exists(factory):
        known = tuple(_action_dist_factories())
        raise ValueError(f'unknown action distribution {distribution!r} - must be one of {known}, an ActionDist (sub)class or instance, an ActionFn, or a factory callable')

    try:
        accepted = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()

    call_kwargs = {name: value for name, value in kwargs.items() if name in accepted}

    if 'sample' in accepted:
        call_kwargs['sample'] = sample

    if 'temperature' in accepted:
        call_kwargs['temperature'] = temperature

    return factory(**call_kwargs)
