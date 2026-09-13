from __future__ import annotations

import inspect

import torch
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import Distribution, Normal, TanhTransform, TransformedDistribution
from torch.nn import Module

from env_ssl_wrapper import action_space_bounds, action_space_is_box, action_space_is_discrete
from mean_conc_beta import Beta as MeanConcBeta

from populora._utils import default, exists

# policy distribution parametrizations - nn.Modules mapping logits to a torch
# distribution with mean / log_prob / from_range; every action factory returns
# an ActionFn (logits -> actions). subclass or register through make_action

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
        # sum exactly the trailing event dims, never the batch dims - `event_dim`
        # decides, since distributions reduce their own event dims or not

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

# unimodal beta, mean-concentration reparam

class Beta(ActionDist):
    from_range = (-1., 1.)
    event_dim = 1

    def __init__(self, bounds = (-1., 1.), **kwargs):
        super().__init__()
        accepted = inspect.signature(MeanConcBeta.__init__).parameters
        beta_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        self.beta = MeanConcBeta(bounds = bounds, **beta_kwargs)
        self.from_range = tuple(map(float, self.beta.bounds))

    def _format_params(self, params):
        if params.ndim >= 3 and params.shape[-1] == 2:
            return params

        raw_mean, raw_conc = params.chunk(2, dim = -1)
        return torch.stack([raw_mean, raw_conc], dim = -1)

    def concentration(self, params, indexed = False):
        if isinstance(params, Distribution):
            return self.beta.concentration(params)

        if indexed or (params.ndim == 1 and params.shape[0] % 2 != 0):
            return self.beta.concentration(params, indexed = True)

        return self.beta.concentration(self._format_params(params))

    def mean(self, params):
        if isinstance(params, Distribution):
            return params.mean

        return self.beta.mean(self._format_params(params))

    def distribution(self, params, temperature = 1.0):
        return self.beta(self._format_params(params), temperature = temperature)

# the uniform ActionFn wrapper every factory returns - callable on logits,
# exposing distribution / mean / log_prob / container / from_range

class ActionFn:
    def __init__(
        self,
        container: ActionDist,
        *,
        sample: bool = True,
        temperature: float = 1.0,
    ):
        self.container = container
        self.sample = sample
        self.temperature = temperature
        self.from_range = container.from_range

    def distribution(self, params, temperature = None):
        temperature = default(temperature, self.temperature)
        return self.container.distribution(params, temperature = max(temperature, 1e-5))

    def mean(self, params):
        return self.container.mean(params)

    def log_prob(
        self,
        params,
        action,
        sum_action_dim = True,
        eps = None
    ):
        return self.container.log_prob(params, action, sum_action_dim = sum_action_dim, eps = eps)

    def __call__(self, params):
        if not self.sample or self.temperature == 0:
            return self.mean(params)

        return self.distribution(params).sample()

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
    bounds: tuple[float, float] | None = None,
    beta_rescale_neg_one_one: bool = True,
    **kwargs
):
    # unimodal beta, rescaled to (-1, 1) by default

    bounds = default(bounds, (-1., 1.) if beta_rescale_neg_one_one else (0., 1.))

    return ActionFn(
        Beta(bounds = bounds, **kwargs),
        sample = sample,
        temperature = temperature,
    )

make_mean_concentration_beta_action = make_beta_action

# custom distributions - researchers register their own factories by name,
# mirroring the mutation / selection / crossover registries. a registered name
# resolves in make_action alongside the builtins

ACTION_DIST_REGISTRY = dict()

def register_action_dist(name: str, factory: callable):
    ACTION_DIST_REGISTRY[name] = factory

def _action_dist_factories():
    return {
        'categorical': make_categorical_action,
        'squashed_gaussian': make_squashed_gaussian_action,
        'beta': make_beta_action,
        **ACTION_DIST_REGISTRY,
    }

def make_action(
    distribution: str | ActionDist | ActionFn | callable,
    *,
    sample: bool = True,
    temperature: float = 1.0,
    **kwargs
):
    # one entry point - a builtin or registered name passes through, an ActionFn
    # passes through as-is, an ActionDist instance / class is wrapped, any other
    # callable is used as a factory. unknown kwargs are dropped

    if isinstance(distribution, ActionFn):
        return distribution

    if isinstance(distribution, ActionDist):
        return ActionFn(distribution, sample = sample, temperature = temperature)

    if isinstance(distribution, type) and issubclass(distribution, ActionDist):
        return ActionFn(distribution(), sample = sample, temperature = temperature)

    if action_space_is_discrete(distribution):
        return make_categorical_action(sample = sample, temperature = temperature, **kwargs)

    if action_space_is_box(distribution):
        bounds = default(kwargs.get('bounds'), action_space_bounds(distribution))
        return make_beta_action(sample = sample, temperature = temperature, bounds = bounds, **kwargs)

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

    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values())
    call_kwargs = {name: value for name, value in kwargs.items() if has_var_kwargs or name in accepted}

    if 'sample' in accepted:
        call_kwargs['sample'] = sample

    if 'temperature' in accepted:
        call_kwargs['temperature'] = temperature

    return factory(**call_kwargs)
