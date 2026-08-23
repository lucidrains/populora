from __future__ import annotations

import inspect
import math

import torch
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution
from torch.nn import Module
from torch.nn import functional as F

from populora._utils import default

# policy distribution parametrizations - map network logits to env actions

# unimodal beta distribution policy - mean-concentration reparameterization.
# the first action_dim logits map through a sigmoid to the exact distribution
# mean, the second action_dim to a positive concentration, so the mean and the
# precision are independent knobs at reachable logit magnitudes

class Beta(Module):
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

    def log_prob(
        self,
        params_or_dist,
        action,
        sum_action_dim = True,
        eps = None
    ):
        eps = default(eps, self.eps)
        action = action.clamp(min = eps, max = 1. - eps)
        dist = params_or_dist if isinstance(params_or_dist, Distribution) else self(params_or_dist)
        log_prob = dist.log_prob(action)
        return log_prob.sum(dim = -1) if sum_action_dim else log_prob

    def forward(self, params):
        # mean via sigmoid, concentration via the positive fn, floored to keep
        # alpha, beta > 1 (unimodal) with exact mean = alpha / (alpha + beta)

        raw_mean, raw_conc = params.chunk(2, dim = -1)

        mean = raw_mean.sigmoid().clamp(min = self.eps, max = 1. - self.eps)
        min_mean = torch.minimum(mean, 1. - mean).clamp(min = self.eps)

        conc = self.concentration(raw_conc) + 1. / min_mean

        return TorchBeta(mean * conc, (1. - mean) * conc)

# action factories - each returns an action_fn (logits -> actions) carrying a
# `from_range` the interactor rescales from into the env's to_range

def make_categorical_action(
    *,
    sample: bool = True,
    temperature: float = 1.0
):
    # one logit per action - softmax(logits / temperature) -> multinomial,
    # temperature 0 is the argmax. discrete, so no from_range - the interactor
    # never rescales it

    def action_fn(logits):
        if not sample or temperature == 0:
            return logits.argmax(dim = -1)

        shape = logits.shape
        scaled = logits.reshape(-1, shape[-1]) / temperature
        probs = F.softmax(scaled, dim = -1)
        actions = torch.multinomial(probs, 1).reshape(shape[:-1])
        return actions

    return action_fn

def make_squashed_gaussian_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    min_log_std: float = -5.0,
    max_log_std: float = 0.5
):
    # 2 * action_dim logits - mean then log std, clipped to the range,
    # sampled and tanh squashed into (-1, 1). temperature 0 is the mean

    def action_fn(logits):
        mean, log_std = logits.chunk(2, dim = -1)
        log_std = log_std.clamp(min_log_std, max_log_std)
        std = log_std.exp() * temperature

        if sample:
            action = mean + std * torch.randn_like(mean)
        else:
            action = mean

        return action.tanh()

    action_fn.from_range = (-1., 1.)
    return action_fn

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

    beta_policy = Beta(**kwargs)

    def action_fn(logits):
        if not sample or temperature == 0:
            action = beta_policy.mean(logits)
        else:
            dist = beta_policy(logits)

            if temperature != 1.0:
                # scaling both concentrations preserves the exact mean - smaller temperature sharpens

                dist = TorchBeta(dist.concentration1 / temperature, dist.concentration0 / temperature)

            action = dist.sample()

        if beta_rescale_neg_one_one:
            action = 2.0 * action - 1.0

        return action

    action_fn.from_range = (-1., 1.) if beta_rescale_neg_one_one else (0., 1.)
    return action_fn

def make_alpha_beta_action(
    *,
    sample: bool = True,
    temperature: float = 1.0,
    beta_rescale_neg_one_one: bool = True
):
    # legacy alpha = 1 + softplus, beta = 1 + softplus - the mode is a ratio
    # of the two concentrations, entangling the mean and the precision

    def action_fn(logits):
        alpha_pre, beta_pre = logits.chunk(2, dim = -1)
        alpha = 1.0 + F.softplus(alpha_pre)
        beta = 1.0 + F.softplus(beta_pre)

        mode = (alpha - 1.0) / (alpha + beta - 2.0)

        if not sample or temperature == 0:
            action = mode
        else:
            if temperature != 1.0:
                alpha = 1.0 + (alpha - 1.0) / temperature
                beta = 1.0 + (beta - 1.0) / temperature

            action = TorchBeta(alpha, beta).sample()

        if beta_rescale_neg_one_one:
            action = 2.0 * action - 1.0

        return action

    action_fn.from_range = (-1., 1.) if beta_rescale_neg_one_one else (0., 1.)
    return action_fn

def make_action(
    distribution: str,
    *,
    sample: bool = True,
    temperature: float = 1.0,
    **kwargs
):
    # one entry point - 'categorical' for discrete, 'squashed_gaussian' /
    # 'beta' for continuous. distribution-specific kwargs are dropped when
    # not accepted

    if distribution == 'categorical':
        factory = make_categorical_action
    elif distribution == 'squashed_gaussian':
        factory = make_squashed_gaussian_action
    elif distribution == 'beta':
        factory = make_beta_action
    else:
        raise ValueError(f'unknown action distribution {distribution!r} - must be one of "categorical", "squashed_gaussian", "beta"')

    accepted = inspect.signature(factory).parameters
    kwargs = {name: value for name, value in kwargs.items() if name in accepted}
    return factory(sample = sample, temperature = temperature, **kwargs)
