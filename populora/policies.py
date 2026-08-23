from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

# policy distribution parametrizations - map network logits to env actions

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

def make_categorical_action(
    *,
    sample: bool = True,
    temperature: float = 1.0
):
    # one logit per action - softmax(logits / temperature) -> multinomial,
    # temperature 0 is the argmax

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

    return action_fn

def make_beta_action(
    *,
    sample: bool = True,
    temperature: float = 1.0
):
    # 2 * action_dim logits - alpha then beta via softplus + 1, so the beta
    # is always unimodal on (0, 1). temperature 0 is the mode

    def action_fn(logits):
        alpha_pre, beta_pre = logits.chunk(2, dim = -1)
        alpha = 1.0 + F.softplus(alpha_pre)
        beta = 1.0 + F.softplus(beta_pre)

        mode = (alpha - 1.0) / (alpha + beta - 2.0)

        if not sample or temperature == 0:
            return mode

        if temperature != 1.0:
            alpha = 1.0 + (alpha - 1.0) / temperature
            beta = 1.0 + (beta - 1.0) / temperature

        return torch.distributions.Beta(alpha, beta).sample()

    return action_fn
