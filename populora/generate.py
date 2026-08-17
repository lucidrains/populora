from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
from torch import Tensor, cat, is_tensor

from einops import repeat

from populora._utils import cast_tensor, default, divisible_by, exists
from populora.distributed import preserve_rng
from populora.population import Population

# helpers

def _split_output(output, cache_enabled):
    # a model call yields logits, and (when a cache is requested) the updated
    # cache alongside - as a tuple, or as a `past_key_values` attribute (hf).
    # tuple outputs are only interpreted as a cache when the user opted in, so
    # models returning (logits, anything-else) keep working without a cache

    if is_tensor(output):
        return output, None

    if cache_enabled:
        if isinstance(output, dict) and 'past_key_values' in output:
            return output['logits'], output['past_key_values']

        if isinstance(output, (tuple, list)) and len(output) == 2:
            return output

    return output, None

def _slice_cache(cache, mask, batch_size):
    # slice a cache structure's batch dimension by an active mask, recursing into
    # tensors and the common cache containers (tuples, dicts, dataclasses)

    if is_tensor(cache):
        return cache[mask] if cache.ndim > 0 and cache.shape[0] == batch_size else cache

    if isinstance(cache, (tuple, list)):
        return type(cache)(_slice_cache(item, mask, batch_size) for item in cache)

    if isinstance(cache, dict):
        return {key: _slice_cache(value, mask, batch_size) for key, value in cache.items()}

    if dataclasses.is_dataclass(cache) and not isinstance(cache, type):
        return dataclasses.replace(cache, **{
            field.name: _slice_cache(getattr(cache, field.name), mask, batch_size)
            for field in dataclasses.fields(cache) if field.init
        })

    return cache

# main function

@torch.no_grad()
def generate(
    population: Population,
    ids: Tensor,
    *,
    individual: int | None = None,
    individuals: Sequence[int] | Tensor | None = None,
    all_individuals: bool = False,
    max_len: int = 128,
    sample_fn: callable | None = None,
    eos_token: int | None = None,
    stop_fn: callable | None = None,
    cache_kwarg: str = 'cache',
    cache_last_token: bool = False,
    ignore_args_kwargs: Sequence[int | str] = tuple(),
    forward_kwargs: dict | None = None,
    cache_kwargs: dict | None = None,
    micro_batch: int | None = None
) -> Tensor:
    # autoregressively generate from a population of loras, with one routed
    # forward per step over the samples still active. each sample is routed to
    # its individual, so a population can be decoded in a single batched loop
    #
    # kv caching is opt-in via `cache_kwarg` + `cache_kwargs` - the updated cache
    # returned by the model is passed back under `cache_kwarg` on the next step,
    # e.g. `cache_kwargs = dict(return_intermediates = True)` for x-transformers
    # or `cache_kwarg = 'past_key_values', cache_kwargs = dict(use_cache = True)`
    # for huggingface
    #
    # `sample_fn` receives the (b, vocab) logits and returns the next tokens;
    # default is greedy argmax. `stop_fn` receives (tokens, logits, step) and
    # returns which samples are done, in addition to `eos_token`. samples that
    # finish early are compacted out of the batch, and their cache rows dropped

    assert sum((exists(individual), exists(individuals), all_individuals)) == 1

    device = population.device
    ids = cast_tensor(ids, device)
    assert ids.ndim == 2, 'ids must be a 2d tensor of token ids'

    batch_size, prompt_len = ids.shape
    assert max_len >= prompt_len, f'max_len {max_len} must be at least the prompt length {prompt_len}'

    if exists(individual):
        individuals = torch.full((batch_size,), individual, dtype = torch.long, device = device)
    elif all_individuals:
        assert divisible_by(batch_size, population.pop_size), f'batch {batch_size} must be a multiple of the population size {population.pop_size}'
        individuals = repeat(torch.arange(population.pop_size, device = device), 'p -> (p b)', b = batch_size // population.pop_size)
    elif not is_tensor(individuals):
        individuals = torch.tensor(individuals, device = device)

    assert len(individuals) == batch_size

    forward_kwargs = default(forward_kwargs, dict())
    cache_kwargs = default(cache_kwargs, dict())
    cache_enabled = len(cache_kwargs) > 0

    assert not (exists(micro_batch) and cache_enabled), 'micro_batch cannot be combined with cache_kwargs'

    fill = default(eos_token, 0)

    seqs = torch.full((batch_size, max_len), fill, dtype = torch.long, device = device)
    seqs[:, :prompt_len] = ids

    # `seqs` stays full-size and final - dropped rows never change again, so
    # only the scratch state (cache, routes, active) is compacted, through
    # `rows`, which maps each compacted row back to its original index

    rows = torch.arange(batch_size, device = device)
    active = torch.ones(batch_size, dtype = torch.bool, device = device)

    cache = None
    length = prompt_len
    step = 0

    def decode(x, **kwargs):
        out = population(
            x,
            individuals = individuals,
            ignore_args_kwargs = ignore_args_kwargs,
            eval_and_no_grad = True,
            **kwargs
        )
        return _split_output(out, cache_enabled)

    def next_tokens(logits):
        logits = logits[:, -1]
        tokens = sample_fn(logits) if exists(sample_fn) else logits.argmax(dim = -1)
        return cast_tensor(tokens, device), logits

    def done(tokens, logits, step):
        mask = (tokens == eos_token) if exists(eos_token) else torch.zeros(len(tokens), dtype = torch.bool, device = device)

        if exists(stop_fn):
            mask = mask | cast_tensor(stop_fn(tokens, logits, step), device).bool()

        return mask

    # prefill - chunked over the batch when micro_batch is set (only valid
    # without a cache, asserted above)

    if exists(micro_batch):
        logits = cat([
            decode(ids[start:start + micro_batch], **forward_kwargs)[0]
            for start in range(0, batch_size, micro_batch)
        ], dim = 0)
    else:
        cache_inject = {cache_kwarg: cache} if cache_enabled else dict()
        logits, cache = decode(ids, **cache_inject, **forward_kwargs, **cache_kwargs)

    with preserve_rng():
        # the first step consumes the prefill logits directly - feeding the
        # full prompt back with its own cache would append the last prompt
        # token to the cache twice

        while active.any() and length < max_len:
            if not active.all():
                # drop the finished samples - cache rows and routes, in lockstep

                cache = _slice_cache(cache, active, len(active))
                rows = rows[active]
                individuals = individuals[active]
                active = active[active]

            if step > 0:
                # x-transformers expects the full prefix with its cache (it
                # trims internally, so the lora hooks only see the new token);
                # huggingface expects only the last token - pass
                # cache_last_token for that

                if cache_enabled and cache_last_token:
                    x = seqs[rows, length - 1][:, None]
                else:
                    x = seqs[rows, :length]

                cache_inject = {cache_kwarg: cache} if cache_enabled else dict()
                logits, cache = decode(x, **cache_inject, **forward_kwargs, **cache_kwargs)

            tokens, logits_last = next_tokens(logits)
            seqs[rows, length] = tokens
            active &= ~done(tokens, logits_last, step)
            length += 1
            step += 1

    return seqs
