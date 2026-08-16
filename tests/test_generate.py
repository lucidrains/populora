import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

from x_transformers import TransformerWrapper, Decoder

from populora import Population, generate
from populora.generate import _slice_cache

# helper

def get_model(num_tokens = 32, max_seq_len = 32):
    return TransformerWrapper(
        num_tokens = num_tokens,
        max_seq_len = max_seq_len,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

def get_pop(num_tokens = 32, max_seq_len = 32, pop_size = 4):
    model = get_model(num_tokens, max_seq_len)

    return Population(
        model,
        pop_size = pop_size,
        low_rank = 4,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
            'attn_layers.layers.0.1.to_k',
            'attn_layers.layers.0.1.to_v'
        ]
    )

# tests

def test_generate_matches_manual_decode():
    # generate's full decode loop (with cache) must reproduce a manual
    # per-step decode loop with the same model calls

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    gen = generate(pop, ids, individual = 1, max_len = 10, cache_kwargs = dict(return_intermediates = True))

    seqs = ids.clone()
    cache = None

    for _ in range(4):
        logits, cache = pop(seqs, individual = 1, cache = cache, return_intermediates = True, eval_and_no_grad = True)
        seqs = torch.cat((seqs, logits[:, -1].argmax(dim = -1)[:, None]), dim = 1)

    assert torch.equal(gen, seqs)

def test_generate_cache_equals_no_cache():
    # with or without a kv cache, the generations are identical

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    for individual in range(4):
        cached = generate(pop, ids, individual = individual, max_len = 10, cache_kwargs = dict(return_intermediates = True))
        uncached = generate(pop, ids, individual = individual, max_len = 10)

        assert torch.equal(cached, uncached)

def test_generate_all_individuals_equals_per_individual():
    # decoding the whole population at once matches decoding each individual
    # separately, for both the routed and cached paths

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    batched = generate(pop, ids, all_individuals = True, max_len = 10)

    for i in range(4):
        single = generate(pop, ids, individual = i, max_len = 10)
        assert torch.equal(batched[i], single[i])

    batched_cached = generate(pop, ids, all_individuals = True, max_len = 10, cache_kwargs = dict(return_intermediates = True))

    for i in range(4):
        single_cached = generate(pop, ids, individual = i, max_len = 10, cache_kwargs = dict(return_intermediates = True))
        assert torch.equal(batched_cached[i], single_cached[i])

def test_generate_per_sample_routing():
    # explicit per-sample individual ids, one per prompt

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))
    individuals = torch.tensor([3, 1, 1, 0])

    routed = generate(pop, ids, individuals = individuals, max_len = 10, cache_kwargs = dict(return_intermediates = True))

    for i, individual in enumerate(individuals.tolist()):
        single = generate(pop, ids[i:i + 1], individual = individual, max_len = 10, cache_kwargs = dict(return_intermediates = True))
        assert torch.equal(routed[i], single[0])

def test_generate_eos_early_stop():
    # samples that emit eos stop immediately, the rest keep going, and the
    # finished samples' cache rows are dropped without corrupting the others

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))
    eos = 31

    # first decode step: eos for the first two samples, argmax for the rest

    counter = dict(count = 0)

    def sample_fn(logits):
        if counter['count'] == 0:
            tokens = logits.argmax(dim = -1)
            tokens[:2] = eos
        else:
            tokens = logits.argmax(dim = -1)

        counter['count'] += 1
        return tokens

    gen = generate(pop, ids, all_individuals = True, max_len = 10, eos_token = eos, sample_fn = sample_fn, cache_kwargs = dict(return_intermediates = True))

    assert (gen[:2, 6:] == eos).all()

    # the samples that kept going match a reference run with the same eos rule

    reference = generate(pop, ids[2:], individuals = [2, 3], max_len = 10, eos_token = eos, cache_kwargs = dict(return_intermediates = True))

    assert torch.equal(gen[2:], reference)

def test_generate_stop_fn_differential_compaction():
    # samples stopping at different steps - each sample's prefix must match a
    # fresh single-sample run with the same stopping rule

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    def stop_fn(tokens, logits, step):
        return tokens % 2 == 0

    gen = generate(pop, ids, all_individuals = True, max_len = 12, stop_fn = stop_fn, cache_kwargs = dict(return_intermediates = True))

    for i in range(4):
        single = generate(pop, ids[i:i + 1], individual = i, max_len = 12, stop_fn = stop_fn, cache_kwargs = dict(return_intermediates = True))
        assert torch.equal(gen[i], single[0])

def test_generate_micro_batch_parity():
    # chunking the prefill must not change the generations

    pop = get_pop()

    ids = torch.randint(0, 32, (8, 6))

    full = generate(pop, ids, all_individuals = True, max_len = 10)
    chunked = generate(pop, ids, all_individuals = True, max_len = 10, micro_batch = 8)

    assert torch.equal(full, chunked)

def test_generate_max_len_prompt_only():
    # nothing to generate - the prompts come back as-is

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    gen = generate(pop, ids, individual = 0, max_len = 6)

    assert torch.equal(gen, ids)

def test_generate_deterministic():
    # repeated calls produce identical generations

    pop = get_pop()

    ids = torch.randint(0, 32, (4, 6))

    a = generate(pop, ids, all_individuals = True, max_len = 10, cache_kwargs = dict(return_intermediates = True))
    b = generate(pop, ids, all_individuals = True, max_len = 10, cache_kwargs = dict(return_intermediates = True))

    assert torch.equal(a, b)

def test_generate_mutated_individual_differs():
    # a mutated individual generates different tokens than before the mutation

    pop = get_pop(pop_size = 2)

    ids = torch.randint(0, 32, (2, 8))

    before = generate(pop, ids, all_individuals = True, max_len = 16)

    pop.mutate_('full_gaussian', individual = 1, epsilon = 1.0)

    after = generate(pop, ids, all_individuals = True, max_len = 16)

    assert not torch.equal(before[1], after[1])

def test_slice_cache():
    # slicing recurses through the container types a cache can take

    cache = (
        torch.arange(12).reshape(4, 3),
        [torch.arange(8).reshape(4, 2), dict(k = torch.arange(4))],
        torch.tensor(3.)  # non-batch scalar, untouched
    )

    mask = torch.tensor([True, False, True, False])

    sliced = _slice_cache(cache, mask, 4)

    assert torch.equal(sliced[0], cache[0][mask])
    assert torch.equal(sliced[1][0], cache[1][0][mask])
    assert torch.equal(sliced[1][1]['k'], torch.tensor([0, 2]))
    assert sliced[2].item() == 3.

def test_forward_micro_batch():
    # the routed forward, chunked or not, gives identical outputs - including
    # a batch that is not a multiple of the chunk size

    pop = get_pop(pop_size = 4)

    x = torch.randint(0, 32, (1, 12))

    full = pop(x, all_individuals = True)
    chunked = pop(x, all_individuals = True, micro_batch = 8)
    chunked_uneven = pop(x, all_individuals = True, micro_batch = 12)

    assert full.shape == (4, 12, 32)
    assert torch.equal(full, chunked)
    assert torch.equal(full, chunked_uneven)

    # micro_batch must be a multiple of the population size

    with pytest.raises(AssertionError):
        pop(x, all_individuals = True, micro_batch = 6)
