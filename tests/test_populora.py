import pytest
import torch
from torch import allclose

from x_transformers import TransformerWrapper, Decoder
from populora import Population, Populations, PopuLoRA, register_mutation

# helper

def get_model():
    return TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 16,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

# tests

def test_population():
    model = get_model()

    pop = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
            'attn_layers.layers.0.1.to_k',
            'attn_layers.layers.0.1.to_v'
        ]
    )

    x = torch.randint(0, 1000, (1, 16))

    # forward passes

    out_orig = pop(x)
    out_0 = pop(x, individual = 0)
    out_1 = pop(x, individual = 1)

    assert out_orig.shape == (1, 16, 1000)
    assert out_0.shape == (1, 16, 1000)

    assert not allclose(out_orig, out_0)
    assert not allclose(out_0, out_1)

    out_all = pop(x, all_individuals = True)
    assert out_all.shape == (4, 16, 1000)

    out_subset = pop(x, individuals = [0, 1])
    assert out_subset.shape == (2, 16, 1000)

def test_populations():
    model = get_model()

    pops = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    out_solver = pops(x, individual = 0, pop_name = 'solver')
    out_conj = pops(x, individual = 0, pop_name = 'conjecturer')

    assert out_solver.shape == (1, 16, 1000)
    assert not allclose(out_solver, out_conj)

def test_populora():
    populora = PopuLoRA(
        model = get_model(),
        num_teachers = 2,
        num_students = 2,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    out_t = populora(x, individual = 0, pop_name = 'teacher')
    out_s = populora(x, individual = 0, pop_name = 'student')

    assert out_t.shape == (1, 16, 1000)
    assert not allclose(out_t, out_s)

def test_mutations():
    pop = Population(
        get_model(),
        pop_size = 4,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    def clone_weights():
        return {k: v.clone() for k, v in pop.weight_down.items()}

    # M1 on individual 0

    before = clone_weights()
    pop.mutate('svd_structured', individual = 0)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][0], before[k][0])
        assert allclose(pop.weight_down[k][1:], before[k][1:])

    # M2 on subset

    before = clone_weights()
    pop.mutate('layer_selective_gaussian', individuals = [1, 2])

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][1:3], before[k][1:3])
        assert allclose(pop.weight_down[k][0], before[k][0])
        assert allclose(pop.weight_down[k][3], before[k][3])

    # M3 on all

    before = clone_weights()
    pop.mutate('component_masking', all_individuals = True)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k], before[k])

    # M4

    before = clone_weights()
    pop.mutate('full_gaussian', individual = 1)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][1], before[k][1])

    # M5

    before = clone_weights()
    pop.mutate('neftune_style', individual = 2)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][2], before[k][2])

    # unknown mutation

    with pytest.raises(ValueError, match = 'Unknown mutation type'):
        pop.mutate('nonexistent_mutation', individual = 0)

def test_custom_mutation():
    pop = Population(
        get_model(),
        pop_size = 2,
        low_rank = 2,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    def mutation_random_signs(population, idx, **kwargs):
        for key in population.weight_down.keys():
            w = population.weight_down[key][idx]
            w.add_(torch.randint_like(w, 0, 2) * 2 - 1)

    register_mutation('random_signs', mutation_random_signs)

    before = {k: v.clone() for k, v in pop.weight_down.items()}

    # mutate random individual

    rand_idx = torch.randint(0, 2, (1,)).item()
    other_idx = 1 - rand_idx

    pop.mutate('random_signs', individual = rand_idx)

    for k in pop.weight_down.keys():
        assert not allclose(pop.weight_down[k][rand_idx], before[k][rand_idx])
        assert allclose(pop.weight_down[k][other_idx], before[k][other_idx])
