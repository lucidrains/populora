import pytest

import torch
from x_transformers import TransformerWrapper, Decoder

from populora import Population, Populations, PopuLoRA

def test_population():
    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 128,
        attn_layers = Decoder(
            dim = 256,
            depth = 2,
            heads = 4
        )
    )

    populora_model = Population(
        model,
        pop_size = 4,
        low_rank = 4,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
            'attn_layers.layers.0.1.to_k',
            'attn_layers.layers.0.1.to_v',
            'attn_layers.layers.0.1.to_out',
        ]
    )

    x = torch.randint(0, 1000, (1, 128))

    out_orig = populora_model(x)

    out_indiv_0 = populora_model(x, individual = 0)
    out_indiv_1 = populora_model(x, individual = 1)

    assert out_orig.shape == (1, 128, 1000)
    assert out_indiv_0.shape == (1, 128, 1000)
    assert out_indiv_1.shape == (1, 128, 1000)

    assert not torch.allclose(out_orig, out_indiv_0)
    assert not torch.allclose(out_indiv_0, out_indiv_1)

    out_all = populora_model(x, all_individuals = True)
    assert out_all.shape == (4, 128, 1000)

    # test batched input

    x_batch_4 = x.repeat(4, 1)
    out_all_from_batch = populora_model(x_batch_4, all_individuals = True)
    assert torch.allclose(out_all, out_all_from_batch)

    x_batch_3 = x.repeat(3, 1)
    with pytest.raises(AssertionError):
        populora_model(x_batch_3, all_individuals = True)

    out_all_chunked = out_all.chunk(4, dim = 0)

    assert torch.allclose(out_all_chunked[0], out_indiv_0, atol = 1e-6)
    assert torch.allclose(out_all_chunked[1], out_indiv_1, atol = 1e-6)

    out_subset = populora_model(x, individuals = [0, 1])
    assert out_subset.shape == (2, 128, 1000)

    # test batched input for subset

    x_batch_2 = x.repeat(2, 1)
    out_subset_from_batch = populora_model(x_batch_2, individuals = [0, 1])
    assert torch.allclose(out_subset, out_subset_from_batch)

    with pytest.raises(AssertionError):
        populora_model(x_batch_3, individuals = [0, 1])

    out_subset_chunked = out_subset.chunk(2, dim = 0)

    assert torch.allclose(out_subset_chunked[0], out_indiv_0, atol = 1e-6)
    assert torch.allclose(out_subset_chunked[1], out_indiv_1, atol = 1e-6)

def test_populations():
    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 128,
        attn_layers = Decoder(
            dim = 256,
            depth = 2,
            heads = 4
        )
    )

    populations = Populations(
        model = model,
        pop_sizes = dict(solver = 4, conjecturer = 4),
        low_ranks = 4,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
            'attn_layers.layers.0.1.to_k',
            'attn_layers.layers.0.1.to_v',
            'attn_layers.layers.0.1.to_out',
        ]
    )

    x = torch.randint(0, 1000, (1, 128))

    out_solver = populations(x, individual = 0, pop_name = 'solver')
    out_conjecturer = populations(x, individual = 0, pop_name = 'conjecturer')
    out_solver_1 = populations(x, individual = 1, pop_name = 'solver')

    assert out_solver.shape == (1, 128, 1000)
    assert out_conjecturer.shape == (1, 128, 1000)

    assert not torch.allclose(out_solver, out_conjecturer)
    assert not torch.allclose(out_solver, out_solver_1)

    out_base = populations(x, pop_name = 'solver')
    out_base_direct = model(x)
    assert torch.allclose(out_base, out_base_direct)

def test_separate_models():
    import copy

    model1 = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )
    model2 = copy.deepcopy(model1)

    populations = Populations(
        models = dict(solver = model1, conjecturer = model2),
        pop_sizes = dict(solver = 2, conjecturer = 3),
        low_ranks = 2,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
        ]
    )

    x = torch.randint(0, 100, (1, 16))

    out_solver = populations(x, individual = 0, pop_name = 'solver')
    out_conjecturer = populations(x, individual = 0, pop_name = 'conjecturer')

    assert out_solver.shape == (1, 16, 100)
    assert out_conjecturer.shape == (1, 16, 100)

def test_guider():
    model = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )

    populations = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2, guider = 2),
        low_ranks = 2,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
        ]
    )

    x = torch.randint(0, 100, (1, 16))
    out_guider = populations(x, individual = 0, pop_name = 'guider')
    assert out_guider.shape == (1, 16, 100)

def test_missing_guider():
    model = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )

    populations = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 2,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
        ]
    )

    x = torch.randint(0, 100, (1, 16))
    with pytest.raises(AssertionError, match = 'unknown population guider'):
        populations(x, pop_name = 'guider')

def test_invalid_role():
    model = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )

    populations = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 2,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
        ]
    )

    x = torch.randint(0, 100, (1, 16))
    with pytest.raises(AssertionError, match = 'unknown population unknown'):
        populations(x, pop_name = 'unknown')

def test_populora_subclass():
    model = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )

    populora = PopuLoRA(
        model = model,
        num_teachers = 2,
        num_students = 2,
        low_rank = 2,
        lora_targets = [
            'attn_layers.layers.0.1.to_q',
        ]
    )

    x = torch.randint(0, 100, (1, 16))
    out_teacher = populora(x, individual = 0, pop_name = 'teacher')
    out_student = populora(x, individual = 0, pop_name = 'student')

    assert out_teacher.shape == (1, 16, 100)
    assert out_student.shape == (1, 16, 100)
    assert not torch.allclose(out_teacher, out_student)

def test_lora_targets_formats():
    model = TransformerWrapper(
        num_tokens = 100,
        max_seq_len = 16,
        attn_layers = Decoder(
            dim = 64,
            depth = 1,
            heads = 1
        )
    )

    pop_tuple = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 2,
        lora_targets = ('attn_layers.layers.0.1.to_q', 'attn_layers.layers.0.1.to_k')
    )
    assert isinstance(pop_tuple.populations['solver'].w_down['attn_layers_layers_0_1_to_q'], torch.Tensor)

    pop_dict_list = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 2,
        lora_targets = dict(
            solver = ['attn_layers.layers.0.1.to_q'],
            conjecturer = ['attn_layers.layers.0.1.to_v']
        )
    )
    assert 'attn_layers_layers_0_1_to_q' in pop_dict_list.populations['solver'].w_down
    assert 'attn_layers_layers_0_1_to_v' not in pop_dict_list.populations['solver'].w_down
    assert 'attn_layers_layers_0_1_to_v' in pop_dict_list.populations['conjecturer'].w_down
    assert 'attn_layers_layers_0_1_to_q' not in pop_dict_list.populations['conjecturer'].w_down

    pop_dict_tuple = Populations(
        model = model,
        pop_sizes = dict(solver = 2, conjecturer = 2),
        low_ranks = 2,
        lora_targets = dict(
            solver = ('attn_layers.layers.0.1.to_q',),
            conjecturer = ('attn_layers.layers.0.1.to_k',)
        )
    )
    assert 'attn_layers_layers_0_1_to_q' in pop_dict_tuple.populations['solver'].w_down
    assert 'attn_layers_layers_0_1_to_k' in pop_dict_tuple.populations['conjecturer'].w_down

    populora_dict = PopuLoRA(
        model = model,
        num_teachers = 2,
        num_students = 2,
        low_rank = 2,
        lora_targets = dict(
            teacher = ['attn_layers.layers.0.1.to_q'],
            student = ['attn_layers.layers.0.1.to_v']
        )
    )

    assert 'attn_layers_layers_0_1_to_q' in populora_dict.populations.populations['teacher'].w_down
    assert 'attn_layers_layers_0_1_to_v' in populora_dict.populations.populations['student'].w_down
