import torch
from x_transformers import TransformerWrapper, Decoder

from populora import PopuLoRA

def test_populora():
    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 128,
        attn_layers = Decoder(
            dim = 256,
            depth = 2,
            heads = 4
        )
    )

    populora_model = PopuLoRA(
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

    # all individuals at once should match individual forwards

    out_all = populora_model(x, all_individuals = True)
    assert out_all.shape == (4, 128, 1000)

    out_all_chunked = out_all.chunk(4, dim = 0)

    assert torch.allclose(out_all_chunked[0], out_indiv_0, atol = 1e-6)
    assert torch.allclose(out_all_chunked[1], out_indiv_1, atol = 1e-6)

    # subset of individuals should match subset of all_individuals

    out_subset = populora_model(x, individuals = [0, 1])
    assert out_subset.shape == (2, 128, 1000)

    out_subset_chunked = out_subset.chunk(2, dim = 0)

    assert torch.allclose(out_subset_chunked[0], out_indiv_0, atol = 1e-6)
    assert torch.allclose(out_subset_chunked[1], out_indiv_1, atol = 1e-6)
