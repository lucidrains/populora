import time

import torch
from x_transformers import Decoder, TransformerWrapper

from populora import Population, is_main_rank

def test_distributed_evolution():
    model = TransformerWrapper(
        num_tokens = 1000,
        max_seq_len = 16,
        attn_layers = Decoder(dim = 64, depth = 1, heads = 1)
    )

    pop = Population(
        model,
        pop_size = 6,
        low_rank = 4,
        lora_targets = ['attn_layers.layers.0.1.to_q']
    )

    x = torch.randint(0, 1000, (1, 16))

    def eval_env(population, idx):
        time.sleep(0.01)
        return population(x, individual = idx).abs().mean()

    for gen in range(2):
        fitnesses = pop.evaluate_distributed(eval_env)

        if is_main_rank():
            print(f'gen {gen:02d} | best: {fitnesses.max():.3f} | mean: {fitnesses.mean():.3f}')

        pop.evolve_(fitnesses)

    assert len(fitnesses) == 6

if __name__ == '__main__':
    test_distributed_evolution()
    print('distributed evolution test passed')
