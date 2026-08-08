import os
import time
import torch
import torch.distributed as dist

from x_transformers import TransformerWrapper, Decoder
from populora import Population, evaluate_population_distributed

def test_distributed_evolution():
    if 'RANK' in os.environ and not dist.is_initialized():
        dist.init_process_group(backend = 'gloo')

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

    def dummy_env_eval(population, idx):
        time.sleep(0.01)
        return population(x, individual = idx).abs().mean() + idx * 0.5

    for _ in range(2):
        fitnesses = evaluate_population_distributed(pop, dummy_env_eval)
        assert fitnesses.shape == (6,) and (fitnesses > 0).all()

        survivors, culled, elites = pop.select('deterministic', fitnesses, survive_frac = 0.5, elite_frac = 0.33)
        parents = pop.select_parents('tournament', fitnesses, num_children = len(culled), culled = culled)

        pop.crossover_('average', parents, culled)
        pop.mutate_('full_gaussian', individuals = culled)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    test_distributed_evolution()
