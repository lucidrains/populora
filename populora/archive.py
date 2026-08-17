# cross-generation memory of individuals

from __future__ import annotations

from collections import namedtuple

import torch
from torch import arange, no_grad

from einops import repeat

from populora._utils import exists

Entry = namedtuple('Entry', ['generation', 'weight_down', 'weight_up'])

class HallOfFame:
    def __init__(self, capacity = None):
        self.capacity = capacity
        self.entries = []

        self._replay = None
        self._replay_population = None

    def __len__(self):
        return len(self.entries)

    @no_grad()
    def add(self, population, individual, generation = None):
        # archive a copy of the individual's weights

        _, (weight_down, weight_up) = population.individual_weights(individual)

        self.entries.append(Entry(
            generation = generation,
            weight_down = {key: w.clone() for key, w in weight_down.items()},
            weight_up = {key: w.clone() for key, w in weight_up.items()}
        ))

        if exists(self.capacity) and len(self.entries) > self.capacity:
            self.entries.pop(0)

        return len(self.entries) - 1

    @no_grad()
    def add_champion(self, population, fitnesses, generation = None):
        # archive the best individual - call once per generation to build the
        # champion ladder the master tournament needs

        return self.add(population, fitnesses.argmax(), generation = generation)

    def sample(self, num, mode = 'uniform', device = None, generator = None):
        # entry indices to evaluate against - `generator` keeps the draw off the global rng

        size = len(self.entries)

        if mode == 'all' or num >= size:
            indices = arange(size)
        elif mode == 'latest':
            indices = arange(size - num, size)
        elif mode == 'uniform':
            indices = torch.rand(size, generator = generator).argsort()[:num]
        else:
            raise ValueError(f'unknown sampling mode {mode}')

        return indices.to(device) if exists(device) else indices

    @no_grad()
    def replay(self, population, indices = None):
        # load archived individuals into the first slots of a replay population, returning the slot indices

        if not exists(indices):
            indices = range(len(self.entries))

        assert len(indices) <= population.pop_size, f'replay population of size {population.pop_size} is smaller than {len(indices)} archived opponents'

        for slot, i in enumerate(indices):
            entry = self.entries[i]
            population.load_individual(
                dict(weight_down = entry.weight_down, weight_up = entry.weight_up),
                individual = slot
            )

        return arange(len(indices), device = population.device)

    @no_grad()
    def probe(self, population, x, num, mode = 'uniform', generator = None):
        # route x through `num` sampled archived champions, returning (k, *x.shape[:-1], -1)
        # outputs - None when nothing has been archived yet. the replay population
        # shares the population's frozen base, so only the adapters are extra memory

        if len(self.entries) == 0:
            return None

        indices = self.sample(min(num, len(self.entries)), mode = mode, generator = generator)
        k = len(indices)

        if not exists(self._replay) or self._replay_population is not population or self._replay.pop_size < k:
            self._replay = population.__class__(
                population.model,
                pop_size = k,
                low_rank = population.low_rank,
                lora_targets = population.lora_targets,
                device = population.device
            )
            self._replay_population = population

        slots = self.replay(self._replay, indices)

        repeated = repeat(x, '... -> k ...', k = k)
        outputs = self._replay(repeated, individuals = slots)

        return outputs.reshape(k, *x.shape[:-1], -1)
