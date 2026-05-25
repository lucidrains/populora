import torch
from torch.nn import Module, ModuleList

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# evolution

# selection

def select(population):
    raise NotImplementedError

# mutation

def mutation(population):
    raise NotImplementedError

# crossover

def crossover(*parents):
    raise NotImplementedError

# main class

class PopuLoRA(Module):
    def __init__(
        self,
        model: Module
    ):
        super().__init__()

        self.model = model
