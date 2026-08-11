from __future__ import annotations

import os
import atexit
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from functools import wraps

import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor, tensor
from torch.nn import Module

from populora.populora import cast_tensor, default, exists, is_tensor

# process group

_initialized_by_us = False

def _ensure_process_group(backend = None, init_method = 'env://'):
    global _initialized_by_us

    if _initialized_by_us or dist.is_initialized() or 'RANK' not in os.environ or not dist.is_available():
        return

    import socket
    if os.environ.get('MASTER_ADDR') in (socket.gethostname(), socket.getfqdn()):
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    dist.init_process_group(backend = default(backend, default_backend()), init_method = init_method)
    _initialized_by_us = True

    sync_seed(0)
    atexit.register(_finalize_process_group)

def _finalize_process_group():
    global _initialized_by_us

    if _initialized_by_us and dist.is_initialized():
        dist.destroy_process_group()

    _initialized_by_us = False

# identity

def is_distributed():
    _ensure_process_group()
    return dist.is_available() and dist.is_initialized()

def distributed_rank():
    return dist.get_rank() if is_distributed() else 0

def distributed_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main_rank():
    return distributed_rank() == 0

def main_rank_only(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        if is_main_rank():
            return fn(*args, **kwargs)

    return inner

DistributedInfo = namedtuple('DistributedInfo', ['rank', 'world_size', 'device', 'is_distributed', 'is_main_rank'])

def distributed_device(device = None):
    if exists(device):
        return torch.device(device)

    if is_distributed() and dist.get_backend() == 'nccl':
        local_rank = int(default(os.environ.get('LOCAL_RANK'), 0))
        return torch.device(f'cuda:{local_rank}')

    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')

def default_backend(device = None):
    if exists(device):
        return 'nccl' if torch.device(device).type == 'cuda' else 'gloo'
    return 'nccl' if torch.cuda.is_available() else 'gloo'

# seeds

def sync_seed(seed = 0, src = 0):
    _ensure_process_group()

    if is_distributed():
        device = distributed_device()
        use_cuda = device.type == 'cuda' and dist.get_backend() == 'nccl'
        seed_tensor = tensor(seed, dtype = torch.long, device = device if use_cuda else 'cpu')
        dist.broadcast(seed_tensor, src = src)
        seed = int(seed_tensor.item())

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

# context manager

@contextmanager
def distributed(
    seed = 0,
    backend: str | None = None,
    device: torch.device | str | None = None,
    init_method = 'env://'
):
    _ensure_process_group(backend, init_method)
    sync_seed(seed)
    device = distributed_device(device)

    if is_distributed() and device.type == 'cuda':
        torch.cuda.set_device(device)

    yield DistributedInfo(distributed_rank(), distributed_world_size(), device, is_distributed(), is_main_rank())

    if is_distributed():
        dist.barrier()

# rng preservation

@contextmanager
def preserve_rng():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()

    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if exists(cuda_states):
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)

# sync population

@torch.no_grad()
def sync_population(population: Module, src = 0):
    if not is_distributed():
        return population

    backend = dist.get_backend()

    for tensor in population.state_dict().values():
        if not is_tensor(tensor):
            continue

        if backend == 'nccl' and not tensor.is_cuda:
            tensor_cuda = tensor.cuda()
            dist.broadcast(tensor_cuda, src = src)
            tensor.copy_(tensor_cuda.cpu())
        elif tensor.device.type == 'cpu':
            dist.broadcast(tensor, src = src)
        else:
            tensor_cpu = tensor.cpu()
            dist.broadcast(tensor_cpu, src = src)
            tensor.copy_(tensor_cpu)

    dist.barrier()
    return population

# partition indices

def partition_indices(num_items: int, contiguous = False):
    rank, world_size = distributed_rank(), distributed_world_size()

    if world_size == 1:
        return list(range(num_items))

    if not contiguous:
        return list(range(rank, num_items, world_size))

    per_rank = (num_items + world_size - 1) // world_size
    start = rank * per_rank
    return list(range(start, min(start + per_rank, num_items)))

# distributed evaluation

def sync_seed_sum(seed):
    # all reduce the seed across ranks

    if not is_distributed():
        return seed

    device = distributed_device()
    use_cuda = device.type == 'cuda' and dist.get_backend() == 'nccl'
    seed_tensor = tensor(seed, dtype = torch.long, device = device if use_cuda else 'cpu')
    dist.all_reduce(seed_tensor, op = dist.ReduceOp.SUM)
    return int(seed_tensor.item())

_synced_populations = set()

def evaluate_population_distributed(
    population,
    eval_fn,
    batch_eval = False,
    device: torch.device | str | None = None,
    contiguous = False,
    preserve_rng_state = True,
    shared_seed = True
) -> Tensor:
    pop_size = population.pop_size
    device = default(device, population.device)

    # auto-sync the eval seed across ranks

    if shared_seed and exists(population._eval_seed):
        population._eval_seed = sync_seed_sum(population._eval_seed + 1)

    if is_distributed() and id(population) not in _synced_populations:
        sync_population(population)
        _synced_populations.add(id(population))

    assigned_indices = partition_indices(pop_size, contiguous = contiguous)

    fitnesses = torch.zeros(pop_size, device = device, dtype = torch.float32)

    rng_guard = preserve_rng() if preserve_rng_state else nullcontext()

    with rng_guard:
        if batch_eval:
            if len(assigned_indices) > 0:
                res = eval_fn(population, assigned_indices)
                res = cast_tensor(res, device = device).to(dtype = torch.float32)
                assert res.shape[0] == len(assigned_indices), 'batch eval fn must return one fitness per assigned index'
                fitnesses[assigned_indices] = res
        else:
            for idx in assigned_indices:
                fitnesses[idx] = cast_tensor(eval_fn(population, idx), device = device).to(dtype = torch.float32)

    if is_distributed():
        if dist.get_backend() == 'nccl' and not fitnesses.is_cuda:
            fitnesses_cuda = fitnesses.cuda()
            dist.all_reduce(fitnesses_cuda, op = dist.ReduceOp.SUM)
            fitnesses.copy_(fitnesses_cuda.cpu())
        elif fitnesses.device.type == 'cpu':
            dist.all_reduce(fitnesses, op = dist.ReduceOp.SUM)
        else:
            fitnesses_cpu = fitnesses.cpu()
            dist.all_reduce(fitnesses_cpu, op = dist.ReduceOp.SUM)
            fitnesses.copy_(fitnesses_cpu)

    return fitnesses
