from __future__ import annotations

import os
import atexit
import random
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor, is_tensor, tensor

from populora._utils import cast_tensor, default, exists

if TYPE_CHECKING:
    from populora.population import Population

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

# collectives

def _collective_device():
    # tensors must live on this device to take part in the process group's collectives.
    # the current device is only set inside the `distributed()` context - use each
    # rank's local device directly so collectives work under plain torchrun too

    return distributed_device() if dist.get_backend() == 'nccl' else torch.device('cpu')

@torch.no_grad()
def _tensor_collective(tensor, fn, copy_back = True):
    # run a collective `fn` on a tensor, moving it to the backend's device when
    # needed and copying the result back when it was moved and `copy_back`

    device = _collective_device()

    if tensor.device.type == device.type:
        fn(tensor)
    else:
        moved = tensor.to(device)
        fn(moved)

        if copy_back:
            tensor.copy_(moved)

    return tensor

def _broadcast_tensor(tensor, src = 0):
    return _tensor_collective(tensor, lambda t: dist.broadcast(t, src = src), copy_back = distributed_rank() != src)

# seeds

def sync_seed(seed = 0, src = 0):
    _ensure_process_group()

    if is_distributed():
        seed_tensor = tensor(seed, dtype = torch.long)
        _broadcast_tensor(seed_tensor, src = src)
        seed = int(seed_tensor.item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

# broadcast helpers

def broadcast_object(value, src = 0):
    # broadcast a tensor, or any picklable value. tensors travel raw - far cheaper
    # than pickling - announced by their (shape, dtype) on the object broadcast;
    # anything else (dicts, lists, small payloads, ...) is pickled in the same
    # call. `value` is only used on the `src` rank

    if not is_distributed():
        return value

    is_src = distributed_rank() == src

    meta = [None, None]

    if is_src:
        if is_tensor(value):
            value = value.contiguous()  # the raw broadcast needs contiguous storage
            meta = [value.dtype, value.shape]
        else:
            meta = ['pickle', value]

    dist.broadcast_object_list(meta, src = src)
    tag, payload = meta

    if tag == 'pickle':
        return payload

    tensor = value if is_src else torch.empty(payload, dtype = tag, device = _collective_device())
    return _broadcast_tensor(tensor, src = src)

# sync population

@torch.no_grad()
def sync_population(population: Population, src = 0, sync_base_model = False):
    if not is_distributed():
        return population

    # only the lora weights are broadcast by default - the base model is shared and
    # identical on every rank by construction. opt in to sync it too

    lora_params = [*population.weight_down.values(), *population.weight_up.values()]
    params = [*lora_params, *population.model.state_dict().values()] if sync_base_model else lora_params

    for param in params:
        if is_tensor(param):
            _broadcast_tensor(param, src = src)

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

    seed_tensor = tensor(seed, dtype = torch.long)
    _tensor_collective(seed_tensor, lambda t: dist.all_reduce(t, op = dist.ReduceOp.SUM))
    return int(seed_tensor.item())

_synced_populations = WeakKeyDictionary()

def evaluate_population_distributed(
    population: Population,
    eval_fn,
    batch_eval = False,
    device: torch.device | str | None = None,
    contiguous = False,
    preserve_rng_state = True,
    shared_seed = True,
    sync_base_model = False
) -> Tensor:
    pop_size = population.pop_size
    device = default(device, population.device)

    # auto-sync the eval seed across ranks

    if shared_seed and exists(population._eval_seed):
        population._eval_seed = sync_seed_sum(population._eval_seed + 1)

    if is_distributed():
        # keyed by the population itself (not id, which is reused after gc), and
        # tracking the base model flag so a later opt-in is not silently dropped

        synced_base_model = _synced_populations.get(population)

        if synced_base_model is None or (sync_base_model and not synced_base_model):
            sync_population(population, sync_base_model = sync_base_model)
            _synced_populations[population] = sync_base_model

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
        _tensor_collective(fitnesses, lambda t: dist.all_reduce(t, op = dist.ReduceOp.SUM))

    return fitnesses
