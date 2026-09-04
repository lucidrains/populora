from __future__ import annotations

from pathlib import Path

import torch
from torch import is_tensor, tensor

from torch_einops_utils import cast_tensor, maybe

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def has_(v):
    return exists(v) and (v.any().item() if is_tensor(v) else v > 0.)

def first(arr):
    return arr[0] if len(arr) > 0 else None

def extract_dict(v, k):
    return v[k] if isinstance(v, dict) else v

def cast_tuple(val, length = 1):
    return val if isinstance(val, (tuple, list)) else ((val,) * length)

maybe_cast_tuple = maybe(cast_tuple)

def torch_save(pkg, path: str | Path):
    # torch.save with directory creation - one canonical save path
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(pkg, path)

def maybe_progress(iterable, enabled = False, desc = ''):
    if not enabled:
        return iterable

    try:
        from tqdm import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, desc = desc)

def resolve_dtype(dtype):
    # accept a torch.dtype or a string name like 'float16' / 'bfloat16'

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        assert isinstance(resolved, torch.dtype), f'invalid dtype {dtype}'
        return resolved

    raise TypeError(f'invalid dtype {dtype}')

def rescale_from_range_to_range(val, from_range = (-1., 1.), to_range = (0., 1.)):
    (from_min, from_max), (to_min, to_max) = from_range, to_range

    if is_tensor(val):
        dd = dict(device = val.device, dtype = val.dtype)
        from_min, from_max, to_min, to_max = [torch.as_tensor(t, **dd) for t in (from_min, from_max, to_min, to_max)]

    return (val - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
