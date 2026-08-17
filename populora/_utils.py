from __future__ import annotations

import torch
from torch import is_tensor, tensor

from torch_einops_utils import maybe

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

def cast_tensor(val, device = None):
    return val if is_tensor(val) else tensor(val, device = device)

def resolve_dtype(dtype):
    # accept a torch.dtype or a string name like 'float16' / 'bfloat16'

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        return getattr(torch, dtype)

    raise TypeError(f'invalid dtype {dtype}')
