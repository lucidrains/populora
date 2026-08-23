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

def rescale_from_range_to_range(value, from_range = (-1., 1.), to_range = (0., 1.)):
    # affine map of a value (or tensor, or per-dimension range pair) from one
    # range to another, e.g. policy outputs on (-1, 1) to an env action range.
    # a pure rescale - no clamping

    from_lo, from_hi = maybe_cast_tuple(from_range, 2)
    to_lo, to_hi = maybe_cast_tuple(to_range, 2)

    from_lo, from_hi = torch.as_tensor(from_lo), torch.as_tensor(from_hi)
    to_lo, to_hi = torch.as_tensor(to_lo), torch.as_tensor(to_hi)

    if is_tensor(value):
        device = value.device
        from_lo, from_hi = from_lo.to(device), from_hi.to(device)
        to_lo, to_hi = to_lo.to(device), to_hi.to(device)

    ratio = (to_hi - to_lo) / (from_hi - from_lo)
    return to_lo + (value - from_lo) * ratio
