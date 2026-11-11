# composable novelty, local competition, and behavioral descriptor primitives

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, arange, cat, is_tensor, no_grad
from torch.nn import LayerNorm, Linear, Module, Sequential, SiLU
from torch.nn.utils import clip_grad_norm_

from populora._utils import default, exists

# distance primitives

def _upper_tri(mat: Tensor) -> Tensor:
    n = mat.shape[0]
    i, j = torch.triu_indices(n, n, offset = 1, device = mat.device)
    return mat[i, j]

def pairwise_distances(
    descriptors: Tensor,
    *,
    normalize: bool = True,
    p: float = 2.0,
    eps: float = 1e-8
) -> Tensor:
    # (p, p) pairwise L_p distance matrix, optionally z-scored per feature dim

    assert descriptors.ndim >= 1, f'descriptors must have at least 1 dim, got {descriptors.shape}'
    desc = rearrange(descriptors.float(), 'n ... -> n (...)')
    n = desc.shape[0]

    if n <= 1:
        return torch.zeros((n, n), dtype = torch.float32, device = desc.device)

    if normalize:
        desc = (desc - desc.mean(dim = 0, keepdim = True)) / desc.std(dim = 0, unbiased = False).clamp(min = eps)

    return torch.cdist(desc, desc, p = p).fill_diagonal_(0.)

def _resolve_distances(
    descriptors: Tensor | None = None,
    distances: Tensor | None = None,
    *,
    normalize: bool = True,
    p: float = 2.0,
    device: torch.device | None = None
) -> Tensor:
    assert exists(descriptors) or exists(distances), 'either descriptors or distances must be provided'

    if exists(distances):
        dist = distances.float()
        assert dist.ndim == 2 and dist.shape[0] == dist.shape[1], f'distances must be square (P, P), got {tuple(dist.shape)}'
        return dist.to(device) if exists(device) else dist

    desc = descriptors.to(device) if exists(device) else descriptors
    return pairwise_distances(desc, normalize = normalize, p = p)

# trajectory sketching

def _parse_traj(item) -> Tensor:
    if isinstance(item, (tuple, list)) and len(item) == 2 and not is_tensor(item):
        a, b = (torch.as_tensor(x, dtype = torch.float32) for x in item)
        a, b = (x.unsqueeze(-1) if x.ndim == 1 else x for x in (a, b))
        t = min(a.shape[0], b.shape[0])
        return cat((a[:t], b[:t]), dim = -1)

    x = torch.as_tensor(item, dtype = torch.float32)
    return x.unsqueeze(-1) if x.ndim == 1 else x

def sketch_trajectories(
    trajectories: Tensor | Sequence[Tensor | np.ndarray | tuple],
    *,
    num_subsamples: int = 8,
    max_horizon: int | None = None,
    include_stats: bool = True,
    include_length: bool = True,
    device: torch.device | str | None = None
) -> Tensor:
    # summarize variable-length rollout trajectories into fixed-size (p, sketch_dim) descriptors

    assert num_subsamples >= 1, f'num_subsamples must be >= 1, got {num_subsamples}'

    if is_tensor(trajectories) and trajectories.ndim == 3:
        inferred_device = default(device, trajectories.device)
        parsed = list(trajectories.unbind(0))
    else:
        inferred_device = device
        parsed = []
        for item in trajectories:
            if not exists(inferred_device) and is_tensor(item):
                inferred_device = item.device
            parsed.append(_parse_traj(item))

    assert len(parsed) > 0, 'trajectories cannot be empty'

    horizon_ref = max(float(default(max_horizon, max(t.shape[0] for t in parsed))), 1.0)
    feat_dim = parsed[0].shape[-1]

    sketches = []
    for traj in parsed:
        t_len = traj.shape[0]
        if t_len == 0:
            sub = torch.zeros(num_subsamples * feat_dim)
            mean_s = std_s = torch.zeros(feat_dim)
        else:
            t_cpu = traj.detach().cpu().float()
            idx = torch.linspace(0, max(t_len - 1, 0), steps = num_subsamples).round().long()
            sub = rearrange(t_cpu[idx], 's d -> (s d)')
            mean_s = t_cpu.mean(dim = 0)
            std_s = t_cpu.std(dim = 0, unbiased = False) if t_len > 1 else torch.zeros_like(mean_s)

        parts = [sub]
        if include_stats:
            parts.extend((mean_s, std_s))
        if include_length:
            parts.append(torch.tensor([float(t_len) / horizon_ref]))

        sketches.append(cat(parts, dim = 0))

    out = torch.stack(sketches, dim = 0)
    return out.to(inferred_device) if exists(inferred_device) else out

# AURORA unsupervised descriptor learner (Cully 2019; Grillotti & Cully 2022)

class AuroraAutoEncoder(Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 6,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        mid = hidden_dim // 2

        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            SiLU(),
            Linear(hidden_dim, mid),
            LayerNorm(mid),
            SiLU(),
            Linear(mid, latent_dim)
        )
        self.decoder = Sequential(
            Linear(latent_dim, mid),
            SiLU(),
            Linear(mid, hidden_dim),
            SiLU(),
            Linear(hidden_dim, input_dim)
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x).tanh()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        return self.decoder(z), z

class Aurora(Module):
    # online behavioral descriptor autoencoder with Welford standardization,
    # circular replay buffer, and VICReg variance/covariance anti-collapse terms

    def __init__(
        self,
        input_dim: int | None = None,
        latent_dim: int = 6,
        *,
        hidden_dim: int = 64,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 512,
        batch_size: int = 64,
        update_steps: int = 6,
        var_weight: float = 0.25,
        cov_weight: float = 0.10,
        num_subsamples: int = 8,
        max_horizon: int | None = None,
        device: torch.device | str | None = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.num_subsamples = num_subsamples
        self.max_horizon = max_horizon
        self._device = torch.device(default(device, 'cpu'))

        self.ae: AuroraAutoEncoder | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.buffer: Tensor | None = None
        self.running_mean: Tensor | None = None
        self.running_M2: Tensor | None = None

        self.buf_ptr = 0
        self.buf_size = 0
        self.count = 0
        self.loss_history: list[float] = []

        if exists(input_dim):
            self._init_modules(input_dim)

    def _init_modules(self, input_dim: int):
        self.input_dim = int(input_dim)
        self.ae = AuroraAutoEncoder(self.input_dim, self.latent_dim, self.hidden_dim).to(self._device)
        self.opt = torch.optim.AdamW(self.ae.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        dd = dict(dtype = torch.float32, device = self._device)
        self.buffer = torch.zeros((self.buffer_capacity, self.input_dim), **dd)
        self.running_mean = torch.zeros(self.input_dim, **dd)
        self.running_M2 = torch.ones(self.input_dim, **dd)

    def _prepare_sketches(self, inputs: Tensor | Sequence) -> tuple[Tensor, torch.device]:
        if is_tensor(inputs) and inputs.ndim == 2:
            out_device = inputs.device
            sketches = inputs.detach().to(self._device, dtype = torch.float32)
        else:
            sketches = sketch_trajectories(
                inputs,
                num_subsamples = self.num_subsamples,
                max_horizon = self.max_horizon,
                device = self._device
            )
            out_device = sketches.device

        if not exists(self.ae):
            self._init_modules(sketches.shape[-1])

        assert sketches.shape[-1] == self.input_dim, f'Aurora expected input_dim={self.input_dim}, got {sketches.shape[-1]}'
        return sketches, out_device

    @no_grad()
    def _update_stats_and_buffer(self, sketches: Tensor):
        x = sketches.detach().to(self._device, dtype = torch.float32)
        b = x.shape[0]
        if b == 0:
            return

        batch_mean = x.mean(dim = 0)
        batch_var = x.var(dim = 0, unbiased = False)

        if self.count == 0:
            self.running_mean.copy_(batch_mean)
            self.running_M2.copy_(batch_var.clamp(min = 1e-4) * b)
            self.count = b
        else:
            delta = batch_mean - self.running_mean
            tot = self.count + b
            self.running_mean.add_(delta * (b / tot))
            self.running_M2.add_(batch_var * b + (delta ** 2) * (self.count * b / tot))
            self.count = tot

        take = min(b, self.buffer_capacity)
        slots = (arange(take, device = self._device) + self.buf_ptr) % self.buffer_capacity
        self.buffer[slots] = x[-take:]
        self.buf_ptr = (self.buf_ptr + take) % self.buffer_capacity
        self.buf_size = min(self.buf_size + take, self.buffer_capacity)

    def _normalize(self, x: Tensor) -> Tensor:
        std = (self.running_M2 / max(self.count, 1)).clamp(min = 1e-4).sqrt()
        return ((x - self.running_mean) / std).clamp(-5., 5.)

    def fit(self, inputs: Tensor | Sequence, *, steps: int | None = None) -> float:
        sketches, _ = self._prepare_sketches(inputs)
        self._update_stats_and_buffer(sketches)

        num_steps = default(steps, self.update_steps)
        if self.buf_size < 4 or num_steps <= 0:
            return 0.0

        self.ae.train()
        total_loss = 0.0
        eye = torch.eye(self.latent_dim, dtype = torch.bool, device = self._device)

        with torch.enable_grad():
            for _ in range(num_steps):
                b_sz = min(self.batch_size, self.buf_size)
                batch = self._normalize(self.buffer[torch.randint(0, self.buf_size, (b_sz,), device = self._device)])

                recon, z = self.ae(batch)
                recon_loss = F.smooth_l1_loss(recon, batch)
                var_loss = F.relu(0.35 - (z.var(dim = 0, unbiased = False) + 1e-4).sqrt()).mean()

                z_c = z - z.mean(dim = 0, keepdim = True)
                cov = (z_c.T @ z_c) / max(b_sz - 1, 1)
                cov_loss = cov.masked_fill(eye, 0.).pow(2).mean()

                loss = recon_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

                self.opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.ae.parameters(), 1.0)
                self.opt.step()
                total_loss += float(loss.item())

        mean_loss = total_loss / num_steps
        self.loss_history.append(mean_loss)
        return mean_loss

    @no_grad()
    def encode(self, inputs: Tensor | Sequence) -> Tensor:
        sketches, out_device = self._prepare_sketches(inputs)
        if self.count == 0:
            self._update_stats_and_buffer(sketches)

        self.ae.eval()
        return self.ae.encode(self._normalize(sketches)).to(out_device)

    def forward(self, inputs: Tensor | Sequence, *, train: bool = True, steps: int | None = None) -> Tensor:
        if train and self.training:
            self.fit(inputs, steps = steps)
        return self.encode(inputs)

# local competition & fitness transforms

def dominated_novelty(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
    **kwargs
) -> Tensor:
    # Dominated Novelty Search (Bahlous-Boldi et al., 2025, arXiv:2502.00593):
    # mean distance to k nearest strictly-fitter neighbors; undefeated champions rank on top

    assert fitnesses.ndim == 1, f'fitnesses must be 1D (P,), got {tuple(fitnesses.shape)}'
    assert k >= 1, f'k must be >= 1, got {k}'

    p, device = fitnesses.shape[0], fitnesses.device
    if p <= 1:
        return fitnesses.float().clone()

    dist_mat = _resolve_distances(descriptors, distances, normalize = normalize, device = device)
    f = fitnesses.float()

    fitter = f.unsqueeze(0) > f.unsqueeze(1)
    num_fitter = fitter.sum(dim = -1)
    dominated = num_fitter > 0

    nearest = dist_mat.masked_fill(~fitter, float('inf')).topk(k = min(k, p - 1), dim = -1, largest = False).values
    valid = arange(nearest.shape[1], device = device) < num_fitter.unsqueeze(-1)
    dns = torch.where(valid, nearest, 0.).sum(dim = -1) / num_fitter.clamp(min = 1, max = k).float()

    if dominated.any():
        dom_vals = dns[dominated]
        max_dom, span = dom_vals.max(), (dom_vals.max() - dom_vals.min()).clamp(min = 1.)
    else:
        max_dom = span = torch.tensor(1., device = device)

    f_norm = (f - f.min()) / f.std(unbiased = False).clamp(min = 1e-6)
    return torch.where(dominated, dns, max_dom + span + 1. + 0.01 * f_norm)

def novelty_scores(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    archive: Tensor | None = None,
    k: int = 5,
    normalize: bool = True
) -> Tensor:
    # unconditional k-NN novelty score (Lehman & Stanley, 2011)

    assert k >= 1, f'k must be >= 1, got {k}'

    if exists(archive):
        assert exists(descriptors), 'descriptors must be provided when archive is passed'
        p = descriptors.shape[0]
        combined = cat((
            rearrange(descriptors.float(), 'n ... -> n (...)'),
            rearrange(archive.to(descriptors.device).float(), 'a ... -> a (...)')
        ), dim = 0)
        dist_mat = pairwise_distances(combined, normalize = normalize)[:p]
    else:
        dist_mat = _resolve_distances(descriptors, distances, normalize = normalize)
        p = dist_mat.shape[0]
        if p <= 1:
            return torch.zeros(p, dtype = torch.float32, device = dist_mat.device)

    eye = torch.eye(p, dist_mat.shape[1], dtype = torch.bool, device = dist_mat.device)
    m = min(k, dist_mat.shape[1] - 1)
    return dist_mat.masked_fill(eye, float('inf')).topk(k = m, dim = -1, largest = False).values.mean(dim = -1)

def fitness_sharing(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    quantile: float = 0.5,
    kernel_power: float = 1.0,
    normalize: bool = False,
    **kwargs
) -> Tensor:
    # Goldberg & Richardson (1987) niche-count fitness sharing

    assert fitnesses.ndim == 1, f'fitnesses must be 1D (P,), got {tuple(fitnesses.shape)}'
    assert 0. <= quantile <= 1., f'quantile must be in [0, 1], got {quantile}'

    n, device = fitnesses.shape[0], fitnesses.device
    if n <= 1 or quantile <= 0.:
        return fitnesses.float().clone()

    dist = _resolve_distances(descriptors, distances, normalize = normalize, device = device)
    triu = _upper_tri(dist)
    radius = max(triu.quantile(quantile).item() if triu.numel() > 0 else 1., 1e-8)

    niche_count = ((1. - (dist / radius).clamp(max = 1.)) ** kernel_power).sum(dim = -1).clamp(min = 1e-6)
    f = fitnesses.to(device).float()
    if (floor := f.min()) <= 0:
        f = f - floor + 1e-6

    return f / niche_count

def quality_gated_fitness(
    fitnesses: Tensor,
    novelty: Tensor,
    *,
    weight: float = 0.75,
    quantile: float = 0.5,
    sharpness: float = 2.0,
    min_fitness: float | None = None
) -> Tensor:
    # blend standardized fitness with viability-gated novelty so crashers cannot outrank stepping stones

    assert fitnesses.shape == novelty.shape, f'shape mismatch: {fitnesses.shape} vs {novelty.shape}'
    assert 0. <= weight <= 1., f'weight must be in [0, 1], got {weight}'
    assert 0. <= quantile <= 1., f'quantile must be in [0, 1], got {quantile}'

    f = fitnesses.float()
    nov = novelty.float().to(f.device)

    if f.numel() <= 1:
        return f.clone()

    viable = (f >= float(min_fitness)) if exists(min_fitness) else torch.ones_like(f, dtype = torch.bool)
    num_viable = int(viable.sum().item())

    if num_viable < 2:
        scores = (f - f.mean()) / f.std(unbiased = False).clamp(min = 1e-6)
        scores[f.argmax()] = scores.max() + 1.
        return scores

    v_f, v_nov = f[viable], nov[viable]
    v_std = v_f.std(unbiased = False).clamp(min = 1e-6)
    v_z = (v_f - v_f.mean()) / v_std

    dom = (arange(num_viable, device = f.device) != v_f.argmax()) if (exists(min_fitness) and num_viable > 2) else None
    nov_ref = v_nov[dom] if exists(dom) else v_nov

    nov_z = (v_nov - nov_ref.mean()) / nov_ref.std(unbiased = False).clamp(min = 1e-6)
    nov_pos = nov_z - (nov_z[dom] if exists(dom) else nov_z).min()

    thresh = float(min_fitness) if exists(min_fitness) else torch.quantile(v_f, quantile)
    gate = torch.sigmoid(sharpness * (v_f - thresh) / v_std) if sharpness > 0. else torch.ones_like(v_f)

    v_scores = (1. - weight) * v_z + weight * gate * nov_pos
    scores = torch.empty_like(f)
    scores[viable] = v_scores

    if num_viable < f.numel():
        nv_f = f[~viable]
        scores[~viable] = v_scores.min() - 2. + (nv_f - nv_f.max()) / (nv_f.max() - nv_f.min()).clamp(min = 1.)

    scores[f.argmax()] = scores.max() + 1.
    return scores

def quality_dominated_novelty(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
    weight: float = 0.75,
    quantile: float = 0.5,
    sharpness: float = 2.0,
    min_fitness: float | None = None,
    **kwargs
) -> Tensor:
    viable = (fitnesses.float() >= float(min_fitness)) if exists(min_fitness) else None

    if exists(viable) and int(viable.sum().item()) >= 2:
        dns = torch.zeros_like(fitnesses, dtype = torch.float32)
        dns[viable] = dominated_novelty(
            fitnesses[viable],
            descriptors[viable] if exists(descriptors) else None,
            distances = distances[viable][:, viable] if exists(distances) else None,
            k = k,
            normalize = normalize
        )
    else:
        dns = dominated_novelty(fitnesses, descriptors, distances = distances, k = k, normalize = normalize)

    return quality_gated_fitness(
        fitnesses,
        dns,
        weight = weight,
        quantile = quantile,
        sharpness = sharpness,
        min_fitness = min_fitness
    )

# fitness transform registry

FITNESS_TRANSFORM_REGISTRY: dict[str, Callable] = dict()

def register_fitness_transform(name: str, fn: Callable | None = None):
    def _decorator(f: Callable):
        FITNESS_TRANSFORM_REGISTRY[name] = f
        return f

    return _decorator(fn) if exists(fn) else _decorator

@register_fitness_transform('novelty')
def _novelty_fitness_transform(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
    **kwargs
) -> Tensor:
    return novelty_scores(descriptors, distances = distances, k = k, normalize = normalize)

register_fitness_transform('dns', dominated_novelty)
register_fitness_transform('dominated_novelty', dominated_novelty)
register_fitness_transform('quality_dns', quality_dominated_novelty)
register_fitness_transform('sharing', fitness_sharing)

# diversity metrics

@no_grad()
def vendi_score(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    normalize: bool = True,
    bandwidth: float | None = None
) -> float:
    # Vendi Score (Friedman & Dieng, 2023): effective number of distinct modes in [1, P]

    dist_mat = _resolve_distances(descriptors, distances, normalize = normalize).cpu()
    p = dist_mat.shape[0]
    if p <= 1:
        return 1.0

    sq_dists = dist_mat ** 2
    if not exists(bandwidth):
        triu = _upper_tri(sq_dists)
        bandwidth = max(float(triu.median().item()) if triu.numel() > 0 else 1., 1e-4)

    eigvals = torch.linalg.eigvalsh(torch.exp(-sq_dists / bandwidth) / float(p)).clamp(min = 1e-12)
    eigvals = eigvals / eigvals.sum()
    return float(math.exp(-(eigvals * torch.log(eigvals)).sum().item()))

@no_grad()
def behavioral_spread(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    normalize: bool = False,
    scale: Tensor | float | None = None
) -> float:
    if exists(descriptors) and exists(scale):
        descriptors = descriptors.float() / torch.as_tensor(scale, dtype = torch.float32, device = descriptors.device).clamp(min = 1e-6)

    dist = _resolve_distances(descriptors, distances, normalize = normalize)
    if dist.shape[0] <= 1:
        return 0.0

    return float(_upper_tri(dist).mean().item())

@no_grad()
def grid_coverage(
    descriptors: Tensor,
    *,
    bins: int = 8,
    dims: tuple[int, int] = (0, 1),
    bounds: tuple[float, float] = (-1., 1.)
) -> float:
    assert descriptors.ndim == 2 and descriptors.shape[1] >= 2, f'descriptors must be 2D with >= 2 columns, got {tuple(descriptors.shape)}'
    low, high = bounds
    d0, d1 = dims
    xy = descriptors[:, [d0, d1]].detach().cpu().float().clamp(low, high)
    coords = ((xy - low) / max(high - low, 1e-6) * bins).long().clamp(0, bins - 1)
    occupied = torch.unique(coords[:, 0] * bins + coords[:, 1]).numel()
    return float(occupied) / float(bins * bins)
