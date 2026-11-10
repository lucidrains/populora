# composable novelty, local competition, and behavioral descriptor primitives

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, is_tensor

from populora._utils import cast_tensor, default, exists


# distance primitives

def pairwise_distances(
    descriptors: Tensor,
    *,
    normalize: bool = True,
    p: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute the `(P, P)` pairwise $L_p$ distance matrix from a `(P, ...)`
    descriptor tensor.

    When `normalize = True`, each descriptor dimension is standardized across
    the population so high-variance dimensions do not dominate distances.
    """
    assert descriptors.ndim >= 1, f'descriptors must have at least 1 dim, got {descriptors.shape}'
    desc = descriptors.float().reshape(descriptors.shape[0], -1)
    n = desc.shape[0]

    if n <= 1:
        return torch.zeros((n, n), dtype = torch.float32, device = desc.device)

    if normalize:
        std = desc.std(dim = 0, unbiased = False).clamp(min = eps)
        desc = (desc - desc.mean(dim = 0, keepdim = True)) / std

    dist = torch.cdist(desc, desc, p = p)
    dist.fill_diagonal_(0.0)
    return dist


def _resolve_distances(
    descriptors: Tensor | None = None,
    distances: Tensor | None = None,
    *,
    normalize: bool = True,
    p: float = 2.0,
    device: torch.device | None = None,
) -> Tensor:
    assert exists(descriptors) or exists(distances), 'either descriptors or distances must be provided'

    if exists(distances):
        dist = distances.float()
        if exists(device) and dist.device != device:
            dist = dist.to(device)
        assert dist.ndim == 2 and dist.shape[0] == dist.shape[1], f'distances must be square (P, P), got {tuple(dist.shape)}'
        return dist

    desc = descriptors.to(device) if (exists(device) and descriptors.device != device) else descriptors
    return pairwise_distances(desc, normalize = normalize, p = p)


# trajectory sketching

def sketch_trajectories(
    trajectories: Tensor | Sequence[Tensor | np.ndarray | tuple],
    *,
    num_subsamples: int = 8,
    max_horizon: int | None = None,
    include_stats: bool = True,
    include_length: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Summarize a batch or list of variable-length rollout trajectories into a
    fixed-size `(P, sketch_dim)` tensor suitable for distance computation or
    unsupervised representation learning (`Aurora`).

    Accepts:
      - A 3D tensor `(P, T, D)`
      - A sequence of `(T_i, D)` tensors or numpy arrays
      - A sequence of `(obs_traj, act_traj)` tuples (concatenated along feature dim)
    """
    assert num_subsamples >= 1, f'num_subsamples must be >= 1, got {num_subsamples}'

    if is_tensor(trajectories) and trajectories.ndim == 3:
        traj_list = [trajectories[i] for i in range(trajectories.shape[0])]
        inferred_device = default(device, trajectories.device)
    else:
        traj_list = list(trajectories)
        inferred_device = device

    assert len(traj_list) > 0, 'trajectories cannot be empty'

    parsed: list[Tensor] = []
    for item in traj_list:
        if isinstance(item, (tuple, list)) and len(item) == 2 and not is_tensor(item):
            a, b = item
            ta = torch.as_tensor(a, dtype = torch.float32)
            tb = torch.as_tensor(b, dtype = torch.float32)
            if ta.ndim == 1:
                ta = ta.unsqueeze(-1)
            if tb.ndim == 1:
                tb = tb.unsqueeze(-1)
            t_len = min(ta.shape[0], tb.shape[0])
            step_tensor = torch.cat([ta[:t_len], tb[:t_len]], dim = -1)
        else:
            step_tensor = torch.as_tensor(item, dtype = torch.float32)
            if step_tensor.ndim == 1:
                step_tensor = step_tensor.unsqueeze(-1)

        if not exists(inferred_device) and is_tensor(item):
            inferred_device = item.device

        parsed.append(step_tensor)

    horizon_ref = float(default(max_horizon, max(max(t.shape[0] for t in parsed), 1)))
    feat_dim = parsed[0].shape[-1]

    sketches = []
    for traj in parsed:
        t_len = traj.shape[0]
        if t_len == 0:
            sub = torch.zeros(num_subsamples * feat_dim, dtype = torch.float32)
            mean_s = torch.zeros(feat_dim, dtype = torch.float32)
            std_s = torch.zeros(feat_dim, dtype = torch.float32)
        else:
            traj_cpu = traj.detach().cpu().float()
            idx = torch.linspace(0, max(t_len - 1, 0), steps = num_subsamples).round().long()
            sub = traj_cpu[idx].reshape(-1)
            mean_s = traj_cpu.mean(dim = 0)
            std_s = traj_cpu.std(dim = 0, unbiased = False) if t_len > 1 else torch.zeros_like(mean_s)

        parts = [sub]
        if include_stats:
            parts.extend([mean_s, std_s])
        if include_length:
            parts.append(torch.tensor([float(t_len) / max(horizon_ref, 1.0)], dtype = torch.float32))

        sketches.append(torch.cat(parts, dim = 0))

    out = torch.stack(sketches, dim = 0)
    if exists(inferred_device):
        out = out.to(inferred_device)
    return out


# AURORA unsupervised descriptor autoencoder (Cully 2019; Grillotti & Cully 2022)

class AuroraAutoEncoder(nn.Module):
    """
    Lightweight MLP autoencoder mapping trajectory sketches or phenotypic
    summaries `(B, input_dim)` into a low-dimensional latent behavioral
    descriptor `z` of shape `(B, latent_dim)`.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 6,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        return torch.tanh(self.encoder(x))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z


class Aurora(nn.Module):
    """
    Online unsupervised behavioral descriptor learner (AURORA; Cully, 2019;
    Grillotti & Cully, 2022) with running Welford input standardization,
    circular replay buffer, and VICReg variance + covariance regularization to
    prevent latent collapse.

    Can be called directly on a `(P, D)` tensor of trajectory sketches / state
    summaries or on a sequence of variable-length rollout trajectories.
    """

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
        device: torch.device | str | None = None,
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
        self.ae = AuroraAutoEncoder(
            input_dim = self.input_dim,
            latent_dim = self.latent_dim,
            hidden_dim = self.hidden_dim,
        ).to(self._device)
        self.opt = torch.optim.AdamW(
            self.ae.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay,
        )
        self.buffer = torch.zeros((self.buffer_capacity, self.input_dim), dtype = torch.float32, device = self._device)
        self.running_mean = torch.zeros(self.input_dim, dtype = torch.float32, device = self._device)
        self.running_M2 = torch.ones(self.input_dim, dtype = torch.float32, device = self._device)

    def _prepare_sketches(self, inputs: Tensor | Sequence) -> tuple[Tensor, torch.device]:
        if is_tensor(inputs) and inputs.ndim == 2:
            out_device = inputs.device
            sketches = inputs.detach().to(self._device, dtype = torch.float32)
        else:
            sketches = sketch_trajectories(
                inputs,
                num_subsamples = self.num_subsamples,
                max_horizon = self.max_horizon,
                device = self._device,
            )
            out_device = sketches.device

        if not exists(self.ae):
            self._init_modules(sketches.shape[-1])

        assert sketches.shape[-1] == self.input_dim, (
            f'Aurora expected input_dim={self.input_dim}, got {sketches.shape[-1]}'
        )
        return sketches, out_device

    @torch.no_grad()
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

        for i in range(b):
            self.buffer[self.buf_ptr] = x[i]
            self.buf_ptr = (self.buf_ptr + 1) % self.buffer_capacity
            self.buf_size = min(self.buf_size + 1, self.buffer_capacity)

    def _normalize(self, x: Tensor) -> Tensor:
        std = (self.running_M2 / max(self.count, 1)).clamp(min = 1e-4).sqrt()
        return ((x - self.running_mean) / std).clamp(-5.0, 5.0)

    def fit(self, inputs: Tensor | Sequence, *, steps: int | None = None) -> float:
        """
        Update running normalization statistics, add `inputs` to the replay
        buffer, and perform online autoencoder gradient steps.
        """
        sketches, _ = self._prepare_sketches(inputs)
        self._update_stats_and_buffer(sketches)

        num_steps = default(steps, self.update_steps)
        if self.buf_size < 4 or num_steps <= 0:
            return 0.0

        self.ae.train()
        total_loss = 0.0

        with torch.enable_grad():
            for _ in range(num_steps):
                b_sz = min(self.batch_size, self.buf_size)
                idx = torch.randint(0, self.buf_size, (b_sz,), device = self._device)
                batch = self._normalize(self.buffer[idx])

                recon, z = self.ae(batch)
                recon_loss = F.smooth_l1_loss(recon, batch)

                # VICReg variance hinge: keep per-dimension latent std >= 0.35
                z_std = torch.sqrt(z.var(dim = 0, unbiased = False) + 1e-4)
                var_loss = F.relu(0.35 - z_std).mean()

                # VICReg covariance penalty: decorrelate latent dimensions
                z_centered = z - z.mean(dim = 0, keepdim = True)
                cov = (z_centered.T @ z_centered) / max(b_sz - 1, 1)
                off_diag = cov - torch.diag(torch.diag(cov))
                cov_loss = (off_diag ** 2).mean()

                loss = recon_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                self.opt.step()
                total_loss += float(loss.item())

        mean_loss = total_loss / num_steps
        self.loss_history.append(mean_loss)
        return mean_loss

    @torch.no_grad()
    def encode(self, inputs: Tensor | Sequence) -> Tensor:
        """
        Project `inputs` (`(P, D)` sketches or variable-length trajectories)
        into `(P, latent_dim)` behavioral descriptors without updating weights.
        """
        sketches, out_device = self._prepare_sketches(inputs)
        if self.count == 0:
            self._update_stats_and_buffer(sketches)

        self.ae.eval()
        norm_x = self._normalize(sketches)
        z = self.ae.encode(norm_x)
        return z.to(out_device)

    def forward(self, inputs: Tensor | Sequence, *, train: bool = True, steps: int | None = None) -> Tensor:
        if train and self.training:
            self.fit(inputs, steps = steps)
        return self.encode(inputs)


# local competition & fitness transformation primitives

def dominated_novelty(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
) -> Tensor:
    """
    Dominated Novelty Search (DNS; Bahlous-Boldi et al., 2025, arXiv:2502.00593).

    For each individual $i$ with fitness $f_i$:
      1. Identify the fitter set $\\mathcal{F}_i = \\{j \\mid f_j > f_i\\}$.
      2. If $\\mathcal{F}_i \\ne \\emptyset$, score $i$ by its mean distance to
         the $\\min(k, |\\mathcal{F}_i|)$ nearest neighbors in $\\mathcal{F}_i$.
      3. Undefeated champions ($\\mathcal{F}_i = \\emptyset$) receive a score
         strictly above all dominated individuals (ties broken by raw fitness).
    """
    assert fitnesses.ndim == 1, f'fitnesses must be 1D (P,), got {tuple(fitnesses.shape)}'
    device = fitnesses.device
    p = fitnesses.shape[0]
    assert k >= 1, f'k must be >= 1, got {k}'

    if p <= 1:
        return fitnesses.float().clone()

    dist_mat = _resolve_distances(descriptors, distances, normalize = normalize, device = device)
    f = fitnesses.float()

    dns_scores = torch.zeros(p, dtype = torch.float32, device = device)
    undefeated_indices = []

    for i in range(p):
        fitter_mask = f > f[i]
        fitter_idx = torch.nonzero(fitter_mask, as_tuple = False).squeeze(-1)

        if fitter_idx.numel() == 0:
            undefeated_indices.append(i)
        else:
            dists_to_fitter = dist_mat[i, fitter_idx]
            m = min(k, int(dists_to_fitter.numel()))
            nearest_dists, _ = torch.topk(dists_to_fitter, k = m, largest = False)
            dns_scores[i] = nearest_dists.mean()

    dominated_mask = torch.ones(p, dtype = torch.bool, device = device)
    if undefeated_indices:
        dominated_mask[undefeated_indices] = False

    if dominated_mask.any():
        max_dom = dns_scores[dominated_mask].max()
        span = (dns_scores[dominated_mask].max() - dns_scores[dominated_mask].min()).clamp(min = 1.0)
    else:
        max_dom = torch.tensor(1.0, device = device)
        span = torch.tensor(1.0, device = device)

    f_std = f.std(unbiased = False).clamp(min = 1e-6)
    f_norm = (f - f.min()) / f_std

    for i in undefeated_indices:
        dns_scores[i] = max_dom + span + 1.0 + 0.01 * f_norm[i]

    return dns_scores


def novelty_scores(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    archive: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
) -> Tensor:
    """
    Unconditional k-NN Novelty Search score (Lehman & Stanley, 2011): mean
    distance from each individual to its $k$ nearest neighbors in the
    population (plus optional behavioral `archive` of shape `(A, ...)`).
    """
    assert k >= 1, f'k must be >= 1, got {k}'

    if exists(archive):
        assert exists(descriptors), 'descriptors must be provided when archive is passed'
        p = descriptors.shape[0]
        desc_flat = descriptors.float().reshape(p, -1)
        arch_flat = archive.to(descriptors.device).float().reshape(archive.shape[0], -1)
        combined = torch.cat([desc_flat, arch_flat], dim = 0)
        full_dist = pairwise_distances(combined, normalize = normalize)
        dist_mat = full_dist[:p, :]
        # exclude self-distance along the first (p, p) block diagonal
        eye = torch.eye(p, full_dist.shape[1], dtype = torch.bool, device = dist_mat.device)
        dist_mat = dist_mat.masked_fill(eye, float('inf'))
        m = min(k, full_dist.shape[1] - 1)
    else:
        dist_mat = _resolve_distances(descriptors, distances, normalize = normalize)
        p = dist_mat.shape[0]
        if p <= 1:
            return torch.zeros(p, dtype = torch.float32, device = dist_mat.device)
        eye = torch.eye(p, dtype = torch.bool, device = dist_mat.device)
        dist_mat = dist_mat.masked_fill(eye, float('inf'))
        m = min(k, p - 1)

    nearest, _ = torch.topk(dist_mat, k = m, dim = -1, largest = False)
    return nearest.mean(dim = -1)


def fitness_sharing(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    quantile: float = 0.5,
    kernel_power: float = 1.0,
    normalize: bool = False,
) -> Tensor:
    """
    Goldberg & Richardson (1987) fitness sharing over any descriptor tensor or
    pairwise distance matrix.

    Shifts `fitnesses` to be strictly positive and divides each individual's
    fitness by its niche count within the `quantile` pairwise distance radius.
    """
    assert fitnesses.ndim == 1, f'fitnesses must be 1D (P,), got {tuple(fitnesses.shape)}'
    assert 0. <= quantile <= 1., f'quantile must be in [0, 1], got {quantile}'

    device = fitnesses.device
    n = fitnesses.shape[0]
    if n <= 1 or quantile <= 0.:
        return fitnesses.float().clone()

    dist = _resolve_distances(descriptors, distances, normalize = normalize, device = device)

    triu = dist.masked_select(torch.triu(torch.ones(n, n, dtype = torch.bool, device = device), diagonal = 1))
    radius = triu.quantile(quantile).item() if triu.numel() > 0 else 1.0
    radius = max(radius, 1e-8)

    overlap = (1.0 - (dist / radius).clamp(max = 1.0)) ** kernel_power
    niche_count = overlap.sum(dim = -1).clamp(min = 1e-6)

    f = fitnesses.to(device).float()
    floor = f.min()
    if floor <= 0:
        f = f - floor + 1e-6

    return f / niche_count


def quality_gated_fitness(
    fitnesses: Tensor,
    novelty: Tensor,
    *,
    weight: float = 0.75,
    quantile: float = 0.5,
    sharpness: float = 2.0,
) -> Tensor:
    """
    Blend standardized task `fitnesses` with standardized `novelty` (or DNS)
    scores gated by a smooth viability threshold around the `quantile` task
    fitness, preventing low-fitness crashers from outranking viable solutions
    purely via extreme novelty.

    When `weight = 1.0` and `sharpness = 0.0`, reduces to pure `novelty`.
    When `weight = 0.0`, reduces to standardized `fitnesses`.
    """
    assert fitnesses.shape == novelty.shape, f'shape mismatch: {fitnesses.shape} vs {novelty.shape}'
    assert 0. <= weight <= 1., f'weight must be in [0, 1], got {weight}'
    assert 0. <= quantile <= 1., f'quantile must be in [0, 1], got {quantile}'

    f = fitnesses.float()
    nov = novelty.float().to(f.device)

    if f.numel() <= 1:
        return f.clone()

    f_std = f.std(unbiased = False).clamp(min = 1e-6)
    f_z = (f - f.mean()) / f_std

    nov_std = nov.std(unbiased = False).clamp(min = 1e-6)
    nov_z = (nov - nov.mean()) / nov_std
    nov_pos = nov_z - nov_z.min()

    if sharpness > 0.:
        q_val = torch.quantile(f, quantile)
        gate = torch.sigmoid(sharpness * (f - q_val) / f_std)
    else:
        gate = torch.ones_like(f)

    scores = (1.0 - weight) * f_z + weight * gate * nov_pos

    # Guarantee the global task-fitness champion ranks strictly #1
    best_idx = int(f.argmax().item())
    scores[best_idx] = scores.max() + 1.0
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
) -> Tensor:
    """
    Compose `dominated_novelty` with `quality_gated_fitness`.
    """
    dns = dominated_novelty(
        fitnesses,
        descriptors,
        distances = distances,
        k = k,
        normalize = normalize,
    )
    return quality_gated_fitness(
        fitnesses,
        dns,
        weight = weight,
        quantile = quantile,
        sharpness = sharpness,
    )


# fitness transform registry

FITNESS_TRANSFORM_REGISTRY: dict[str, Callable] = dict()


def register_fitness_transform(name: str, fn: Callable):
    FITNESS_TRANSFORM_REGISTRY[name] = fn


def _novelty_fitness_transform(
    fitnesses: Tensor,
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    k: int = 5,
    normalize: bool = True,
    **kwargs,
) -> Tensor:
    return novelty_scores(
        descriptors,
        distances = distances,
        k = k,
        normalize = normalize,
    )


register_fitness_transform('dns', dominated_novelty)
register_fitness_transform('dominated_novelty', dominated_novelty)
register_fitness_transform('quality_dns', quality_dominated_novelty)
register_fitness_transform('sharing', fitness_sharing)
register_fitness_transform('novelty', _novelty_fitness_transform)


# standalone diversity metrics

@torch.no_grad()
def vendi_score(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    normalize: bool = True,
    bandwidth: float | None = None,
) -> float:
    """
    Vendi Score (Friedman & Dieng, 2023): effective number of distinct
    behavioral modes $\\exp(-\\sum_i \\lambda_i \\log \\lambda_i) \\in [1, P]$
    computed from the RBF similarity kernel over `descriptors` or `distances`.
    """
    dist_mat = _resolve_distances(descriptors, distances, normalize = normalize).cpu()
    p = dist_mat.shape[0]
    if p <= 1:
        return 1.0

    sq_dists = dist_mat ** 2
    if not exists(bandwidth):
        triu = sq_dists[torch.triu(torch.ones(p, p, dtype = torch.bool), diagonal = 1)]
        med_sq = float(triu.median().item()) if triu.numel() > 0 else 1.0
        bandwidth = max(med_sq, 1e-4)

    K = torch.exp(-sq_dists / bandwidth) / float(p)
    eigvals = torch.linalg.eigvalsh(K).clamp(min = 1e-12)
    eigvals = eigvals / eigvals.sum()
    entropy = -(eigvals * torch.log(eigvals)).sum().item()
    return float(math.exp(entropy))


@torch.no_grad()
def behavioral_spread(
    descriptors: Tensor | None = None,
    *,
    distances: Tensor | None = None,
    normalize: bool = False,
    scale: Tensor | float | None = None,
) -> float:
    """
    Mean off-diagonal pairwise distance across the population.
    """
    if exists(descriptors) and exists(scale):
        scale_t = torch.as_tensor(scale, dtype = torch.float32, device = descriptors.device)
        descriptors = descriptors.float() / scale_t.clamp(min = 1e-6)

    dist = _resolve_distances(descriptors, distances, normalize = normalize)
    p = dist.shape[0]
    if p <= 1:
        return 0.0

    triu = dist[torch.triu(torch.ones(p, p, dtype = torch.bool, device = dist.device), diagonal = 1)]
    return float(triu.mean().item())


@torch.no_grad()
def grid_coverage(
    descriptors: Tensor,
    *,
    bins: int = 8,
    dims: tuple[int, int] = (0, 1),
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """
    Fraction of occupied cells in a 2D `bins x bins` grid over descriptor
    dimensions `dims` within `bounds`.
    """
    assert descriptors.ndim == 2 and descriptors.shape[1] >= 2, (
        f'descriptors must be 2D with at least 2 columns, got {tuple(descriptors.shape)}'
    )
    low, high = bounds
    span = max(high - low, 1e-6)

    d0, d1 = dims
    xy = descriptors[:, [d0, d1]].detach().cpu().float().clamp(low, high)
    coords = ((xy - low) / span * bins).long().clamp(0, bins - 1)
    occupied = set(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
    return float(len(occupied)) / float(bins * bins)
