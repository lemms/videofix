"""Cheap per-frame image signals, computed on the GPU in batches.

Every spatial map is pooled onto the same (gh, gw) grid as the ViT patch
tokens so pixel- and feature-level evidence can be combined cell by cell.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LUMA = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def _pool(x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, grid)


def exposure(rgb_small: torch.Tensor) -> dict[str, torch.Tensor]:
    """Brightness statistics for black/white/flash frames."""
    luma = (rgb_small * LUMA.to(rgb_small)).sum(1)
    flat = luma.flatten(1)
    return {
        "mean_y": flat.mean(1),
        "std_y": flat.std(1),
        "dark_frac": (flat < 0.04).float().mean(1),
        "bright_frac": (flat > 0.97).float().mean(1),
    }


def blockiness(y: torch.Tensor, grid: tuple[int, int], block: int = 8) -> torch.Tensor:
    """Ratio of gradient energy on the codec block grid to energy off it, per cell.

    ~1 for natural (well-encoded) content; macroblock corruption puts hard
    edges exactly on multiples of 8 in *both* directions and pushes the ratio
    well above 1.  Taking the minimum over the two directions ignores object
    edges that happen to sit on the grid in one direction.  The floor (~1
    8-bit code) keeps smooth areas, where both energies are near zero, from
    producing huge ratios out of ordinary banding.
    """
    eps = 1.0 / 255
    ratios = []
    # horizontal differences, then vertical ones transposed so boundaries are columns
    for yy, g, transposed in ((y, grid, False), (y.transpose(-1, -2), grid[::-1], True)):
        d = (yy[..., 1:] - yy[..., :-1]).abs()
        mean_on = _pool(d[..., block - 1::block], g)       # differences across block boundaries
        mean_all = _pool(d, g)
        mean_off = (block * mean_all - mean_on) / (block - 1)
        r = (mean_on + eps) / (mean_off.clamp_min(0) + eps)
        ratios.append(r.transpose(-1, -2) if transposed else r)
        del d
    return torch.minimum(*ratios).squeeze(1)


def line_scores(y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Peak of the row/column difference profile relative to its local average.

    Horizontal tearing, dropped rows and sensor line noise create a single
    row boundary that is much stronger than its neighbours.
    """
    eps = 1e-3
    out = {}
    for name, d, dim in (("row_score", y[..., 1:, :] - y[..., :-1, :], -1),
                         ("col_score", y[..., 1:] - y[..., :-1], -2)):
        prof = d.abs().mean(dim).squeeze(1)                    # (B, L)
        local = F.avg_pool1d(prof.unsqueeze(1), 31, 1, 15, count_include_pad=False).squeeze(1)
        out[name] = (prof / (local + eps)).amax(1)
    return out


def high_freq(y: torch.Tensor) -> torch.Tensor:
    """Mean absolute Laplacian: rises with noise/static, falls with blur."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    return F.conv2d(y, k).abs().flatten(1).mean(1)


def pixel_dist(a: torch.Tensor, b: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Mean absolute RGB difference per grid cell, (B, gh, gw)."""
    return _pool((a - b).abs().mean(1, keepdim=True), grid).squeeze(1)


def patch_dist(cur: torch.Tensor, ref: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Cosine distance from each patch of ``cur`` to its best match in a
    (2r+1)^2 neighbourhood of ``ref`` -- tolerant to small motion.

    cur, ref: (B, gh, gw, C) L2-normalised.  Returns (B, gh, gw).
    """
    b, gh, gw, c = cur.shape
    k = 2 * radius + 1
    r = ref.permute(0, 3, 1, 2).float()
    r = F.pad(r, (radius,) * 4, mode="replicate")
    r = F.unfold(r, k).view(b, c, k * k, gh * gw)
    q = cur.permute(0, 3, 1, 2).float().reshape(b, c, 1, gh * gw)
    sim = (q * r).sum(1).amax(1)
    return (1 - sim).view(b, gh, gw)
