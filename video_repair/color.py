"""GPU conversion between native semi-planar YUV frames and RGB tensors."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .io import Frame, VideoInfo

_KR_KB = {1: (0.2126, 0.0722), 5: (0.299, 0.114), 6: (0.299, 0.114), 9: (0.2627, 0.0593), 10: (0.2627, 0.0593)}


def _coeffs(info: VideoInfo) -> tuple[float, float]:
    if info.colorspace in _KR_KB:
        return _KR_KB[info.colorspace]
    return _KR_KB[1] if info.height >= 720 else _KR_KB[6]


def _levels(info: VideoInfo) -> tuple[float, float, float, float]:
    """(y_offset, y_scale, c_offset, c_scale) in normalised code units."""
    full = info.color_range == 2
    if info.bit_depth > 8:
        m = 1023.0
        return (0.0, 1.0, 512 / m, 1.0) if full else (64 / m, 876 / m, 512 / m, 896 / m)
    m = 255.0
    return (0.0, 1.0, 128 / m, 1.0) if full else (16 / m, 219 / m, 128 / m, 224 / m)


def _code_max(info: VideoInfo) -> float:
    return 65472.0 if info.bit_depth > 8 else 255.0   # p010 keeps 10 bits in the MSBs


def upload(frames: Sequence[Frame], info: VideoInfo, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Native planes -> (y (B,1,H,W), uv (B,2,H/2,W/2)) normalised code values, float32."""
    y = torch.from_numpy(np.stack([f.y for f in frames])).to(device)
    uv = torch.from_numpy(np.stack([f.uv for f in frames])).to(device)
    m = _code_max(info)
    y = (y.float() / m).unsqueeze(1)
    b, h2, w = uv.shape
    uv = (uv.float() / m).view(b, h2, w // 2, 2).permute(0, 3, 1, 2).contiguous()
    return y, uv


def yuv_to_rgb(y: torch.Tensor, uv: torch.Tensor, info: VideoInfo) -> torch.Tensor:
    kr, kb = _coeffs(info)
    yo, ys, co, cs = _levels(info)
    if uv.shape[-2:] != y.shape[-2:]:
        uv = F.interpolate(uv, size=y.shape[-2:], mode="bilinear", align_corners=False)
    Y = (y - yo) / ys
    Pb = (uv[:, :1] - co) / cs
    Pr = (uv[:, 1:2] - co) / cs
    kg = 1 - kr - kb
    r = Y + 2 * (1 - kr) * Pr
    b = Y + 2 * (1 - kb) * Pb
    g = (Y - kr * r - kb * b) / kg
    return torch.cat([r, g, b], 1)   # not clamped: keeps the YUV round trip exact


def rgb_to_yuv(rgb: torch.Tensor, info: VideoInfo) -> tuple[torch.Tensor, torch.Tensor]:
    kr, kb = _coeffs(info)
    yo, ys, co, cs = _levels(info)
    r, g, b = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
    Y = kr * r + (1 - kr - kb) * g + kb * b
    Pb = (b - Y) / (2 * (1 - kb))
    Pr = (r - Y) / (2 * (1 - kr))
    y = Y * ys + yo
    uv = torch.cat([Pb, Pr], 1) * cs + co
    uv = F.avg_pool2d(uv, 2)
    return y, uv


def download(y: torch.Tensor, uv: torch.Tensor, info: VideoInfo) -> list[tuple[np.ndarray, np.ndarray]]:
    """Normalised (y, uv) tensors -> native numpy planes, one tuple per batch item."""
    m = _code_max(info)
    if info.bit_depth > 8:
        def q(t):  # quantise to 10 bits, store in the MSBs
            return ((t * 1023).round().clamp(0, 1023).to(torch.int32) * 64)
        np_dtype = np.uint16
    else:
        def q(t):
            return (t * m).round().clamp(0, 255).to(torch.int32)
        np_dtype = np.uint8
    yq = q(y[:, 0]).cpu().numpy().astype(np_dtype)
    b, _, h2, w2 = uv.shape
    uvq = q(uv).permute(0, 2, 3, 1).reshape(b, h2, w2 * 2).cpu().numpy().astype(np_dtype)
    return [(yq[i], uvq[i]) for i in range(b)]


def frames_to_rgb(frames: Sequence[Frame], info: VideoInfo, device: torch.device,
                  size: tuple[int, int] | None = None) -> torch.Tensor:
    """Frames -> (B,3,H,W) RGB in [0,1]; optionally area-resized to ``size`` (h, w)."""
    y, uv = upload(frames, info, device)
    if size is not None:
        y = F.interpolate(y, size=size, mode="area")
        uv = F.interpolate(uv, size=size, mode="bilinear", align_corners=False)
    return yuv_to_rgb(y, uv, info).clamp_(0, 1)
