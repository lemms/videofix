"""RIFE 4.25 (Practical-RIFE, MIT licence) arbitrary-timestep interpolation.

Network definition ported from hzwer/Practical-RIFE ``train_log/IFNet_HDv3.py``
(v4.25/4.26 architecture: five IFBlocks, 4-channel feature head).  Weights are
loaded with ``weights_only=True`` so the pickle cannot execute code.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

WEIGHTS_URL = "https://huggingface.co/MonsterMMORPG/Practical-RIFE/resolve/main/train_log/flownet.pkl"
WEIGHTS_SHA256 = "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"
CACHE = Path.home() / ".cache" / "video_repair" / "rife"


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = flow.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((gx, gy), 0).unsqueeze(0)
    flow = torch.cat([flow[:, :1] / ((x.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((x.shape[2] - 1.0) / 2.0)], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def _conv(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=True), nn.LeakyReLU(0.2, True))


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class _ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class _IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[_ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1.0):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = _IFBlock(7 + 8, c=192)
        self.block1 = _IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = _IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = _IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = _IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = _Head()

    def forward(self, img0, img1, timestep: torch.Tensor, scale_list):
        f0, f1 = self.encode(img0), self.encode(img1)
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, blk in enumerate(blocks):
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale_list[i])
            else:
                wf0, wf1 = _warp(f0, flow[:, :2]), _warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow, scale=scale_list[i])
                flow = flow + fd
            w0, w1 = _warp(img0, flow[:, :2]), _warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m)


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading RIFE 4.25 weights to %s", dest)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError(f"RIFE weights checksum mismatch ({digest})")
    tmp.rename(dest)


class Rife:
    """``interpolate(a, b, t)`` with a, b (B,3,H,W) RGB in [0,1] on the model device."""

    def __init__(self, device: torch.device | str = "cuda", weights: Path | None = None, fp16: bool = True):
        self.device = torch.device(device)
        path = Path(weights) if weights else CACHE / "flownet.pkl"
        if not path.exists():
            _download(path)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        net = IFNet()
        missing, _ = net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()}, strict=False)
        if missing:
            raise RuntimeError(f"RIFE weights incompatible, missing {missing[:5]}")
        self.dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.net = net.to(self.device, self.dtype).eval()

    @torch.inference_mode()
    def interpolate(self, a: torch.Tensor, b: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        h, w = a.shape[-2:]
        # RIFE recommends half-resolution flow for 4K content
        scale = 0.5 if h * w > 2_500_000 else 1.0
        mult = max(128, int(128 / scale))
        ph, pw = (h + mult - 1) // mult * mult, (w + mult - 1) // mult * mult
        pad = (0, pw - w, 0, ph - h)
        a = F.pad(a.to(self.device, self.dtype), pad)
        b = F.pad(b.to(self.device, self.dtype), pad)
        if not torch.is_tensor(t):
            t = torch.tensor([t] * a.shape[0])
        ts = t.to(self.device, self.dtype).view(-1, 1, 1, 1).expand(a.shape[0], 1, ph, pw)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        out = self.net(a, b, ts, scale_list)
        return out[..., :h, :w].float().clamp_(0, 1)
