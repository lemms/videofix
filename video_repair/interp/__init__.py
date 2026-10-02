"""Frame interpolation backends sharing ``interpolate(a, b, t) -> frame``."""

from __future__ import annotations

import torch


def load(name: str = "rife", device: torch.device | str = "cuda"):
    if name == "rife":
        from .rife import Rife
        return Rife(device)
    raise ValueError(f"unknown interpolator {name!r}")
