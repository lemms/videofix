"""Self-supervised ViT features (DINOv3, falling back to DINOv2)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

BACKBONES = {
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",   # gated: accept the licence on HF, then `hf auth login`
    "dinov2": "facebook/dinov2-small",
}
GRID_W = 24          # patch columns; rows follow the aspect ratio

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone:
    def __init__(self, name: str = "auto", device: torch.device | str = "cuda", frame_hw: tuple[int, int] = (1080, 1920)):
        from transformers import AutoModel

        order = ["dinov3", "dinov2"] if name == "auto" else [name]
        last: Exception | None = None
        for key in order:
            repo = BACKBONES.get(key, key)
            try:
                self.dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
                self.model = AutoModel.from_pretrained(repo, dtype=self.dtype).to(device).eval()
                self.name = key
                break
            except Exception as e:
                last = e
                if key == "dinov3":
                    log.warning("DINOv3 unavailable (%s). Using DINOv2. To enable DINOv3, accept the licence at "
                                "https://huggingface.co/%s and run `hf auth login`.", type(e).__name__, repo)
        else:
            raise RuntimeError(f"could not load a backbone: {last}")
        cfg = self.model.config
        self.patch = cfg.patch_size
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.dim = cfg.hidden_size
        h, w = frame_hw
        self.grid = (max(4, round(GRID_W * h / w)), GRID_W)
        self.input_hw = (self.grid[0] * self.patch, self.grid[1] * self.patch)
        self.device = torch.device(device)
        self.mean = _MEAN.to(self.device, self.dtype)
        self.std = _STD.to(self.device, self.dtype)

    @torch.inference_mode()
    def __call__(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """rgb (B,3,H,W) in [0,1] -> (cls (B,C), patches (B,gh,gw,C)), both L2-normalised fp16."""
        x = rgb.to(self.device, self.dtype)
        if x.shape[-2:] != self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode="bilinear", antialias=True, align_corners=False)
        x = (x - self.mean) / self.std
        tokens = self.model(pixel_values=x).last_hidden_state
        cls = F.normalize(tokens[:, 0].float(), dim=-1).half()
        p = tokens[:, self.n_prefix:]
        p = F.normalize(p.float(), dim=-1).half().view(x.shape[0], *self.grid, self.dim)
        return cls, p
