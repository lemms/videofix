"""Pass 1: stream the video once and store compact per-frame features.

Per frame we keep scalars (exposure, blockiness, line and noise scores,
decoder flags) and a few maps on the ViT patch grid:

* ``pd{k}``  pixel distance to frame t-k (k = 1..3)
* ``fd{k}``  DINO patch distance to frame t-k (motion-tolerant)
* ``block``  codec block-edge ratio
* ``novel``  distance of each patch to a memory bank of patches sampled from
  keyframes across the whole video (PatchCore/AnomalyDINO style)

Distances to *previous* frames are enough: the detector reads "next" distances
from later rows (d(t, t+1) == pd1[t+1]).  Features go to chunk files so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import color, io, signals
from .features import Backbone

log = logging.getLogger(__name__)

FORMAT_VERSION = 2
CHUNK = 1024
LAGS = (1, 2, 3)
BANK_FRAMES = 96
BANK_SIZE = 16384


def _batch_size(info: io.VideoInfo) -> int:
    return 8 if info.width * info.height <= 2_100_000 else 3


class _Ring:
    """The last few frames' thumbnails/patches, for lagged distances."""

    def __init__(self, n: int):
        self.n = n
        self.items: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def push(self, idx, thumb, patches, cls, yq):
        self.items.append((idx, thumb, patches, cls, yq))
        del self.items[: -self.n]

    def get(self, idx):
        for it in self.items:
            if it[0] == idx:
                return it
        return None


class FeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def chunk_path(self, c: int) -> Path:
        return self.root / f"chunk_{c:06d}.npz"

    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def done_chunks(self) -> set[int]:
        return {int(p.stem.split("_")[1]) for p in self.root.glob("chunk_*.npz")}

    def load(self) -> dict[str, np.ndarray]:
        """Concatenate every chunk into one dict of arrays, sorted by frame index."""
        parts = [dict(np.load(self.chunk_path(c))) for c in sorted(self.done_chunks())]
        if not parts:
            raise FileNotFoundError(f"no features in {self.root}")
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(out["index"], kind="stable")
        return {k: v[order] for k, v in out.items()}


def _build_bank(path, info, backbone, device, hwaccel) -> torch.Tensor:
    frames = io.sample_keyframes(path, min(BANK_FRAMES, max(4, info.n_frames // 30)), info, hwaccel)
    feats = []
    for i in range(0, len(frames), 8):
        rgb = color.frames_to_rgb(frames[i:i + 8], info, device, backbone.input_hw)
        _, p = backbone(rgb)
        feats.append(p.reshape(-1, backbone.dim))
    bank = torch.cat(feats)
    if len(bank) > BANK_SIZE:
        g = torch.Generator(device="cpu").manual_seed(0)
        bank = bank[torch.randperm(len(bank), generator=g)[:BANK_SIZE].to(bank.device)]
    log.info("patch memory bank: %d patches from %d keyframes", len(bank), len(frames))
    return bank


def _novelty(patches: torch.Tensor, bank: torch.Tensor, k: int = 3) -> torch.Tensor:
    b, gh, gw, c = patches.shape
    sim = patches.reshape(-1, c) @ bank.T                     # fp16 matmul
    top = sim.float().topk(k, dim=1).values.mean(1)
    return (1 - top).view(b, gh, gw)


def analyze(path: str | Path, out_dir: str | Path, backbone_name: str = "auto",
            hwaccel: bool = True, device: str = "cuda", progress: bool = True) -> FeatureStore:
    path = Path(path)
    store = FeatureStore(Path(out_dir))
    store.root.mkdir(parents=True, exist_ok=True)
    info = io.probe(path)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    backbone = Backbone(backbone_name, dev, (info.height, info.width))
    grid = backbone.grid

    meta = {"version": FORMAT_VERSION, "source": str(path.resolve()), "size": path.stat().st_size,
            "backbone": backbone.name, "grid": list(grid), "chunk": CHUNK, "lags": list(LAGS),
            "info": {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in asdict(info).items()}}
    if store.meta_path.exists():
        old = store.meta()
        keep = ("version", "source", "size", "backbone", "grid", "chunk")
        if any(old.get(k) != meta.get(k) for k in keep):
            log.warning("feature store %s belongs to a different input/config; recomputing", store.root)
            for p in store.root.glob("chunk_*.npz"):
                p.unlink()
    store.meta_path.write_text(json.dumps(meta, indent=1))

    done = store.done_chunks()
    first_missing = 0
    while first_missing in done:
        first_missing += 1
    start = max(0, first_missing * CHUNK - max(LAGS))
    if done:
        log.info("resuming: %d chunks already analysed, decoding from frame %d", len(done), start)

    bank_path = store.root / "bank.pt"
    if bank_path.exists():
        bank = torch.load(bank_path, map_location=dev, weights_only=True)
    else:
        bank = _build_bank(path, info, backbone, dev, hwaccel)
        torch.save(bank.cpu(), bank_path)

    ring = _Ring(max(LAGS))
    rows: dict[int, list[dict]] = {}
    bs = _batch_size(info)
    batch: list[io.Frame] = []
    t0, n_done = time.time(), 0
    thumb_hw = (grid[0] * 4, grid[1] * 4)
    still_hw = (max(8, info.height // 4), max(8, info.width // 4))

    def flush_chunks(final: bool, upto: int) -> None:
        for c in sorted(rows):
            if c in done:
                rows.pop(c)
                continue
            if final or (c + 1) * CHUNK <= upto:
                recs = rows.pop(c)
                if recs:
                    arrays = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
                    np.savez(store.chunk_path(c), **arrays)

    def process(frames: list[io.Frame]) -> None:
        nonlocal n_done
        y, uv = color.upload(frames, info, dev)
        rgb = color.yuv_to_rgb(F.interpolate(y, size=backbone.input_hw, mode="area"),
                               F.interpolate(uv, size=backbone.input_hw, mode="bilinear", align_corners=False),
                               info).clamp_(0, 1)
        cls, patches = backbone(rgb)
        thumb = F.interpolate(rgb, size=thumb_hw, mode="area")
        yq = F.interpolate(y, size=still_hw, mode="area")
        expo = signals.exposure(rgb)
        lines = signals.line_scores(y)
        hf = signals.high_freq(y)
        block = signals.blockiness(y, grid)
        novel = _novelty(patches, bank)
        code = 1023.0 if info.bit_depth > 8 else 255.0

        for i, fr in enumerate(frames):
            rec = {
                "index": fr.index, "pts": fr.pts if fr.pts is not None else -1, "time": fr.time,
                "key": fr.key, "pict_type": "IPB".find(fr.pict_type[:1]) if fr.pict_type else -1,
                "corrupt": fr.corrupt, "decode_errors": fr.decode_errors, "packet_size": fr.packet_size,
                "hf": hf[i].item(), "block_mean": block[i].mean().item(),
                "block": block[i].half().cpu().numpy(), "novel": novel[i].half().cpu().numpy(),
            }
            for k, v in expo.items():
                rec[k] = v[i].item()
            for k, v in lines.items():
                rec[k] = v[i].item()
            for lag in LAGS:
                prev = ring.get(fr.index - lag)
                if prev is None:
                    rec[f"pd{lag}"] = np.full(grid, np.nan, np.float16)
                    rec[f"fd{lag}"] = np.full(grid, np.nan, np.float16)
                    rec[f"cd{lag}"] = np.nan
                    if lag == 1:
                        rec["still"] = np.nan
                    continue
                _, p_thumb, p_patch, p_cls, p_yq = prev
                rec[f"pd{lag}"] = signals.pixel_dist(thumb[i:i + 1], p_thumb, grid)[0].half().cpu().numpy()
                rec[f"fd{lag}"] = signals.patch_dist(patches[i:i + 1], p_patch)[0].half().cpu().numpy()
                rec[f"cd{lag}"] = 1 - float((cls[i].float() * p_cls.float()).sum())
                if lag == 1:
                    rec["still"] = float((yq[i:i + 1] - p_yq).abs().mean()) * code
            ring.push(fr.index, thumb[i:i + 1], patches[i:i + 1], cls[i], yq[i:i + 1])
            rows.setdefault(fr.index // CHUNK, []).append(rec)
        n_done += len(frames)

    def process_safe(frames: list[io.Frame]) -> None:
        """Process, halving the batch on CUDA OOM (the GPU may be shared)."""
        nonlocal bs
        try:
            process(frames)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(frames) == 1:
                raise
            bs = max(1, len(frames) // 2)
            log.warning("CUDA out of memory; batch size -> %d", bs)
            for i in range(0, len(frames), bs):
                process_safe(frames[i:i + bs])

    last_index = -1
    for fr in io.iter_frames(path, info, hwaccel=hwaccel, start_index=start):
        if fr.index // CHUNK in done and fr.index % CHUNK < CHUNK - max(LAGS):
            continue          # inside an already-analysed chunk and not needed as context
        batch.append(fr)
        last_index = fr.index
        if len(batch) >= bs:
            process_safe(batch)
            batch = []
            flush_chunks(False, last_index + 1)
            if progress and n_done % (bs * 50) == 0:
                el = time.time() - t0
                pct = 100 * last_index / max(info.n_frames, 1)
                log.info("analyse: frame %d/%d (%.1f%%), %.1f fps", last_index, info.n_frames, pct, n_done / el)
    if batch:
        process_safe(batch)
    flush_chunks(True, last_index)
    log.info("analyse: %d frames in %.1fs", n_done, time.time() - t0)
    return store
