"""Pass 2: score frames from stored features, group defects into runs.

Evidence channels (all unsupervised, self-referenced to the same video):

transient  A corrupt frame is far from *both* good neighbours while the
           neighbours are close to each other:  r = log(A / C) per patch with
           A = min(d(t, t-1), d(t, t+1)) and C = d(t-1, t+1).  A scene cut is
           close to one side, so min() suppresses it.  Two-frame runs use the
           (t-1, t+2) bracket.  Computed on pixel and DINO patch distances.
smear      Codec errors persist until the next keyframe: a patch-level step at
           a P/B-frame whose cells step again exactly at the next keyframe.
freeze     Near-zero luma change to the previous frame, relative to the
           local level of change.
intrinsic  Robust temporal z-scores of blockiness, patch novelty (vs. a
           memory bank of the whole video), row/column line score, noise.
decoder    Frames flagged corrupt or with decoder errors.

Output: one row per frame (CSV) plus per-frame patch masks (npz).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .analyze import FeatureStore

log = logging.getLogger(__name__)


@dataclass
class DetectConfig:
    threshold: float = 0.5            # on the combined confidence
    window: int = 61                  # frames for rolling robust statistics
    # (start, saturation, weight) of each evidence channel; calibrated on
    # natural footage so that clean frames stay below "start"
    ramps: dict = field(default_factory=lambda: {
        "trans_pd": (1.25, 2.0, 1.0),   # log(A/C), pixel distances
        "trans_ft": (1.15, 1.8, 1.0),   # log(A/C), DINO patch distances
        "block": (1.0, 3.0, 1.0),       # block-edge ratio above the cell's rolling median
        "novel": (0.65, 0.9, 0.4),      # memory-bank distance above rolling median (support only)
        "row": (5.0, 10.0, 1.0),        # robust z of row-boundary peak
        "col": (6.0, 12.0, 1.0),
        "hf": (5.0, 10.0, 1.0),         # robust z of Laplacian energy
    })
    cell_on: float = 1.0              # log-ratio for a patch to join the mask
    freeze_rel: float = 0.12          # still < freeze_rel * local median change
    freeze_abs: float = 1.5           # ... and below this many 10-bit codes
    max_smear: int = 120              # longest run attributed to keyframe smear
    extend_to_keyframe: bool = False  # assume codec errors propagate to the next keyframe
    dilate: int = 1                   # mask dilation in patches


# --------------------------------------------------------------------------- helpers

def _dense(feats: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Re-index rows onto 0..max(index); missing frames become NaN.

    Patch maps stay float16 to keep whole-video arrays small.
    """
    idx = feats["index"].astype(np.int64)
    n = int(idx.max()) + 1
    present = np.zeros(n, bool)
    present[idx] = True
    out = {}
    for k, v in feats.items():
        if k == "index":
            continue
        shape = (n,) + v.shape[1:]
        if v.dtype == bool:
            a = np.zeros(shape, bool)
        else:
            a = np.full(shape, np.nan, np.float16 if v.ndim > 1 else np.float64)
        a[idx] = v
        out[k] = a
    out["index"] = np.arange(n)
    return out, present


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """b[t] = a[t + k] (NaN outside)."""
    b = np.full_like(a, np.nan, dtype=np.float32)
    if k > 0:
        b[:-k] = a[k:]
    elif k < 0:
        b[-k:] = a[:k]
    else:
        b[:] = a
    return b


def _rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    half = w // 2
    pad = np.pad(x.astype(np.float32), [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    win = sliding_window_view(pad, w, axis=0)
    return fn(win, axis=-1)


def robust_z(x: np.ndarray, w: int, floor: float) -> np.ndarray:
    med = _rolling(x, w, np.nanmedian)
    mad = _rolling(np.abs(x - med), w, np.nanmedian)
    return (x - med) / (1.4826 * mad + floor)


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.nan_to_num(x, nan=0.0) - lo) / (hi - lo), 0.0, 1.0)


def _topk_mean(maps: np.ndarray, frac: float = 0.02, k_min: int = 2) -> np.ndarray:
    flat = maps.reshape(len(maps), -1)
    k = max(k_min, int(flat.shape[1] * frac))
    flat = np.nan_to_num(flat, nan=-np.inf)
    part = -np.partition(-flat, k - 1, axis=1)[:, :k]
    part[~np.isfinite(part)] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(part, axis=1)


# --------------------------------------------------------------------------- channels

def transient(d1, d2, d3, floor):
    """Per-cell log-ratios for 1-frame and 2-frame runs, (N, gh, gw) each."""
    A1 = np.fmin(d1, _shift(d1, 1))
    C1 = _shift(d2, 1)
    r1 = np.log((A1 + floor) / (C1 + floor))
    # run {t, t+1}, bracket (t-1, t+2)
    a_t = np.fmin(d1, _shift(d2, 2))
    a_t1 = np.fmin(_shift(d2, 1), _shift(d1, 2))
    C2 = _shift(d3, 2)
    r2_start = np.log((np.fmin(a_t, a_t1) + floor) / (C2 + floor))
    r2 = np.fmax(r2_start, _shift(r2_start, -1))          # credit both frames of the pair
    return r1, r2


@dataclass
class Detection:
    index: np.ndarray
    time: np.ndarray
    defective: np.ndarray
    confidence: np.ndarray
    defect_type: np.ndarray
    run_id: np.ndarray
    masks: np.ndarray            # (N, gh, gw) bool
    channels: dict[str, np.ndarray]
    present: np.ndarray

    def runs(self) -> list[tuple[int, int, int]]:
        """(run_id, first, last) for each defective run."""
        out = []
        for rid in np.unique(self.run_id[self.run_id >= 0]):
            where = np.flatnonzero(self.run_id == rid)
            out.append((int(rid), int(where[0]), int(where[-1])))
        return out


BLOCK = 2048          # frames scored at once; bounds memory for long videos


def detect(store: FeatureStore, cfg: DetectConfig | None = None) -> Detection:
    cfg = cfg or DetectConfig()
    meta = store.meta()
    f, present = _dense(store.load())
    n = len(f["index"])
    grid = tuple(meta["grid"])
    ten_bit = int(meta["info"]["bit_depth"]) > 8

    # global noise floors for the transient ratios (subsampled for speed)
    step = max(1, n // 4000)
    floors = {"pd": max(float(np.nanmedian(f["pd2"][::step].astype(np.float32))) * 0.5, 0.004),
              "fd": max(float(np.nanmedian(f["fd2"][::step].astype(np.float32))) * 0.5, 0.02)}

    conf = np.zeros(n)
    defective = np.zeros(n, bool)
    types = np.empty(n, dtype=object)
    types[:] = ""
    masks = np.zeros((n, *grid), bool)
    chans: dict[str, np.ndarray] = {}
    pad = max(cfg.window, cfg.max_smear) + 8
    for s0 in range(0, n, BLOCK):
        s1 = min(n, s0 + BLOCK)
        lo, hi = max(0, s0 - pad), min(n, s1 + pad)
        part = {k: v[lo:hi] for k, v in f.items()}
        b = _score_block(part, present[lo:hi], cfg, floors, ten_bit)
        sl, bl = slice(s0, s1), slice(s0 - lo, s1 - lo)
        conf[sl], defective[sl], types[sl], masks[sl] = b["conf"][bl], b["defective"][bl], b["types"][bl], b["masks"][bl]
        for k, v in b["chans"].items():
            chans.setdefault(k, np.full(n, np.nan))[sl] = v[bl]
    run_id = np.full(n, -1)
    rid = -1
    for t in range(n):
        if defective[t]:
            if t == 0 or not defective[t - 1]:
                rid += 1
            run_id[t] = rid
    return Detection(f["index"], np.nan_to_num(f["time"], nan=0), defective, conf, types, run_id,
                     masks, chans, present)


def _score_block(f: dict[str, np.ndarray], present: np.ndarray, cfg: DetectConfig,
                 floors: dict[str, float], ten_bit: bool) -> dict:
    n = len(present)
    w = min(cfg.window, max(5, (n - 1) // 2 * 2 - 1))
    f = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in f.items()}

    # --- transient (pixel + feature) ------------------------------------------------
    chans: dict[str, np.ndarray] = {}
    trans_maps = []
    for key, src in (("pd", "pd"), ("ft", "fd")):
        d1, d2, d3 = (f[f"{src}{k}"] for k in (1, 2, 3))
        r1, r2 = transient(d1, d2, d3, floors[src])
        r = np.fmax(r1, r2)
        trans_maps.append(r)
        chans[f"trans_{key}"] = _topk_mean(r)
    trans_map = np.fmax(*trans_maps)

    # --- intrinsic signals ------------------------------------------------------------
    nov_cell = f["novel"] - _rolling(f["novel"], w, np.nanmedian)
    blk_cell = f["block"] - _rolling(f["block"], w, np.nanmedian)
    chans["novel"] = _topk_mean(nov_cell)
    chans["block"] = _topk_mean(blk_cell)
    chans["row"] = robust_z(f["row_score"], w, 0.15)
    chans["col"] = robust_z(f["col_score"], w, 0.15)
    chans["hf"] = robust_z(f["hf"], w, 1e-3)
    chans["mean_y"] = robust_z(f["mean_y"], w, 0.01)

    # --- decoder -----------------------------------------------------------------------
    s_dec = np.where(f["corrupt"] | (np.nan_to_num(f["decode_errors"]) > 0), 0.6, 0.0)

    # combined confidence: noisy-OR of the ramped channels
    keep = np.ones(n)
    for k, (lo, hi, wgt) in cfg.ramps.items():
        keep *= 1 - wgt * _ramp(chans[k], lo, hi)
    keep *= 1 - s_dec

    # --- freeze ------------------------------------------------------------------------
    still = f["still"] / (1.0 if ten_bit else 0.25)        # express in ~10-bit codes
    level = _rolling(still, w, np.nanmedian)
    frozen = (still < cfg.freeze_rel * level) & (still < cfg.freeze_abs)
    chans["still"] = still

    conf = 1 - keep
    conf = np.where(frozen, np.maximum(conf, 0.9), conf)
    conf[~present] = 0
    defective = conf >= cfg.threshold

    # --- masks ---------------------------------------------------------------------------
    masks = np.nan_to_num(trans_map, nan=0) >= cfg.cell_on
    masks |= np.nan_to_num(nov_cell, nan=0) >= cfg.ramps["novel"][0]
    masks |= np.nan_to_num(blk_cell, nan=0) >= cfg.ramps["block"][0]

    # --- keyframe smear: step at a P/B-frame that reverses at the next keyframe ------------
    key = f["key"].astype(bool)
    keys = np.flatnonzero(key)
    fd1 = np.nan_to_num(f["fd1"], nan=0)
    base = np.nan_to_num(_rolling(f["fd1"], w, np.nanmedian), nan=0)
    step = fd1 - 2.5 * base
    smear = np.zeros(n, bool)
    if len(keys) >= 2:
        for o in np.flatnonzero(defective & ~key):
            if smear[o]:
                continue
            nk = keys[np.searchsorted(keys, o, side="right")] if keys[-1] > o else None
            if nk is None or nk - o > cfg.max_smear or nk - o < 2:
                continue
            m = (step[o] > 0.05) | masks[o]
            if m.sum() < 1 or m.mean() > 0.9:
                continue
            onset = step[o][m].mean()
            release = step[nk][m].mean()
            outside = step[nk][~m].mean() if (~m).any() else 0.0   # scene cuts change everything
            between = fd1[o + 1:nk][:, m].mean() if nk - o > 1 else 0
            if onset > 0.05 and release > 0.05 and release > 2 * between and release > 2 * max(outside, 0) + 0.05:
                smear[o:nk] = True
                masks[o:nk] |= m
    if cfg.extend_to_keyframe and len(keys):
        # bitstream damage in a reference frame persists, often too faintly to
        # see frame by frame, until the decoder resets at the next keyframe
        z = {k: np.nan_to_num(chans[k]) for k in ("mean_y", "row", "col", "hf")}
        codec_like = (defective & ~frozen & (np.abs(z["mean_y"]) < 6)
                      & (np.maximum(z["row"], z["col"]) < cfg.ramps["row"][0]) & (z["hf"] < cfg.ramps["hf"][0]))
        for o in np.flatnonzero(codec_like & ~key & ~smear):
            nxt = keys[keys > o]
            if not len(nxt) or nxt[0] - o > cfg.max_smear:
                continue
            nk = nxt[0]
            m = masks[o] if masks[o].any() else np.ones_like(masks[o])
            for t in range(o, nk):
                grow = (t - o) // 8                       # errors drift with motion
                masks[t] |= _dilate(m[None], grow)[0] if grow else m
            smear[o:nk] = True
    defective |= smear
    conf = np.where(smear, np.maximum(conf, 0.75), conf)

    # --- labels ----------------------------------------------------------------------------
    types = np.empty(n, dtype=object)
    types[:] = ""
    z = {k: np.nan_to_num(v) for k, v in chans.items()}
    for t in np.flatnonzero(defective):
        if frozen[t]:
            types[t] = "freeze"
        elif z["mean_y"][t] < -6 and f["dark_frac"][t] > 0.5:
            types[t] = "black"
        elif z["mean_y"][t] > 6 and f["bright_frac"][t] > 0.2 or z["mean_y"][t] > 10:
            types[t] = "flash"
        elif smear[t]:
            types[t] = "smear"
        elif z["block"][t] > cfg.ramps["block"][0]:
            types[t] = "blocks"
        elif max(z["row"][t], z["col"][t]) > cfg.ramps["row"][0]:
            types[t] = "lines"
        elif z["hf"][t] > cfg.ramps["hf"][0]:
            types[t] = "noise"
        elif s_dec[t] > 0:
            types[t] = "decode"
        else:
            types[t] = "glitch"
        if types[t] in ("freeze", "black", "flash", "noise", "lines"):
            masks[t] = True
    if cfg.dilate:
        masks = _dilate(masks, cfg.dilate)
    masks[~defective] = False
    return {"conf": conf, "defective": defective, "types": types, "masks": masks, "chans": chans}



def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation of (N, gh, gw) masks by a (2r+1)^2 square, no wrap-around."""
    gh, gw = m.shape[1:]
    p = np.pad(m, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(m)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[:, dy:dy + gh, dx:dx + gw]
    return out


# --------------------------------------------------------------------------- output

CSV_COLUMNS = ["frame_num", "pts_time", "defective", "confidence", "defect_type", "mask_area", "run_id"]


def write_csv(det: Detection, path: Path, all_frames: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_COLUMNS)
        for t in range(len(det.index)):
            if not all_frames and not det.defective[t]:
                continue
            wr.writerow([t, f"{det.time[t]:.6f}", int(det.defective[t]), f"{det.confidence[t]:.3f}",
                         det.defect_type[t], f"{det.masks[t].mean():.3f}", int(det.run_id[t])])


def save_masks(det: Detection, path: Path) -> None:
    idx = np.flatnonzero(det.defective)
    np.savez_compressed(path, index=idx, masks=det.masks[idx], run_id=det.run_id[idx],
                        defect_type=det.defect_type[idx].astype(str))


def load_csv(path: Path) -> dict[int, dict]:
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if int(row["defective"]):
                out[int(row["frame_num"])] = row
    return out


def load_detection(csv_path: Path, masks_path: Path | None, grid: tuple[int, int]) -> Detection:
    """Rebuild a Detection from a (possibly hand-edited) CSV and its mask file.

    Frames marked defective in the CSV without a stored mask get a full mask;
    runs are recomputed from the ``defective`` column.
    """
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    n = max(int(r["frame_num"]) for r in rows) + 1
    defective = np.zeros(n, bool)
    conf = np.zeros(n)
    times = np.zeros(n)
    types = np.empty(n, dtype=object)
    types[:] = ""
    for r in rows:
        i = int(r["frame_num"])
        defective[i] = bool(int(r["defective"]))
        conf[i] = float(r["confidence"] or 0)
        times[i] = float(r["pts_time"] or 0)
        types[i] = r["defect_type"] or ("glitch" if defective[i] else "")
    masks = np.zeros((n, *grid), bool)
    if masks_path and Path(masks_path).exists():
        z = np.load(masks_path, allow_pickle=False)
        for i, m in zip(z["index"], z["masks"]):
            if i < n:
                masks[i] = m
    for i in np.flatnonzero(defective & ~masks.reshape(n, -1).any(1)):
        masks[i] = True
    run_id = np.full(n, -1)
    rid = -1
    for t in range(n):
        if defective[t]:
            if t == 0 or not defective[t - 1]:
                rid += 1
            run_id[t] = rid
    return Detection(np.arange(n), times, defective, conf, types, run_id, masks, {}, np.ones(n, bool))
