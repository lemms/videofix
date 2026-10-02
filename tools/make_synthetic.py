#!/usr/bin/env python3
"""Inject known defects into a clean clip and write a ground-truth CSV.

Pixel mode (default) edits decoded frames and re-encodes:
  black, flash, freeze, blocks (green/smeared macroblocks that persist until the
  next keyframe), tear (horizontally shifted lower part), lines, noise.

Bitstream mode (--bitstream) flips bytes inside compressed packets, which gives
genuine decoder corruption; ground truth is every frame whose decode differs
from the clean decode by more than --psnr-threshold dB.

    python tools/make_synthetic.py clean.mp4 broken.mp4 --truth truth.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_repair import io  # noqa: E402

TYPES = ["black", "flash", "freeze", "blocks", "tear", "lines", "noise"]


class Planes:
    """Helpers on native semi-planar frames with values in code units."""

    def __init__(self, info: io.VideoInfo):
        self.shift = 6 if info.bit_depth > 8 else 0
        self.max = 1023 if info.bit_depth > 8 else 255
        self.mid = (self.max + 1) // 2
        self.lo = 64 if info.bit_depth > 8 else 16
        self.hi = 940 if info.bit_depth > 8 else 235

    def get(self, fr: io.Frame) -> tuple[np.ndarray, np.ndarray]:
        return (fr.y.astype(np.int32) >> self.shift), (fr.uv.astype(np.int32) >> self.shift)

    def put(self, fr: io.Frame, y: np.ndarray, uv: np.ndarray) -> None:
        dt = fr.y.dtype
        fr.y = (np.clip(y, 0, self.max).astype(np.int32) << self.shift).astype(dt)
        fr.uv = (np.clip(uv, 0, self.max).astype(np.int32) << self.shift).astype(dt)


def _region(rng: random.Random, h: int, w: int, area: tuple[float, float]) -> tuple[int, int, int, int]:
    a = rng.uniform(*area)
    rw = int(w * min(1.0, np.sqrt(a) * rng.uniform(0.7, 1.4))) // 16 * 16 or 16
    rh = int(h * w * a / max(rw, 1)) // 16 * 16 or 16
    rh = min(rh, h // 16 * 16)
    x0 = rng.randrange(0, max(1, w - rw + 1), 16)
    y0 = rng.randrange(0, max(1, h - rh + 1), 16)
    return x0, y0, min(w, x0 + rw), min(h, y0 + rh)


def _blockify(a: np.ndarray, b: int) -> np.ndarray:
    """Replace each bxb block by its mean (DC-only macroblocks)."""
    h, w = a.shape[:2]
    hh, ww = h // b * b, w // b * b
    out = a.copy()
    v = a[:hh, :ww].reshape(hh // b, b, ww // b, b, *a.shape[2:]).mean(axis=(1, 3), keepdims=True)
    out[:hh, :ww] = np.broadcast_to(v, (hh // b, b, ww // b, b, *a.shape[2:])).reshape(hh, ww, *a.shape[2:])
    return out


def pixel_mode(src: Path, dst: Path, truth: Path, seed: int, density: float, types: list[str],
               gop: int, max_run: int) -> None:
    rng = random.Random(seed)
    info = io.probe(src)
    P = Planes(info)
    n = info.n_frames
    events: dict[int, tuple[str, int, int]] = {}      # frame -> (type, run_id, run_len)
    params: dict[int, dict] = {}
    run_id, i = 0, 10
    while i < n - 10:
        i += rng.randint(int(1 / density * 0.5), int(1 / density * 1.5))
        if i >= n - 10:
            break
        t = rng.choice(types)
        if t == "freeze":
            length = rng.randint(1, min(4, max_run))
        elif t == "blocks":
            to_key = gop - (i % gop)                   # smear persists until the next keyframe
            length = min(to_key, max_run, n - 10 - i) if rng.random() < 0.6 else rng.randint(1, 2)
        elif t == "tear":
            length = rng.randint(1, 2)
        else:
            length = 1
        length = max(1, length)
        for k in range(length):
            events[i + k] = (t, run_id, length)
        params[run_id] = {"t": t, "start": i, "len": length, "seed": rng.random(),
                          "region": _region(rng, info.height, info.width, (0.03, 0.35)),
                          "green": rng.random() < 0.5, "row": rng.randint(info.height // 5, info.height * 4 // 5),
                          "shift": rng.choice([-1, 1]) * rng.randint(24, 96)}
        run_id += 1
        i += length + 15

    prev: list[io.Frame] = []
    rows = []
    with io.Encoder(dst, info, quality=16) as enc:
        for fr in io.iter_frames(src, info):
            ev = events.get(fr.index)
            orig = io.Frame(**{**fr.__dict__, "y": fr.y.copy(), "uv": fr.uv.copy()})
            if ev:
                t, rid, length = ev
                p = params[rid]
                frng = np.random.default_rng(int(p["seed"] * 1e9) + fr.index)
                y, uv = P.get(fr)
                h, w = y.shape
                if t == "black":
                    y = P.lo + frng.normal(0, 2, y.shape)
                    uv = np.full_like(uv, P.mid)
                elif t == "flash":
                    y = np.minimum(P.hi, y * 0.3 + P.hi * 0.8)
                    uv = P.mid + (uv - P.mid) * 0.2
                elif t == "freeze":
                    src_fr = prev_clean_before(prev, p["start"])
                    y, uv = P.get(src_fr)
                elif t == "blocks":
                    x0, y0, x1, y1 = p["region"]
                    k = fr.index - p["start"]
                    # region grows a little as the error propagates through P-frames
                    grow = 16 * min(k, 6)
                    x0, y0 = max(0, x0 - grow), max(0, y0 - grow)
                    x1, y1 = min(w, x1 + grow), min(h, y1 + grow)
                    ys, xs = slice(y0, y1), slice(x0, x1)
                    cs = (slice(y0 // 2, y1 // 2), slice(x0, x1))
                    if p["green"]:
                        y[ys, xs] = _blockify(y[ys, xs] * 0.35 + P.lo, 16)
                        uv[cs] = (P.mid - 0.30 * P.max) + frng.normal(0, 4, uv[cs].shape)
                    else:
                        stale = prev_clean_before(prev, p["start"] - 8) if prev else orig
                        sy, suv = P.get(stale)
                        dx = int(16 * (1 + k))
                        y[ys, xs] = _blockify(np.roll(sy, dx, axis=1)[ys, xs], 8)
                        uv[cs] = _blockify(np.roll(suv, dx, axis=1)[cs], 8)
                elif t == "tear":
                    r = p["row"] // 2 * 2
                    y[r:] = np.roll(y[r:], p["shift"], axis=1)
                    uv[r // 2:] = np.roll(uv[r // 2:], p["shift"] // 2 * 2, axis=1)
                elif t == "lines":
                    for _ in range(int(frng.integers(3, 12))):
                        r = int(frng.integers(0, h - 4))
                        th = int(frng.integers(1, 4))
                        y[r:r + th] = frng.choice([P.lo, P.hi]) + frng.normal(0, 30, (th, w))
                elif t == "noise":
                    y = y + frng.normal(0, 0.12 * P.max, y.shape)
                    uv = uv + frng.normal(0, 0.05 * P.max, uv.shape)
                P.put(fr, y, uv)
                x0, y0, x1, y1 = params[rid]["region"] if t == "blocks" else (0, 0, info.width, info.height)
                rows.append([fr.index, t, rid, length, x0 / info.width, y0 / info.height,
                             x1 / info.width, y1 / info.height])
            enc.write(fr)
            prev.append(orig)
            del prev[:-16]
    with open(truth, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame_num", "defect_type", "run_id", "run_len", "x0", "y0", "x1", "y1"])
        wr.writerows(rows)
    print(f"{len(rows)} defective frames in {len(params)} events -> {dst}, truth {truth}")


def prev_clean_before(prev: list[io.Frame], index: int) -> io.Frame:
    cands = [f for f in prev if f.index < index]
    return cands[-1] if cands else prev[0]


def bitstream_mode(src: Path, dst: Path, truth: Path, seed: int, density: float, psnr_thr: float) -> None:
    import av
    rng = random.Random(seed)
    inp = av.open(str(src))
    vin = inp.streams.video[0]
    out = av.open(str(dst), "w")
    vout = out.add_stream_from_template(vin)
    n_hit = 0
    for i, pkt in enumerate(inp.demux(vin)):
        if pkt.size == 0:
            continue
        if not pkt.is_keyframe and i > 5 and rng.random() < density:
            data = bytearray(bytes(pkt))
            for _ in range(rng.randint(1, 4)):
                pos = rng.randrange(min(len(data) - 1, 32), len(data))
                data[pos] ^= 1 << rng.randrange(8)
            new = av.Packet(bytes(data))
            new.pts, new.dts, new.time_base = pkt.pts, pkt.dts, pkt.time_base
            new.stream = vout
            out.mux(new)
            n_hit += 1
        else:
            pkt.stream = vout
            out.mux(pkt)
    out.close()
    inp.close()
    # ground truth: frames that decode differently from the clean stream
    info = io.probe(src)
    shift = 6 if info.bit_depth > 8 else 0
    peak = 1023 if info.bit_depth > 8 else 255
    clean = {f.index: (f.y >> shift).astype(np.float32) for f in io.iter_frames(src, info, hwaccel=False)}
    rows = []
    for f in io.iter_frames(dst, io.probe(dst), hwaccel=False):
        ref = clean.pop(f.index, None)
        if ref is None:
            continue
        mse = float(((f.y >> shift).astype(np.float32) - ref).__pow__(2).mean())
        psnr = 99.0 if mse == 0 else 10 * np.log10(peak ** 2 / mse)
        if psnr < psnr_thr:
            rows.append([f.index, "bitstream", -1, 0, 0, 0, 1, 1, round(psnr, 2)])
    for idx in sorted(clean):   # frames that could not be decoded at all
        rows.append([idx, "missing", -1, 0, 0, 0, 1, 1, 0.0])
    with open(truth, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame_num", "defect_type", "run_id", "run_len", "x0", "y0", "x1", "y1", "psnr"])
        wr.writerows(sorted(rows))
    print(f"corrupted {n_hit} packets; {len(rows)} frames differ (<{psnr_thr} dB) -> {dst}, truth {truth}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=0.025, help="events per frame (pixel) / packet hit rate (bitstream)")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--gop", type=int, default=50, help="keyframe interval of the source, for smear length")
    ap.add_argument("--max-run", type=int, default=12)
    ap.add_argument("--bitstream", action="store_true")
    ap.add_argument("--psnr-threshold", type=float, default=35.0)
    a = ap.parse_args()
    if a.bitstream:
        bitstream_mode(a.src, a.dst, a.truth, a.seed, a.density, a.psnr_threshold)
    else:
        pixel_mode(a.src, a.dst, a.truth, a.seed, a.density, a.types.split(","), a.gop, a.max_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
