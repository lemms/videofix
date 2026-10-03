"""Repair defect runs that touch a chapter boundary.

Cameras such as GoPros split one recording into chapter files (``GOPR0008``,
``GP010008``, ``GP020008`` ...).  A defect run at the very end of a chapter has
no good frame after it *in that file*, so the per-file repair leaves it
(``unrepaired:edge``).  But the frame after it is the first frame of the next
chapter.  This module pairs consecutive chapters of the same recording and
interpolates across the boundary, then rewrites only the affected chapters.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import io
from .detect import Detection
from .repair import Plan, RepairConfig, Repairer

log = logging.getLogger(__name__)

_GOPRO = re.compile(r"^G[OPHX]([A-Z0-9]{2})(\d{4})$", re.I)


def recording_key(path: Path) -> tuple[str, int] | None:
    """('0008', chapter) for GoPro names (GOPR0008 = chapter 0, GP010008 = chapter 1); else None."""
    m = _GOPRO.match(path.stem)
    if not m:
        return None
    ch = m.group(1).upper()
    return m.group(2), 0 if ch in ("PR", "OP") else int(ch) if ch.isdigit() else 0


def recordings(files: list[Path]) -> list[list[Path]]:
    """Group chapter files by recording, each list in chapter order."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for f in files:
        k = recording_key(f)
        if k:
            groups.setdefault(k[0], []).append((k[1], f))
    return [[f for _, f in sorted(v)] for _, v in sorted(groups.items())]


@dataclass
class Chapter:
    video: Path                  # file to read (and rewrite, if ``writable``)
    csv: Path | None             # detection CSV with a 'repair' column
    writable: bool = True
    frames: int = 0
    head: tuple[int, int] | None = None     # unrepaired run at the start (first, last)
    tail: tuple[int, int] | None = None     # unrepaired run at the end
    replacements: dict[int, io.Frame] = field(default_factory=dict)


def edge_runs(csv_path: Path | None, n_frames: int) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Unrepaired runs touching the first / last frame, from a detection CSV."""
    if csv_path is None or not Path(csv_path).exists():
        return None, None
    bad = set()
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("defective") == "1" and r.get("repair", "").startswith("unrepaired"):
                bad.add(int(r["frame_num"]))
    head = tail = None
    if 0 in bad:
        e = 0
        while e + 1 in bad:
            e += 1
        head = (0, e)
    if n_frames - 1 in bad:
        s = n_frames - 1
        while s - 1 in bad:
            s -= 1
        tail = (s, n_frames - 1)
    return head, tail


def plan_boundaries(chapters: list[Chapter]) -> list[tuple[int, int]]:
    """Pairs (i, i+1) of consecutive chapters with a run at their shared boundary."""
    return [(i, i + 1) for i in range(len(chapters) - 1)
            if chapters[i].tail is not None or chapters[i + 1].head is not None]


def _synthetic_detection(n: int, info: io.VideoInfo) -> Detection:
    """Whole-frame 'streak' repair for every frame (boundary runs are full-frame)."""
    grid = (4, 4)
    return Detection(np.arange(n), np.zeros(n), np.ones(n, bool), np.ones(n), np.array(["streak"] * n, dtype=object),
                     np.zeros(n, int), np.ones((n, *grid), bool), {}, np.ones(n, bool))


def fix_recording(chapters: list[Chapter], cfg: RepairConfig | None = None, device: str = "cuda",
                  hwaccel: bool = True) -> dict:
    """Interpolate across chapter boundaries and rewrite the affected chapters in place.

    Each rewritten file is replaced only after the new version has the same
    frame count; returns a report."""
    cfg = cfg or RepairConfig()
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    for ch in chapters:
        info = io.probe(ch.video)
        ch.frames = info.n_frames
        ch.head, ch.tail = edge_runs(ch.csv, ch.frames)
    pairs = plan_boundaries(chapters)
    report = {"boundaries": [], "rewritten": []}
    if not pairs:
        return report
    repairer = None
    for i, j in pairs:
        A, B = chapters[i], chapters[j]
        info = io.probe(A.video)
        tail = A.tail or (A.frames, A.frames - 1)          # empty range if no tail run
        head = B.head or (0, -1)
        a_idx, b_idx = tail[0] - 1, head[1] + 1
        if a_idx < 0 or b_idx >= B.frames:
            report["boundaries"].append({"pair": [A.video.name, B.video.name], "status": "unrepaired:whole-chapter"})
            continue
        if (tail[0] <= tail[1] and not A.writable) or (head[0] <= head[1] and not B.writable):
            report["boundaries"].append({"pair": [A.video.name, B.video.name], "status": "skipped:read-only"})
            continue
        want_a = set(range(tail[0], tail[1] + 1)) | {a_idx}
        want_b = set(range(head[0], head[1] + 1)) | {b_idx}
        fa = io.read_frames(A.video, want_a, info, hwaccel)
        fb = io.read_frames(B.video, want_b, io.probe(B.video), hwaccel)
        off = A.frames                                     # B's frames continue A's numbering
        run = [fa[k] for k in range(tail[0], tail[1] + 1)]
        for k in range(head[0], head[1] + 1):
            f = fb[k]
            run.append(io.Frame(**{**f.__dict__, "index": f.index + off}))
        ref_b = io.Frame(**{**fb[b_idx].__dict__, "index": b_idx + off})
        n_total = off + B.frames
        if repairer is None:
            repairer = Repairer(info, _synthetic_detection(n_total, info), cfg, dev)
        else:
            repairer.det = _synthetic_detection(n_total, info)
        plan = Plan(run_id=0, first=run[0].index, last=run[-1].index, a=a_idx, b=b_idx + off, status="interpolate")
        motion = repairer.motion(fa[a_idx], ref_b)
        fixed = repairer.run_frames(run, fa[a_idx], ref_b, plan)
        for f in fixed:
            if f.index >= off:
                B.replacements[f.index - off] = io.Frame(**{**f.__dict__, "index": f.index - off})
            else:
                A.replacements[f.index] = f
        report["boundaries"].append({"pair": [A.video.name, B.video.name], "status": "interpolated",
                                     "frames": len(run), "tail": tail[1] - tail[0] + 1, "head": head[1] - head[0] + 1,
                                     "motion": round(motion, 2)})
        log.info("boundary %s | %s: %d frames interpolated (motion %.1f)", A.video.name, B.video.name, len(run), motion)
    for ch in chapters:
        if ch.replacements:
            rewrite(ch.video, ch.replacements, hwaccel=hwaccel, hw_encode=hwaccel)
            report["rewritten"].append({"file": str(ch.video), "frames_replaced": len(ch.replacements)})
    return report


def rewrite(path: Path, replacements: dict[int, io.Frame], hwaccel: bool = True, quality: int = 18,
            hw_encode: bool = True) -> None:
    """Re-encode ``path`` with some frames replaced; swap it in only if complete."""
    path = Path(path)
    info = io.probe(path)
    tmp_video = path.with_name(path.stem + ".boundary.video.tmp.mp4")
    tmp = path.with_name(path.stem + ".boundary.partial" + path.suffix)
    n = 0
    with io.Encoder(tmp_video, info, quality=quality, hw=hw_encode) as enc:
        for fr in io.iter_frames(path, info, hwaccel=hwaccel):
            rep = replacements.get(fr.index)
            if rep is not None:
                fr.y, fr.uv = rep.y, rep.uv
            enc.write(fr)
            n += 1
    for w in io.mux(tmp_video, path, tmp):
        log.warning(w)
    tmp_video.unlink(missing_ok=True)
    out = io.probe(tmp)
    if out.n_frames != info.n_frames or n != info.n_frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"rewrite of {path.name} incomplete ({out.n_frames}/{info.n_frames} frames); kept original")
    os.replace(tmp, path)
