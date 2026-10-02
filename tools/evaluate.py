#!/usr/bin/env python3
"""Score a detection CSV against a ground-truth CSV from make_synthetic.py.

    python tools/evaluate.py truth.csv defective_frames.csv [--clean clean.mp4 --repaired out.mp4]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_truth(path: Path) -> dict[int, str]:
    with open(path) as fh:
        return {int(r["frame_num"]): r["defect_type"] for r in csv.DictReader(fh)}


def load_det(path: Path) -> dict[int, dict]:
    with open(path) as fh:
        return {int(r["frame_num"]): r for r in csv.DictReader(fh) if int(r["defective"])}


def detection_report(truth: dict[int, str], det: dict[int, dict]) -> dict:
    tp = set(truth) & set(det)
    fp = set(det) - set(truth)
    fn = set(truth) - set(det)
    prec = len(tp) / max(1, len(det))
    rec = len(tp) / max(1, len(truth))
    print(f"frames: truth={len(truth)} detected={len(det)} TP={len(tp)} FP={len(fp)} FN={len(fn)}")
    print(f"precision={prec:.3f} recall={rec:.3f} F1={2 * prec * rec / max(1e-9, prec + rec):.3f}")
    per = Counter(truth.values())
    hit = Counter(truth[i] for i in tp)
    for t in sorted(per):
        print(f"  {t:10s} recall {hit[t]}/{per[t]}")
    if fn:
        print("missed:", ", ".join(f"{i}({truth[i]})" for i in sorted(fn)))
    if fp:
        print("false positives:", ", ".join(f"{i}({det[i]['defect_type']},{det[i]['confidence']})" for i in sorted(fp)))
    return {"precision": prec, "recall": rec}


def repair_report(truth: dict[int, str], clean: Path, broken: Path, repaired: Path) -> None:
    from video_repair import io

    def lumas(p):
        info = io.probe(p)
        s = 6 if info.bit_depth > 8 else 0
        peak = 1023 if info.bit_depth > 8 else 255
        return {f.index: (f.y >> s).astype(np.float32) for f in io.iter_frames(p, info, hwaccel=False)
                if f.index in truth}, peak

    ref, peak = lumas(clean)
    rows = []
    for name, p in (("broken", broken), ("repaired", repaired)):
        got, _ = lumas(p)
        ps = []
        for i in sorted(truth):
            if i in ref and i in got:
                mse = float(((got[i] - ref[i]) ** 2).mean())
                ps.append(99.0 if mse == 0 else 10 * np.log10(peak ** 2 / mse))
        rows.append((name, ps))
    for name, ps in rows:
        print(f"PSNR on defective frames vs clean: {name:9s} mean {np.mean(ps):.2f} dB  median {np.median(ps):.2f} dB")
    by_type: dict[str, list] = {}
    for (i, (b, r)) in zip(sorted(truth), zip(rows[0][1], rows[1][1])):
        by_type.setdefault(truth[i], []).append((b, r))
    for t, v in sorted(by_type.items()):
        v = np.array(v)
        print(f"  {t:10s} broken {v[:, 0].mean():6.2f} dB -> repaired {v[:, 1].mean():6.2f} dB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", type=Path)
    ap.add_argument("detected", type=Path)
    ap.add_argument("--clean", type=Path)
    ap.add_argument("--broken", type=Path)
    ap.add_argument("--repaired", type=Path)
    a = ap.parse_args()
    truth = load_truth(a.truth)
    detection_report(truth, load_det(a.detected))
    if a.clean and a.broken and a.repaired:
        repair_report(truth, a.clean, a.broken, a.repaired)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
