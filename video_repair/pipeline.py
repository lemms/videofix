"""Command-line interface.

    video-repair run INPUT -o OUTPUT          # analyse + detect + repair
    video-repair analyze INPUT                # pass 1 only (resumable)
    video-repair detect INPUT                 # CSV + masks + review page
    video-repair repair INPUT -o OUTPUT       # uses the (editable) CSV
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("video_repair")


def _work_dir(args) -> Path:
    # default to the current directory, never next to the input: originals
    # often live on archive drives that should stay untouched
    return Path(args.work) if args.work else Path.cwd() / (args.input.stem + "_vrwork")


def _paths(args) -> dict[str, Path]:
    w = _work_dir(args)
    return {"work": w, "features": w / "features", "masks": w / "masks.npz",
            "csv": Path(args.csv) if getattr(args, "csv", None) else w / "defective_frames.csv",
            "review": w / "review"}


def _device(args) -> str:
    import torch
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def cmd_analyze(args) -> int:
    from .analyze import analyze
    p = _paths(args)
    analyze(args.input, p["features"], backbone_name=args.backbone, hwaccel=not args.no_hwaccel,
            device=_device(args))
    return 0


def _detect(args):
    from .analyze import FeatureStore
    from .detect import DetectConfig, detect, save_masks, write_csv
    p = _paths(args)
    cfg = DetectConfig(threshold=args.threshold, extend_to_keyframe=args.extend_to_keyframe)
    det = detect(FeatureStore(p["features"]), cfg)
    if args.labels:
        det = _calibrate(det, args.labels, cfg)
    write_csv(det, p["csv"])
    save_masks(det, p["masks"])
    runs = det.runs()
    by_type: dict[str, int] = {}
    for t in det.defect_type[det.defective]:
        by_type[t] = by_type.get(t, 0) + 1
    log.info("detect: %d defective frames in %d runs %s -> %s", int(det.defective.sum()), len(runs), by_type, p["csv"])
    return det


def _calibrate(det, labels_path: Path, cfg):
    """Pick the confidence threshold that maximises F1 on hand-labelled frames."""
    with open(labels_path) as fh:
        lab = {int(r["frame_num"]): bool(int(r["defective"])) for r in csv.DictReader(fh)}
    idx = np.array([i for i in lab if i < len(det.confidence)])
    y = np.array([lab[i] for i in idx])
    best = (cfg.threshold, -1.0)
    for thr in np.linspace(0.05, 0.95, 91):
        pred = det.confidence[idx] >= thr
        tp = (pred & y).sum()
        f1 = 2 * tp / max(1, pred.sum() + y.sum())
        if f1 > best[1]:
            best = (float(thr), float(f1))
    log.info("calibration on %d labelled frames: threshold %.2f (F1 %.3f)", len(idx), *best)
    return _rethreshold(det, best[0])


def _rethreshold(det, thr: float):
    """Apply a new threshold; frames that become defective get a full mask."""
    newly = (det.confidence >= thr) & ~det.defective
    det.defective = det.confidence >= thr
    det.masks[newly] = True
    det.masks[~det.defective] = False
    rid, run = -1, np.full(len(det.defective), -1)
    for t in range(len(det.defective)):
        if det.defective[t]:
            if t == 0 or not det.defective[t - 1]:
                rid += 1
            run[t] = rid
    det.run_id = run
    return det


def cmd_detect(args) -> int:
    from . import io
    from .review import review_detections
    det = _detect(args)
    if not args.no_review:
        page = review_detections(args.input, io.probe(args.input), det, _paths(args)["review"],
                                 hwaccel=not args.no_hwaccel)
        log.info("review page: %s", page)
    return 0


def cmd_repair(args, det=None) -> int:
    from . import io
    from .analyze import FeatureStore
    from .detect import load_detection
    from .repair import RepairConfig, repair
    from .review import ReviewWriter
    p = _paths(args)
    info = io.probe(args.input)
    if det is None:
        grid = tuple(FeatureStore(p["features"]).meta()["grid"]) if (p["features"] / "meta.json").exists() else (14, 24)
        det = load_detection(p["csv"], p["masks"], grid)
    cfg = RepairConfig(max_gap=args.max_gap, interpolator=args.interpolator)
    rw = None if args.no_review else ReviewWriter(p["review"], info, det)

    def on_frame(fr, before):
        if rw is not None:
            rw.add(fr, before)

    status = repair(args.input, det, args.output, cfg, device=_device(args), hwaccel=not args.no_hwaccel,
                    on_frame=on_frame, quality=args.quality)
    # record what happened to each defective frame
    rows = list(csv.DictReader(open(p["csv"])))
    with open(p["csv"], "w", newline="") as fh:
        cols = list(rows[0].keys()) if rows else []
        if "repair" not in cols:
            cols.append("repair")
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        for r in rows:
            r["repair"] = status.get(int(r["frame_num"]), "")
            wr.writerow(r)
    if rw is not None:
        page = rw.write(f"Repair · {args.input.name}", status)
        log.info("review page: %s", page)
    n_fix = sum(1 for s in status.values() if not s.startswith("unrepaired"))
    log.info("done: %d frames repaired, %d left unrepaired -> %s", n_fix, len(status) - n_fix, args.output)
    return 0


def cmd_run(args) -> int:
    cmd_analyze(args)
    det = _detect(args)
    return cmd_repair(args, det)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="video-repair", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, output=False):
        sp.add_argument("input", type=Path)
        if output:
            sp.add_argument("-o", "--output", type=Path, required=True)
        sp.add_argument("--work", help="work directory (default: ./<input stem>_vrwork)")
        sp.add_argument("--csv", help="detection CSV path (default: <work>/defective_frames.csv)")
        sp.add_argument("--device", default="auto", help="auto | cuda | cpu")
        sp.add_argument("--no-hwaccel", action="store_true", help="software decoding (slower, more decoder error reports)")
        sp.add_argument("-v", "--verbose", action="store_true")

    def det_opts(sp):
        sp.add_argument("--backbone", default="auto",
                        help="auto (DINOv2 with registers, else DINOv2) | dinov2reg | dinov2 | dinov3 (gated)")
        sp.add_argument("--threshold", type=float, default=0.5, help="defect confidence threshold")
        sp.add_argument("--labels", type=Path, help="CSV frame_num,defective to calibrate the threshold")
        sp.add_argument("--extend-to-keyframe", action="store_true",
                        help="treat codec errors as persisting until the next keyframe (bitstream damage)")
        sp.add_argument("--no-review", action="store_true", help="skip the HTML review page")

    def rep_opts(sp):
        sp.add_argument("--max-gap", type=int, default=12, help="longest defective run to interpolate")
        sp.add_argument("--interpolator", default="rife", choices=["rife"])
        sp.add_argument("--quality", type=int, default=18, help="encoder CQ/CRF (lower = better)")
        if "--no-review" not in sp._option_string_actions:
            sp.add_argument("--no-review", action="store_true")

    sp = sub.add_parser("analyze", help="pass 1: extract features (resumable)")
    common(sp)
    sp.add_argument("--backbone", default="auto")
    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("detect", help="pass 2: score frames, write CSV/masks/review")
    common(sp)
    det_opts(sp)
    sp.set_defaults(fn=cmd_detect)

    sp = sub.add_parser("repair", help="pass 3: interpolate defective runs and re-encode")
    common(sp, output=True)
    rep_opts(sp)
    sp.set_defaults(fn=cmd_repair)

    sp = sub.add_parser("run", help="analyze + detect + repair")
    common(sp, output=True)
    det_opts(sp)
    rep_opts(sp)
    sp.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.input.exists():
        ap.error(f"input not found: {args.input}")
    if getattr(args, "output", None) and args.output.resolve() == args.input.resolve():
        ap.error("output must differ from input")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
