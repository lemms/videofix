#!/usr/bin/env python3
"""Detect and repair defects in many videos, resumably.

    python tools/batch_repair.py --out ~/FixedVideos --work ~/FixedVideos/.work \\
        --source Mehndi=/media/Storage_Small/Mehndi --source Wedding=/media/Storage/Wedding

For every video in ``<source>/<camera>/`` (camera folders matching --cameras)
it analyses, detects and, only if defects were found, writes the repaired copy
to ``<out>/<event>/<camera>/<original name>`` -- or, with ``--beside``, to
``<source>/Fixed/<camera>/<original name>`` on the same drive.  Clean videos are
only listed.

With ``--delete-source`` the original is deleted after a successful repair, but
only if the repaired file has the same frame count and duration as the original
*and* the file appears with identical MD5 checksums in both ``--local-md5`` and
``--backup-md5`` manifests (i.e. a verified backup exists).

* Resumable: finished videos are recorded in ``<work>/status.json`` and skipped
  on the next run; partial outputs are never left under their final name.
* Space-aware: before repairing, it checks the output disk has room for the
  repaired copy plus a margin; if not, it stops cleanly and says so.
* Without --delete-source, originals are only read, never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

PY = sys.executable
VIDEO = re.compile(r"\.(mp4|mov)$", re.I)


def gopro_key(p: Path):
    """GoPro chapters: GOPRxxxx is chapter 0 of recording xxxx, GPnnxxxx chapter nn."""
    m = re.match(r"G[OP]([A-Z\d]{2})(\d{4})", p.stem.upper())
    if m:
        ch = 0 if m.group(1) == "PR" else int(m.group(1)) if m.group(1).isdigit() else 99
        return (int(m.group(2)), ch, p.name)
    return (10**9, 0, p.name)


def list_jobs(sources: list[tuple[str, Path]], cameras: str, skip: set[str]):
    jobs = []
    for event, root in sources:
        for cam in sorted((d for d in root.iterdir() if d.is_dir() and re.fullmatch(cameras, d.name)),
                          key=lambda d: (len(d.name), d.name)):
            for f in sorted((f for f in cam.rglob("*") if f.is_file() and VIDEO.search(f.name)), key=gopro_key):
                if "_fixed" in f.stem or f.name in skip:
                    continue
                jobs.append((event, cam.name, f))
    return jobs


class Status:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key):
        with self.lock:
            return self.data.get(key)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(self.path)


def run(cmd: list[str], log: Path) -> None:
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[3]} failed (exit {r.returncode}); see {log}")


def load_md5(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        for line in open(Path(p).expanduser(), encoding="utf-8", errors="surrogateescape"):
            m = re.match(r"^([0-9a-f]{32}) [ *](.+)$", line.rstrip("\n"))
            if m:
                out[m.group(2)] = m.group(1)
    return out


def probe(path: Path) -> tuple[int, float]:
    """(frame count, duration in seconds) of the first video stream, from the container."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    return int(st["nb_frames"]), float(st["duration"])


def summarize(csv_path: Path) -> dict:
    frames = runs = 0
    by_type: dict[str, int] = {}
    repaired = unrepaired = 0
    seen_runs = set()
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["defective"] == "1":
                frames += 1
                seen_runs.add(r["run_id"])
                by_type[r["defect_type"]] = by_type.get(r["defect_type"], 0) + 1
                st = r.get("repair", "")
                if st.startswith("unrepaired"):
                    unrepaired += 1
                elif st:
                    repaired += 1
    runs = len(seen_runs)
    return {"defective_frames": frames, "runs": runs, "types": by_type,
            "repaired_frames": repaired, "unrepaired_frames": unrepaired}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="EVENT=PATH (repeat, processed in order)")
    ap.add_argument("--out", type=Path, help="output root (or use --beside)")
    ap.add_argument("--beside", action="store_true", help="write to <source>/Fixed/<camera>/ on the same drive")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete each original after a verified repair (needs --local-md5 and --backup-md5)")
    ap.add_argument("--local-md5", action="append", default=[], help="manifest of the originals (EVENT/cam/file paths)")
    ap.add_argument("--backup-md5", action="append", default=[], help="manifest of the verified backup")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cameras", default=r"\d+", help="regex for camera folder names (default: digits)")
    ap.add_argument("--skip-name", action="append", default=[], help="file names to skip")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--margin-gb", type=float, default=20.0, help="keep this much free on the output disk")
    a = ap.parse_args()

    sources = []
    for s in a.source:
        event, _, path = s.partition("=")
        sources.append((event, Path(path).expanduser()))
    if not a.out and not a.beside:
        ap.error("give --out or --beside")
    a.work = a.work.expanduser()
    local_md5, backup_md5 = load_md5(a.local_md5), load_md5(a.backup_md5)
    if a.delete_source and not (local_md5 and backup_md5):
        ap.error("--delete-source needs --local-md5 and --backup-md5 manifests")
    roots = dict(sources)
    a.work.mkdir(parents=True, exist_ok=True)
    status = Status(a.work / "status.json")
    jobs = list_jobs(sources, a.cameras, set(a.skip_name))
    stop = Event()
    space_lock = Lock()
    reserved = [0]
    print(f"{len(jobs)} videos queued", flush=True)

    def process(job):
        event, cam, src = job
        key = f"{event}/{cam}/{src.name}"
        prev = status.get(key)
        if prev and prev.get("state") in ("clean", "repaired"):
            return
        if stop.is_set():
            return
        work = a.work / event / cam / src.stem
        work.mkdir(parents=True, exist_ok=True)
        log = work / "log.txt"
        out = (roots[event] / "Fixed" / cam / src.name) if a.beside else (a.out.expanduser() / event / cam / src.name)
        t0 = time.time()
        try:
            status.set(key, {"state": "analysing", "started": t0})
            run([PY, "-m", "video_repair", "analyze", str(src), "--work", str(work)], log)
            run([PY, "-m", "video_repair", "detect", str(src), "--work", str(work), "--no-review"], log)
            summary = summarize(work / "defective_frames.csv")
            if summary["defective_frames"] == 0:
                status.set(key, {"state": "clean", "seconds": round(time.time() - t0), **summary})
                print(f"clean     {key}", flush=True)
                return
            need = src.stat().st_size * 1.05 + a.margin_gb * 1e9
            with space_lock:                                   # reserve space for concurrent repairs
                free = shutil.disk_usage(src.parent).free if a.beside else \
                    shutil.disk_usage(a.out.expanduser() if a.out.expanduser().exists() else a.out.expanduser().parent).free
                if free - reserved[0] < need:
                    stop.set()
                    status.set(key, {"state": "waiting_for_space", **summary})
                    print(f"STOPPED   out of space before {key}: {free / 1e9:.0f} GB free, "
                          f"need {need / 1e9:.0f} GB (incl. {a.margin_gb:.0f} GB margin)", flush=True)
                    return
                reserved[0] += need
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                part = out.with_name(out.stem + ".partial" + out.suffix)
                status.set(key, {"state": "repairing", "started": t0, **summary})
                run([PY, "-m", "video_repair", "repair", str(src), "-o", str(part), "--work", str(work),
                     "--no-review"], log)
                part.replace(out)                                  # only complete files get the real name
            finally:
                with space_lock:
                    reserved[0] -= need
            summary = summarize(work / "defective_frames.csv")
            n_in, d_in = probe(src)
            n_out, d_out = probe(out)
            complete = n_in == n_out and abs(d_in - d_out) < 0.2
            deleted = False
            note = ""
            if not complete:
                note = f"output incomplete ({n_out}/{n_in} frames, {d_out:.2f}/{d_in:.2f} s); original kept"
            elif a.delete_source:
                h_local, h_backup = local_md5.get(key), backup_md5.get(key)
                if h_local and h_local == h_backup:
                    src.unlink()
                    deleted = True
                else:
                    note = "no verified backup checksum for this file; original kept"
            status.set(key, {"state": "repaired", "seconds": round(time.time() - t0), "output": str(out),
                             "bytes": out.stat().st_size, "frames": n_out, "source_deleted": deleted,
                             "note": note, **summary})
            print(f"repaired  {key}: {summary['defective_frames']} frames in {summary['runs']} runs "
                  f"({summary['unrepaired_frames']} left unrepaired), {time.time() - t0:.0f}s"
                  f"{', original deleted' if deleted else ''}{'  NOTE: ' + note if note else ''}", flush=True)
        except Exception as e:  # keep going with the other videos
            status.set(key, {"state": "error", "error": str(e)})
            print(f"ERROR     {key}: {e}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(process, jobs))
    states: dict[str, int] = {}
    for v in status.data.values():
        states[v["state"]] = states.get(v["state"], 0) + 1
    print(f"done: {states}", flush=True)
    return 1 if stop.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
