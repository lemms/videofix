#!/usr/bin/env python3
"""After a batch run: repair defect runs that sit at GoPro chapter boundaries.

    python tools/fix_boundaries.py --source Mehndi=/media/Storage_Small/Mehndi \\
        --work scratch/batch/work [--flat-work Mehndi=scratch/batch/work_pass2] --cameras 4 [--dry-run]

For each camera folder it pairs the chapters of every recording, using the
repaired copy in ``<source>/Fixed/<camera>/`` when there is one (only those are
rewritten) and the original otherwise.  Detection results come from the work
directories (``<work>/<event>/<camera>/<stem>/``); a ``--flat-work EVENT=DIR``
(``<DIR>/<stem>/``, e.g. a second pass) wins for that event's files.
Each rewritten file is swapped in only once the new version is complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_repair import boundaries as B  # noqa: E402

VIDEO = re.compile(r"\.(mp4|mov)$", re.I)


def find_csv(works: list[Path], flat: list[Path], event: str, cam: str, stem: str) -> Path | None:
    found = None
    for c in [w / event / cam / stem / "defective_frames.csv" for w in works] + \
             [w / stem / "defective_frames.csv" for w in flat]:
        if c.exists():
            found = c
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="EVENT=PATH")
    ap.add_argument("--work", action="append", type=Path, required=True, help="batch work dir(s), later wins")
    ap.add_argument("--flat-work", action="append", default=[], help="EVENT=DIR with <DIR>/<stem>/ layout (wins)")
    ap.add_argument("--cameras", default=r"\d+")
    ap.add_argument("--dry-run", action="store_true", help="only list the boundaries that need repair")
    ap.add_argument("--report", type=Path, default=Path("scratch/batch/boundaries.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    done = json.loads(a.report.read_text()) if a.report.exists() else {}
    for s in a.source:
        event, _, root = s.partition("=")
        root = Path(root)
        flat = [Path(d) for e, _, d in (f.partition("=") for f in a.flat_work) if e == event]
        cams = sorted({d.name for d in root.iterdir() if d.is_dir() and re.fullmatch(a.cameras, d.name)}
                      | {d.name for d in (root / "Fixed").glob("*") if d.is_dir() and re.fullmatch(a.cameras, d.name)})
        for cam in cams:
            files: dict[str, tuple[Path, bool]] = {}
            for f in sorted((root / cam).glob("*")) if (root / cam).exists() else []:
                if VIDEO.search(f.name) and ".partial" not in f.name:
                    files[f.name] = (f, False)
            for f in sorted((root / "Fixed" / cam).glob("*")) if (root / "Fixed" / cam).exists() else []:
                if VIDEO.search(f.name) and ".partial" not in f.name:
                    files[f.name] = (f, True)                 # the repaired copy wins
            for rec in B.recordings([p for p, _ in files.values()]):
                key = f"{event}/{cam}/{rec[0].stem}"
                if done.get(key, {}).get("state") == "done":
                    continue
                chapters = [B.Chapter(p, find_csv(a.work, flat, event, cam, p.stem), writable=files[p.name][1]) for p in rec]
                for ch in chapters:
                    n = B.io.probe(ch.video).n_frames
                    ch.head, ch.tail = B.edge_runs(ch.csv, n)
                pairs = B.plan_boundaries(chapters)
                for i, j in pairs:
                    print(f"{key}: {chapters[i].video.name} tail {chapters[i].tail} | "
                          f"{chapters[j].video.name} head {chapters[j].head}", flush=True)
                if a.dry_run:
                    continue
                try:
                    report = B.fix_recording(chapters) if pairs else {"boundaries": [], "rewritten": []}
                    done[key] = {"state": "done", **report}
                    for r in report["boundaries"]:
                        print(f"  {r['pair'][0]} | {r['pair'][1]}: {r['status']}"
                              f"{' (%d frames)' % r['frames'] if 'frames' in r else ''}", flush=True)
                except Exception as e:
                    done[key] = {"state": "error", "error": str(e)}
                    print(f"  ERROR {key}: {e}", flush=True)
                a.report.write_text(json.dumps(done, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
