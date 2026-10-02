"""Static HTML contact sheet of detected (and repaired) frames."""

from __future__ import annotations

import html
from pathlib import Path

import cv2
import numpy as np
import torch

from . import color, io
from .detect import Detection

THUMB_W = 480


def _jpeg(frame: io.Frame, info: io.VideoInfo, mask: np.ndarray | None, path: Path) -> None:
    h = round(THUMB_W * info.height / info.width) // 2 * 2
    rgb = color.frames_to_rgb([frame], info, torch.device("cpu"), size=(h, THUMB_W))[0]
    img = (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if mask is not None and mask.any():
        m = cv2.resize(mask.astype(np.uint8), (THUMB_W, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        edge = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[m] = (0.75 * img[m] + 0.25 * np.array([255, 40, 40])).astype(np.uint8)
        img[edge] = (255, 40, 40)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


class ReviewWriter:
    """Collects thumbnails while frames stream past, then writes ``index.html``."""

    def __init__(self, out_dir: Path, info: io.VideoInfo, det: Detection, max_items: int = 400):
        self.dir = Path(out_dir)
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        self.info, self.det, self.max_items = info, det, max_items
        self.items: dict[int, dict] = {}
        # show the most confident frames of each run first if there are too many
        order = sorted(np.flatnonzero(det.defective), key=lambda i: -det.confidence[i])
        self.wanted = set(order[:max_items])
        self.context: dict[int, int] = {}            # neighbour frame -> run id
        for rid, first, last in det.runs():
            if first in self.wanted or last in self.wanted:
                self.context[first - 1] = rid
                self.context[last + 1] = rid

    def add(self, frame: io.Frame, before: io.Frame | None = None, status: str = "") -> None:
        i = frame.index
        if i in self.context:
            _jpeg(frame, self.info, None, self.dir / "img" / f"{i:08d}_ctx.jpg")
        if i not in self.wanted:
            return
        mask = self.det.masks[i]
        if before is not None:
            _jpeg(before, self.info, mask, self.dir / "img" / f"{i:08d}_before.jpg")
            _jpeg(frame, self.info, None, self.dir / "img" / f"{i:08d}_after.jpg")
        else:
            _jpeg(frame, self.info, mask, self.dir / "img" / f"{i:08d}_before.jpg")
        self.items[i] = {"status": status or frame.meta.get("repair", "")}

    def write(self, title: str, statuses: dict[int, str] | None = None) -> Path:
        det = self.det
        statuses = statuses or {}
        runs = []
        for rid, first, last in det.runs():
            frames = [i for i in range(first, last + 1) if i in self.items]
            if frames:
                runs.append((rid, first, last, frames))
        cards = []
        for rid, first, last, frames in runs:
            t0 = det.time[first]
            imgs = []
            ctx_a = self.dir / "img" / f"{first - 1:08d}_ctx.jpg"
            if ctx_a.exists():
                imgs.append(f'<figure class="ctx"><img loading="lazy" src="img/{ctx_a.name}"><figcaption>#{first - 1} good</figcaption></figure>')
            for i in frames:
                st = statuses.get(i) or self.items[i]["status"]
                imgs.append(
                    f'<figure><img loading="lazy" src="img/{i:08d}_before.jpg">'
                    f'<figcaption>#{i} {html.escape(str(det.defect_type[i]))} · {det.confidence[i]:.2f}</figcaption></figure>')
                after = self.dir / "img" / f"{i:08d}_after.jpg"
                if after.exists():
                    imgs.append(f'<figure class="after"><img loading="lazy" src="img/{after.name}">'
                                f'<figcaption>#{i} {html.escape(st)}</figcaption></figure>')
            ctx_b = self.dir / "img" / f"{last + 1:08d}_ctx.jpg"
            if ctx_b.exists():
                imgs.append(f'<figure class="ctx"><img loading="lazy" src="img/{ctx_b.name}"><figcaption>#{last + 1} good</figcaption></figure>')
            st = statuses.get(first, "")
            cards.append(
                f'<section><h2>Run {rid} · frames {first}–{last} · {t0:.2f}s'
                f'{" · " + html.escape(st) if st else ""}</h2><div class="row">{"".join(imgs)}</div></section>')
        n_def = int(det.defective.sum())
        body = "".join(cards) or "<p>No defective frames detected.</p>"
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f6f6f4;--fg:#1c1c1c;--muted:#666;--card:#fff;--line:#ddd;--accent:#c62828;--ok:#2e7d32;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--muted:#999;--card:#1e1e1e;--line:#333;--accent:#ef5350;--ok:#66bb6a;color-scheme:dark}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:var(--muted);margin:0 0 16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:0 0 12px}}
h2{{font-size:14px;margin:0 0 8px}}
.row{{display:flex;gap:8px;overflow-x:auto}}
figure{{margin:0;flex:0 0 auto;width:min(320px,80vw)}} img{{width:100%;display:block;border-radius:4px;border:2px solid var(--accent)}}
figure.ctx img{{border-color:var(--line)}} figure.after img{{border-color:var(--ok)}}
figcaption{{font-size:12px;color:var(--muted);margin-top:2px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{n_def} defective frames in {len(det.runs())} runs; showing {len(self.items)}.
Red outline = detected patch mask, green = repaired, grey = good neighbours.</p>
{body}</body></html>"""
        out = self.dir / "index.html"
        out.write_text(page)
        return out


def review_detections(path: Path, info: io.VideoInfo, det: Detection, out_dir: Path,
                      hwaccel: bool = True, max_items: int = 400) -> Path:
    """Detection-only review: seek to each flagged run and its neighbours."""
    rw = ReviewWriter(out_dir, info, det, max_items)
    wanted = set(rw.wanted) | set(rw.context)
    for i, fr in sorted(io.read_frames(path, wanted, info, hwaccel).items()):
        rw.add(fr)
    return rw.write(f"Detections · {path.name}")
