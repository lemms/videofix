import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_clip(path: Path, n: int = 60, w: int = 320, h: int = 180, fps: int = 30, codec: str = "libx264",
              pix_fmt: str = "yuv420p", gop: int = 15) -> Path:
    """A moving gradient + square, with B-frames, as a small test clip."""
    c = av.open(str(path), "w")
    s = c.add_stream(codec, rate=fps)
    s.width, s.height, s.pix_fmt = w, h, pix_fmt
    s.options = {"g": str(gop), "bf": "2", "crf": "18"} if codec == "libx264" else {}
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = (xx + 3 * i) % 256
        img[..., 1] = (yy * 255 // h)
        img[..., 2] = 128
        x0 = 20 + 4 * i % (w - 60)
        img[60:100, x0:x0 + 40] = 255
        f = av.VideoFrame.from_ndarray(img, format="rgb24")
        f.pts = i
        f.time_base = Fraction(1, fps)
        for p in s.encode(f):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("clips") / "clip.mp4")
