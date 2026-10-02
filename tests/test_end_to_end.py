"""Full pipeline on a tiny clip with an injected black frame (CPU, downloads models once)."""

import csv

import numpy as np
import pytest

from video_repair import io
from video_repair.pipeline import main

pytestmark = pytest.mark.slow


def test_black_frame_is_detected_and_repaired(clip, tmp_path):
    info = io.probe(clip)
    broken = tmp_path / "broken.mp4"
    with io.Encoder(broken, info, hw=False, quality=12) as enc:
        for f in io.iter_frames(clip, info, hwaccel=False):
            if f.index == 30:
                f.y[:] = 16
                f.uv[:] = 128
            enc.write(f)
    out = tmp_path / "fixed.mp4"
    work = tmp_path / "work"
    assert main(["run", str(broken), "-o", str(out), "--work", str(work), "--device", "cpu",
                 "--no-hwaccel", "--backbone", "dinov2reg"]) == 0
    rows = {int(r["frame_num"]): r for r in csv.DictReader(open(work / "defective_frames.csv"))}
    flagged = {i for i, r in rows.items() if int(r["defective"])}
    assert 30 in flagged
    assert len(flagged) <= 2
    assert rows[30]["repair"] in ("interpolated", "blended")
    fixed = io.read_frames(out, {30}, hwaccel=False)[30]
    assert fixed.y.mean() > 40          # no longer black
    assert io.probe(out).n_frames == info.n_frames
    assert (work / "review" / "index.html").exists()


def test_blockwise_detection_matches_whole(clip, tmp_path, monkeypatch):
    from video_repair import detect as D
    from video_repair.analyze import analyze

    store = analyze(clip, tmp_path / "feat", backbone_name="dinov2", hwaccel=False, device="cpu")
    cfg = D.DetectConfig(window=11, max_smear=10, threshold=0.2)   # overlap 18 < clip length 60
    whole = D.detect(store, cfg)
    monkeypatch.setattr(D, "BLOCK", 7)
    parts = D.detect(store, cfg)
    assert np.array_equal(whole.defective, parts.defective)
    assert np.allclose(whole.confidence, parts.confidence)


def test_analyze_resumes_after_interruption(clip, tmp_path, monkeypatch):
    from video_repair import analyze as A

    monkeypatch.setattr(A, "CHUNK", 16)            # 60 frames -> 4 chunks
    full = A.analyze(clip, tmp_path / "a", backbone_name="dinov2", hwaccel=False, device="cpu").load()
    store = A.analyze(clip, tmp_path / "b", backbone_name="dinov2", hwaccel=False, device="cpu")
    for c in (2, 3):                               # simulate a crash after chunk 1
        store.chunk_path(c).unlink()
    resumed = A.analyze(clip, tmp_path / "b", backbone_name="dinov2", hwaccel=False, device="cpu").load()
    assert np.array_equal(full["index"], resumed["index"])
    for k in ("pd1", "fd1", "fd3", "block", "still"):
        assert np.allclose(full[k].astype(np.float32), resumed[k].astype(np.float32), atol=2e-3, equal_nan=True), k
