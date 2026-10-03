"""Defect runs at chapter boundaries are interpolated across the two files."""

import csv
from pathlib import Path

import pytest

from video_repair import boundaries as B
from video_repair import io


def test_recording_key_and_grouping():
    files = [Path(n) for n in ("GP010008.MP4", "GOPR0008.MP4", "GP020008.MP4", "GOPR0007.MP4", "notes.MP4")]
    assert B.recording_key(Path("GOPR0008.MP4")) == ("0008", 0)
    assert B.recording_key(Path("GP120008.MP4")) == ("0008", 12)
    assert B.recording_key(Path("notes.MP4")) is None
    assert [[f.name for f in r] for r in B.recordings(files)] == \
        [["GOPR0007.MP4"], ["GOPR0008.MP4", "GP010008.MP4", "GP020008.MP4"]]


def _csv(path: Path, n: int, unrepaired: set[int], repaired: set[int] = frozenset()) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_num", "defective", "repair"])
        for i in range(n):
            st = "unrepaired:edge" if i in unrepaired else "interpolated" if i in repaired else ""
            w.writerow([i, int(i in unrepaired or i in repaired), st])
    return path


def test_edge_runs_and_plan(tmp_path):
    assert B.edge_runs(_csv(tmp_path / "a.csv", 20, {17, 18, 19}, {5}), 20) == (None, (17, 19))
    assert B.edge_runs(_csv(tmp_path / "b.csv", 20, {0, 1}), 20) == ((0, 1), None)
    assert B.edge_runs(_csv(tmp_path / "c.csv", 20, {10}), 20) == (None, None)
    chs = [B.Chapter(Path("x"), None, tail=(17, 19)), B.Chapter(Path("y"), None),
           B.Chapter(Path("z"), None, head=(0, 1))]
    assert B.plan_boundaries(chs) == [(0, 1), (1, 2)]


@pytest.mark.slow
def test_black_tail_and_head_are_interpolated_across_chapters(clip, tmp_path):
    info = io.probe(clip)
    a, b = tmp_path / "GOPR0001.MP4", tmp_path / "GP010001.MP4"
    with io.Encoder(a, info, hw=False, quality=12) as ea, io.Encoder(b, info, hw=False, quality=12) as eb:
        for f in io.iter_frames(clip, info, hwaccel=False):
            if f.index in (28, 29, 30):          # end of chapter 0 and start of chapter 1
                f.y[:] = 16
                f.uv[:] = 128
            if f.index < 30:
                ea.write(f)
            else:
                f.index -= 30
                eb.write(f)
    na, nb = io.probe(a).n_frames, io.probe(b).n_frames
    assert (na, nb) == (30, 30)
    chapters = [B.Chapter(a, _csv(tmp_path / "a.csv", na, {28, 29})),
                B.Chapter(b, _csv(tmp_path / "b.csv", nb, {0}))]
    report = B.fix_recording(chapters, device="cpu", hwaccel=False)
    assert report["boundaries"][0]["status"] == "interpolated"
    assert report["boundaries"][0]["frames"] == 3
    assert io.probe(a).n_frames == 30 and io.probe(b).n_frames == 30
    fa = io.read_frames(a, {27, 28, 29}, hwaccel=False)
    fb = io.read_frames(b, {0, 1}, hwaccel=False)
    for f in (fa[28], fa[29], fb[0]):
        assert f.y.mean() > 40                   # no longer black
    assert abs(float(fa[27].y.mean()) - float(fa[28].y.mean())) < 15
