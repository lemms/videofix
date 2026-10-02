import numpy as np
import torch

from video_repair import color, io


def test_probe(clip):
    info = io.probe(clip)
    assert (info.width, info.height) == (320, 180)
    assert info.bit_depth == 8 and info.native_format == "nv12"
    assert info.n_frames == 60


def test_iter_frames_indices_in_order(clip):
    frames = list(io.iter_frames(clip, hwaccel=False))
    assert [f.index for f in frames] == list(range(60))
    assert frames[0].key
    assert frames[0].y.shape == (180, 320) and frames[0].uv.shape == (90, 320)


def test_read_frames_seeks_exactly(clip):
    want = {0, 14, 15, 16, 44, 59}
    got = io.read_frames(clip, want, hwaccel=False)
    assert set(got) == want
    full = {f.index: f for f in io.iter_frames(clip, hwaccel=False)}
    for i in want:
        assert np.array_equal(got[i].y, full[i].y)


def test_encode_roundtrip_keeps_frames(clip, tmp_path):
    info = io.probe(clip)
    out = tmp_path / "out.mp4"
    with io.Encoder(out, info, hw=False) as enc:
        for f in io.iter_frames(clip, info, hwaccel=False):
            enc.write(f)
    o = io.probe(out)
    assert o.n_frames == info.n_frames
    assert [f.index for f in io.iter_frames(out, o, hwaccel=False)] == list(range(60))


def test_mux_copies_audio(clip, tmp_path):
    import av
    src = tmp_path / "with_audio.mp4"
    c = av.open(str(src), "w")
    vin = av.open(str(clip))
    v = c.add_stream_from_template(vin.streams.video[0])
    a = c.add_stream("aac", rate=48000)
    for p in vin.demux(vin.streams.video[0]):
        if p.dts is None:
            continue
        p.stream = v
        c.mux(p)
    frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
    frame.sample_rate = 48000
    for i in range(90):
        frame.pts = i * 1024
        for p in a.encode(frame):
            c.mux(p)
    for p in a.encode():
        c.mux(p)
    c.close()
    vin.close()
    out = tmp_path / "muxed.mp4"
    io.mux(clip, src, out)
    with av.open(str(out)) as m:
        assert len(m.streams.video) == 1 and len(m.streams.audio) == 1


def test_color_roundtrip_luma_exact(clip):
    info = io.probe(clip)
    f = next(io.iter_frames(clip, info, hwaccel=False))
    y, uv = color.upload([f], info, torch.device("cpu"))
    rgb = color.yuv_to_rgb(y, uv, info)
    (y2, uv2), = color.download(*color.rgb_to_yuv(rgb, info), info)
    assert np.abs(y2.astype(int) - f.y.astype(int)).max() <= 1
