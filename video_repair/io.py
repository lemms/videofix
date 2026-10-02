"""Streaming video I/O built on PyAV.

Frames are never written to disk.  Each decoded frame is kept in the native
semi-planar layout that NVDEC produces and NVENC consumes (``nv12`` for 8-bit,
``p010le`` for 10-bit sources) so good frames can be passed to the encoder
untouched, and colour conversion happens on the GPU (see :mod:`.color`).
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import av.logging
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    time_base: Fraction
    start_pts: int
    codec: str
    pix_fmt: str
    bit_depth: int
    n_frames: int          # container estimate; exact count comes from decoding
    duration: float
    color_range: int
    colorspace: int
    color_primaries: int
    color_trc: int

    @property
    def native_format(self) -> str:
        return "p010le" if self.bit_depth > 8 else "nv12"

    @property
    def frame_duration_pts(self) -> int:
        return max(1, round(1 / (self.fps * self.time_base)))


@dataclass
class Frame:
    """One decoded frame in native semi-planar layout.

    ``y`` is (H, W) and ``uv`` is (H/2, W) with interleaved U/V samples.  For
    10-bit sources both are uint16 with the sample in the top 10 bits (p010).
    """

    index: int
    pts: int | None
    time: float
    key: bool
    pict_type: str
    corrupt: bool
    decode_errors: int
    packet_size: int
    y: np.ndarray
    uv: np.ndarray
    meta: dict = field(default_factory=dict)


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        fmt = cc.format or s.format
        depth = max((comp.bits for comp in fmt.components), default=8) if fmt else 8
        fps = s.average_rate or s.guessed_rate or Fraction(30)
        if s.duration is not None:
            duration = float(s.duration * s.time_base)
        elif c.duration is not None:
            duration = c.duration / av.time_base
        else:
            duration = 0.0
        n = s.frames or round(duration * fps)
        # container headers can disagree with the bitstream (GoPro yuvj420p files
        # say "limited" but are full range); trust the first decoded frame
        color_range = int(cc.color_range or 0)
        try:
            first = next(c.decode(s))
            color_range = int(first.color_range or color_range)
        except (StopIteration, av.error.FFmpegError):
            pass
        if fmt is not None and fmt.name.startswith("yuvj"):
            color_range = 2
        return VideoInfo(
            path=path,
            width=cc.width,
            height=cc.height,
            fps=Fraction(fps),
            time_base=Fraction(s.time_base),
            start_pts=s.start_time or 0,
            codec=cc.name,
            pix_fmt=fmt.name if fmt else "unknown",
            bit_depth=depth,
            n_frames=int(n),
            duration=duration,
            color_range=color_range,
            colorspace=int(cc.colorspace or 0),
            color_primaries=int(cc.color_primaries or 0),
            color_trc=int(cc.color_trc or 0),
        )


_PICT = {1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_type(pt) -> str:
    """AVPictureType (int or enum, depending on the PyAV version) -> 'I'/'P'/'B'/..."""
    name = getattr(pt, "name", None)
    return name if isinstance(name, str) else _PICT.get(int(pt), "?")


def _plane(p: av.video.plane.VideoPlane, dtype: np.dtype, width: int) -> np.ndarray:
    """Copy one plane out of a frame, dropping line padding."""
    itemsize = np.dtype(dtype).itemsize
    a = np.frombuffer(p, dtype).reshape(p.height, p.line_size // itemsize)
    return np.ascontiguousarray(a[:, :width])


def _to_native(frame: av.VideoFrame, native: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.format.name != native:
        frame = frame.reformat(format=native)
    dtype = np.uint16 if native == "p010le" else np.uint8
    # chroma plane has W/2 interleaved pairs == W samples per row
    return _plane(frame.planes[0], dtype, frame.width), _plane(frame.planes[1], dtype, frame.width)


def open_input(path: str | Path, hwaccel: bool = True) -> av.container.InputContainer:
    if hwaccel:
        from av.codec.hwaccel import HWAccel
        try:
            return av.open(str(path), hwaccel=HWAccel(device_type="cuda", allow_software_fallback=True))
        except Exception as e:  # no CUDA device / driver
            log.warning("CUDA decode unavailable (%s); using software decode", e)
    return av.open(str(path))


def iter_frames(
    path: str | Path,
    info: VideoInfo | None = None,
    hwaccel: bool = True,
    start_index: int = 0,
) -> Iterator[Frame]:
    """Decode *path* frame by frame in presentation order.

    Frame indices are counted from the first decodable frame.  When
    ``start_index`` > 0 the decoder seeks to the preceding keyframe and drops
    frames until the index is reached (indices are derived from pts, so this
    relies on a constant frame rate, which GoPro files have).

    Decoder log errors and packets that fail to decode are attributed to the
    frame with the same pts and reported in ``Frame.decode_errors``.
    """
    info = info or probe(path)
    native = info.native_format
    dur = info.frame_duration_pts
    container = open_input(path, hwaccel)
    stream = container.streams.video[0]
    if not hwaccel:
        stream.thread_type = "AUTO"

    if start_index > 0:
        # seek a little early: MP4 seeks by DTS, and with B-frame reordering the
        # keyframe after the wanted frame can have a DTS before the wanted PTS
        target = info.start_pts + max(0, start_index - 16) * dur
        container.seek(target, stream=stream, backward=True, any_frame=False)

    sizes: dict[int, int] = {}
    errors: dict[int, int] = {}
    base = info.start_pts
    counter = start_index
    prev_level = av.logging.get_level()
    av.logging.set_level(av.logging.ERROR)
    try:
        with av.logging.Capture(local=False) as logs:
            for packet in container.demux(stream):
                n_logs = len(logs)
                key = packet.pts if packet.pts is not None else packet.dts
                if packet.size and key is not None:
                    sizes[key] = packet.size
                try:
                    decoded = packet.decode()
                except av.error.InvalidDataError:
                    decoded = []
                    if key is not None:
                        errors[key] = errors.get(key, 0) + 1
                new = len(logs) - n_logs
                if new and key is not None:
                    errors[key] = errors.get(key, 0) + new
                for f in decoded:
                    pts = f.pts
                    idx = round((pts - base) / dur) if pts is not None else counter
                    counter = idx + 1
                    if idx < start_index:
                        continue
                    y, uv = _to_native(f, native)
                    yield Frame(
                        index=idx,
                        pts=pts,
                        time=float(f.time) if f.time is not None else idx / float(info.fps),
                        key=bool(f.key_frame),
                        pict_type=_pict_type(f.pict_type),
                        corrupt=bool(getattr(f, "is_corrupt", False)),
                        decode_errors=errors.pop(pts, 0) if pts is not None else 0,
                        packet_size=sizes.pop(pts, 0) if pts is not None else 0,
                        y=y,
                        uv=uv,
                    )
                if len(sizes) > 512:   # B-frame reordering never needs this many
                    for k in sorted(sizes)[:-256]:
                        sizes.pop(k, None)
                        errors.pop(k, None)
    finally:
        av.logging.set_level(prev_level)
        container.close()
        del container, stream
        gc.collect()   # NVDEC sessions are only released when the codec context is freed


def read_frames(path: str | Path, indices: set[int], info: VideoInfo | None = None,
                hwaccel: bool = True) -> dict[int, Frame]:
    """Fetch a sparse set of frames, seeking between distant ones."""
    info = info or probe(path)
    out: dict[int, Frame] = {}
    todo = sorted(indices)
    while todo:
        gen = iter_frames(path, info, hwaccel=hwaccel, start_index=todo[0])
        try:
            for fr in gen:
                if fr.index in indices:
                    out[fr.index] = fr
                todo = [i for i in todo if i > fr.index]
                # stop and seek again if the next wanted frame is far away
                if not todo or todo[0] - fr.index > 120:
                    break
            else:
                break
        finally:
            gen.close()
    return out


def sample_keyframes(path: str | Path, n: int, info: VideoInfo | None = None,
                     hwaccel: bool = True) -> list[Frame]:
    """Decode ~n keyframes spread evenly over the file (one seek each)."""
    info = info or probe(path)
    native = info.native_format
    dur = info.frame_duration_pts
    container = open_input(path, hwaccel)
    stream = container.streams.video[0]
    out: list[Frame] = []
    seen: set[int] = set()
    try:
        total = max(info.n_frames, 1)
        for i in range(n):
            target = info.start_pts + int((i + 0.5) * total / n) * dur
            try:
                container.seek(target, stream=stream, backward=True, any_frame=False)
                f = next(container.decode(stream))
            except (StopIteration, av.error.FFmpegError):
                continue
            if f.pts is None or f.pts in seen:
                continue
            seen.add(f.pts)
            y, uv = _to_native(f, native)
            idx = round((f.pts - info.start_pts) / dur)
            out.append(Frame(idx, f.pts, float(f.time or 0), bool(f.key_frame), "I", False, 0, 0, y, uv))
    finally:
        container.close()
        del container, stream
        gc.collect()
    return out


# --------------------------------------------------------------------------- encoding

_HW_ENCODERS = {"hevc": "hevc_nvenc", "h264": "h264_nvenc", "av1": "av1_nvenc"}
_SW_ENCODERS = {"hevc": "libx265", "h264": "libx264", "av1": "libsvtav1"}


class Encoder:
    """Encode native-layout frames, preserving pts and colour metadata.

    Writes a video-only file; :func:`mux` adds audio/telemetry afterwards.
    """

    def __init__(self, out_path: str | Path, info: VideoInfo, codec: str | None = None,
                 quality: int = 18, hw: bool = True):
        self.info = info
        self.out_path = Path(out_path)
        family = info.codec if info.codec in _HW_ENCODERS else "hevc"
        if info.bit_depth > 8 and family == "h264":
            family = "hevc"   # NVENC has no 10-bit H.264
        candidates = [codec] if codec else []
        if hw:
            candidates.append(_HW_ENCODERS[family])
        candidates.append(_SW_ENCODERS[family])
        last_err: Exception | None = None
        for name in candidates:
            try:
                self._open(name, quality)
                self.codec_name = name
                break
            except Exception as e:  # encoder missing or no GPU
                last_err = e
                log.warning("encoder %s unavailable: %s", name, e)
                try:
                    self.container.close()
                except Exception:
                    pass
                self.container = self.stream = None
                gc.collect()
        else:
            raise RuntimeError(f"No usable encoder: {last_err}")

    def _open(self, name: str, quality: int) -> None:
        info = self.info
        self.container = av.open(str(self.out_path), "w")
        s = self.container.add_stream(name, rate=info.fps)
        s.width, s.height = info.width, info.height
        s.time_base = info.time_base
        if name.endswith("_nvenc"):
            s.pix_fmt = info.native_format
            s.options = {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": str(quality),
                         "b": "0", "spatial-aq": "1", "temporal-aq": "1"}
            if name == "hevc_nvenc" and info.bit_depth > 8:
                s.options["profile"] = "main10"
        else:
            s.pix_fmt = "yuv420p10le" if info.bit_depth > 8 else "yuv420p"
            key = "crf"
            s.options = {key: str(quality), "preset": "medium"}
        cc = s.codec_context
        for attr in ("color_range", "colorspace", "color_primaries", "color_trc"):
            val = getattr(info, attr)
            if val:
                try:
                    setattr(cc, attr, val)
                except Exception:
                    pass
        if name.startswith("hevc") or name == "libx265":
            s.codec_tag = "hvc1"
        self.stream = s
        cc.open()   # fail here, not on the first frame, if the encoder is unusable

    def write(self, frame: Frame) -> None:
        native = self.info.native_format
        vf = av.VideoFrame(self.info.width, self.info.height, native)
        dtype = np.uint16 if native == "p010le" else np.uint8
        itemsize = np.dtype(dtype).itemsize
        for plane, src in zip(vf.planes, (frame.y, frame.uv)):
            dst = np.frombuffer(plane, dtype).reshape(plane.height, plane.line_size // itemsize)
            dst[:, : src.shape[1]] = src
        if self.info.color_range:
            vf.color_range = self.info.color_range
        if self.stream.pix_fmt != native:
            vf = vf.reformat(format=self.stream.pix_fmt)
        vf.pts = frame.pts
        vf.time_base = self.info.time_base
        for pkt in self.stream.encode(vf):
            self.container.mux(pkt)

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()
        self.container = self.stream = None
        gc.collect()   # release the NVENC session

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _ffprobe_streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out).get("streams", [])


def mux(video_only: Path, source: Path, output: Path) -> list[str]:
    """Combine the re-encoded video with every other stream of *source*.

    Audio and data streams (GoPro ``gpmd`` telemetry etc.) are stream-copied
    with container metadata; the camera timecode is carried over as video
    metadata so the muxer writes a fresh ``tmcd`` track.  Data streams the MP4
    muxer cannot store (e.g. GoPro's ``fdsc``) are dropped one by one with a
    warning.  Returns warnings.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (needed for the final mux)")
    streams = _ffprobe_streams(source)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    data = [s for s in streams if s.get("codec_type") in ("data", "subtitle")
            and s.get("codec_tag_string") != "tmcd"]
    timecode = next((s.get("tags", {}).get("timecode") for s in streams
                     if s.get("tags", {}).get("timecode")), None)
    is_hevc = _ffprobe_streams(video_only)[0].get("codec_name") == "hevc"

    def run(keep: list[dict]) -> subprocess.CompletedProcess:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(video_only), "-i", str(source), "-map", "0:v:0"]
        for st in audio + keep:
            cmd += ["-map", f"1:{st['index']}"]
        cmd += ["-c", "copy", "-map_metadata", "1", "-map_chapters", "1"]
        if keep:
            cmd += ["-copy_unknown"]
            for out_i, st in enumerate(keep, start=1 + len(audio)):
                tag = st.get("codec_tag_string")
                if tag and not tag.startswith("["):
                    cmd += [f"-tag:{out_i}", tag]
        if timecode:
            cmd += ["-metadata:s:v:0", f"timecode={timecode}"]
        if is_hevc:
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", str(output)]
        return subprocess.run(cmd, capture_output=True, text=True)

    warnings: list[str] = []
    keep = list(data)
    proc = run(keep)
    if proc.returncode != 0 and keep:
        # find the data streams the muxer accepts, one at a time
        keep = [st for st in data if run([st]).returncode == 0]
        dropped = [st.get("codec_tag_string", str(st["index"])) for st in data if st not in keep]
        warnings.append(f"MP4 cannot store data stream(s) {', '.join(dropped)}; dropped")
        proc = run(keep)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr}")
    return warnings
