# video-repair

Finds and repairs defective frames in large videos (built for footage from a
faulty GoPro): codec corruption (blocky, smeared or green areas), frozen or
duplicated frames, black/white flashes, lines, tearing and noise bursts.

Nothing needs labelling. Each video is compared against itself, and
defective runs are rebuilt by interpolating between the nearest good frames
with RIFE 4.25.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e '.[test]'
```

FFmpeg/ffprobe must be on `PATH` for the final mux. Decoding and encoding
use the FFmpeg bundled with PyAV (NVDEC/NVENC when an NVIDIA GPU is present).

Models download on first use:
- **DINOv2-small with registers** (Apache-2.0, no account needed). Register
  tokens remove the high-norm artifact patches of plain DINOv2, which matters
  because detection compares patches individually. `--backbone dinov2` uses
  plain DINOv2. `--backbone dinov3` uses DINOv3, which needs a Hugging Face
  account with the licence accepted and `hf auth login`.
- **RIFE 4.25** (Practical-RIFE, MIT licence): checksum-verified and loaded with
  `weights_only=True`, cached in `~/.cache/video_repair/`.

## Usage

```bash
# everything: analyse -> detect -> repair
video-repair run GX010042.MP4 -o GX010042_fixed.mp4

# or step by step
video-repair analyze GX010042.MP4          # pass 1, resumable after interruption
video-repair detect  GX010042.MP4          # writes CSV, masks and a review page
#   ... inspect GX010042_vrwork/review/index.html, edit the CSV if needed ...
video-repair repair  GX010042.MP4 -o GX010042_fixed.mp4
```

The work directory (`<input>_vrwork/` by default, or `--work`) contains:

| File | Contents |
|---|---|
| `features/` | per-frame features in chunks of 1,024 frames (about 5.5 KB per frame) |
| `defective_frames.csv` | `frame_num, pts_time, defective, confidence, defect_type, mask_area, run_id, repair` |
| `masks.npz` | patch masks (about 14×24 cells) of each defective frame |
| `review/index.html` | defective frames with their masks, good neighbours, and before/after thumbnails |

**Editing the CSV:** `repair` reads `defective_frames.csv`, so you can set
`defective` to 0 for a false positive or to 1 for a missed frame before
running it. Frames you add get a full-frame mask.

**Calibration:** `detect --labels labels.csv` (columns `frame_num,defective`)
picks the confidence threshold that maximises F1 on your hand-labelled frames.

Useful options: `--threshold` (default 0.5), `--max-gap` (longest run to
interpolate, default 12), `--extend-to-keyframe` (see Limitations), `--quality` (NVENC CQ / x265 CRF, default 18),
`--device cpu`, `--no-hwaccel` (software decoding, which also reports
decoder errors), `--backbone`.

## How it works

Three streaming passes. Frames are never written to disk, and memory use
doesn't depend on video length.

1. **analyze**: Decode with NVDEC and keep frames in native `nv12`/`p010`.
   For each frame, store:
   - DINO patch-token distances to frames t-1, t-2 and t-3, matched within a
     3×3 neighbourhood so small motion doesn't count;
   - pixel distances to the same frames;
   - distance of each patch to a memory bank of patches sampled from keyframes
     across the whole video;
   - codec-grid blockiness, row/column line scores, noise energy, exposure;
   - decoder flags.
2. **detect**: Combines these signals:
   - *Transient*: a corrupt frame is far from both good neighbours while the
     neighbours are close to each other. The score is `log(A/C)` per patch,
     for runs of one and two frames. A scene cut is close to one side, so it
     isn't flagged.
   - *Keyframe smear*: codec corruption persists until the next keyframe. It
     shows as a patch-level step at a P/B-frame that reverses at that
     keyframe, in the same cells and not across the whole frame.
   - *Freeze*: almost no luma change compared with the local level of motion.
   - *Intrinsic*: robust temporal z-scores of blockiness, novelty, lines and
     noise.

   Runs longer than `max_gap`, and runs at the start or end of the clip, are
   reported but not repaired.
3. **repair**: For each run, RIFE interpolates between the frames on either
   side at t = (i-a)/(b-a). Small localised defects are blended through a
   feathered mask, so untouched pixels keep their original values. The output
   is encoded with NVENC (same codec family and bit depth, colour metadata
   preserved), then audio, GoPro GPMF telemetry and container metadata are
   copied with ffmpeg.

## Measured results

Test material: three Xiph 1080p50 sequences (crowd_run, park_joy,
ducks_take_off) joined into a 1,500-frame 10-bit HEVC clip with a keyframe
every 50 frames and two real scene cuts. Defects were injected with
`tools/make_synthetic.py`.

| Test clip | Defects | Precision | Recall |
|---|---|---|---|
| Pixel defects, seed 1 (thresholds tuned here) | 40 frames, all 7 types | 1.000 | 0.950 |
| Pixel defects, seed 7 (held out) | 82 frames | 1.000 | 0.963 |
| Real bitstream damage (bytes flipped in 11 packets) | 53 frames | 1.000 | 0.189 |
| ...same, with `--extend-to-keyframe` | | 0.981 | 0.981 |

- Neither scene cut was ever flagged.
- Misses: noise bursts that the encoder mostly smoothed away (about 1.5× the
  normal coding error) and one tear.
- Real bitstream damage is mostly *persistent*. A damaged reference frame
  corrupts every frame until the next keyframe, often too faintly to see in
  individual frames. By default only the onset is detected.
  `--extend-to-keyframe` assumes propagation. It's right for that kind of
  damage, but it over-flags footage with isolated glitches: precision on seed
  1 fell to 0.28 with it on. Check the review page before using it.

Repair quality on the seed-1 clip: PSNR of defective frames against the clean
source, before → after.

| Type | Before | After |
|---|---|---|
| black | 8.3 dB | 27.9 dB |
| flash | 6.5 dB | 29.1 dB |
| tear | 18.2 dB | 27.9 dB |
| blocks | 18.8 dB | 23.2 dB |
| freeze | 26.9 dB | 30.3 dB |
| lines | 27.0 dB | 31.2 dB |
| all | 18.7 dB | 26.5 dB |

RIFE against linear blending, predicting a dropped frame: 29.0 vs 23.9 dB and
27.5 vs 18.7 dB on normal motion; about equal on chaotic splashing water.

Reproduce:

```bash
python tools/make_synthetic.py clean.mp4 broken.mp4 --truth truth.csv --seed 7
video-repair run broken.mp4 -o fixed.mp4 --work w
python tools/evaluate.py truth.csv w/defective_frames.csv --clean clean.mp4 --broken broken.mp4 --repaired fixed.mp4
python tools/make_synthetic.py clean.mp4 bits.mp4 --truth truth_bits.csv --bitstream   # real decoder corruption
```

Speed on an RTX 4060 Ti (8 GB), 1080p: analysis about 135 fps; the full
`run` on 1,500 frames takes 35 s. CPU-only analysis runs at about 9 fps.

## Tests

```bash
.venv/bin/python -m pytest -q            # all (the end-to-end tests load models)
.venv/bin/python -m pytest -q -m "not slow"
```

## Limitations

- **Persistent bitstream damage** (see above) is detected only at its onset
  unless you pass `--extend-to-keyframe`. Even then, its masks usually cover
  most of the frame, so runs longer than `--max-gap` are flagged
  `unrepaired:run>12` rather than interpolated across a second or more.
  Damaged clips decode differently with NVDEC and with FFmpeg's software
  decoder (NVDEC hides more), so pick one decoder and use it for every pass.
- Noise bursts that the camera's encoder mostly smoothed away can be missed.
- Long runs and chaotic motion (splashing water) interpolate poorly. Runs
  longer than `--max-gap` (full-frame) or `--max-masked-gap` (mask-only) are
  left in place and marked in the CSV.
- The whole video is re-encoded once at high quality (about 50 dB PSNR
  against the source), so untouched frames are not bit-exact copies.
- The detection results above are identical with DINOv2 and DINOv2 with
  registers (registers separate novelty slightly better). DINOv3 is untested.
