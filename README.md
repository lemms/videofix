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
- **DINOv3 ViT-S/16** (gated): accept the licence at
  <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> and run
  `hf auth login`. Without access it falls back to **DINOv2-small** (Apache-2.0).
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
interpolate, default 12), `--quality` (NVENC CQ / x265 CRF, default 18),
`--device cpu`, `--no-hwaccel` (software decoding, which also reports
decoder errors), `--backbone dinov2`.

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

On a 1080p50 10-bit HEVC test clip (three Xiph sequences joined, with two real
scene cuts), with defects injected by `tools/make_synthetic.py`:

| Clip | Precision | Recall |
|---|---|---|
| seed 1 (tuning set, 40 defective frames) | see `tools/evaluate.py` | |
| seed 7 (held out, 82 defective frames) | | |

Repair (seed 1): defective frames improve from 18.7 dB to 26.5 dB mean PSNR
against the clean source.

Reproduce:

```bash
python tools/make_synthetic.py clean.mp4 broken.mp4 --truth truth.csv --seed 7
video-repair run broken.mp4 -o fixed.mp4 --work w
python tools/evaluate.py truth.csv w/defective_frames.csv --clean clean.mp4 --broken broken.mp4 --repaired fixed.mp4
python tools/make_synthetic.py clean.mp4 bits.mp4 --truth truth_bits.csv --bitstream   # real decoder corruption
```

## Tests

```bash
.venv/bin/python -m pytest -q            # all (the end-to-end tests load models)
.venv/bin/python -m pytest -q -m "not slow"
```

## Limitations

- Noise bursts that the camera's encoder mostly smoothed away can be missed.
- Long runs (longer than `--max-gap`) and large, chaotic motion (splashing
  water) interpolate poorly. These are left in place and marked in the CSV.
- The whole video is re-encoded once at high quality (about 50 dB PSNR
  against the source); untouched frames are not copied bit-exactly.
