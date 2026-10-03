import numpy as np

from video_repair.detect import Detection, _dilate, robust_z, transient
from video_repair.repair import RepairConfig, plan_repairs


def _dists(pos: np.ndarray):
    """Distances to t-1..t-3 for 1-D 'frames' at positions pos, shaped (N,1,1)."""
    n = len(pos)
    out = []
    for k in (1, 2, 3):
        d = np.full(n, np.nan, np.float32)
        d[k:] = np.abs(pos[k:] - pos[:-k])
        out.append(d.reshape(n, 1, 1))
    return out


def test_transient_flags_isolated_glitch_not_scene_cut():
    pos = np.arange(40, dtype=np.float32) * 0.01      # smooth motion
    pos[25:] += 5.0                                      # scene cut at 25
    pos[10] += 3.0                                       # one corrupt frame
    r1, r2 = transient(*_dists(pos), floor=0.004)
    r = np.fmax(r1, r2)[:, 0, 0]
    assert r[10] > 3
    assert np.nanmax(np.delete(r, [9, 10, 11])) < 1.0   # cut at 25 is not flagged


def test_transient_flags_two_frame_run():
    pos = np.arange(40, dtype=np.float32) * 0.01
    pos[20] += 3.0
    pos[21] += 3.2
    _, r2 = transient(*_dists(pos), floor=0.004)
    assert r2[20, 0, 0] > 3 and r2[21, 0, 0] > 3


def test_robust_z_spike():
    x = np.ones(200) + np.random.default_rng(0).normal(0, 0.01, 200)
    x[100] = 2
    z = robust_z(x, 61, 1e-3)
    assert z[100] > 20 and np.abs(np.delete(z, 100)).max() < 6


def test_dilate():
    m = np.zeros((1, 5, 5), bool)
    m[0, 2, 2] = True
    assert _dilate(m, 1).sum() == 9


def _det(defective):
    n = len(defective)
    d = np.array(defective, bool)
    run, rid = np.full(n, -1), -1
    for t in range(n):
        if d[t]:
            rid += (t == 0 or not d[t - 1])
            run[t] = rid
    types = np.array(["glitch" if x else "" for x in d], dtype=object)
    return Detection(np.arange(n), np.zeros(n), d, d.astype(float), types, run,
                     np.zeros((n, 2, 2), bool), {}, np.ones(n, bool))


def test_plan_repairs_brackets_and_limits():
    flags = [0] * 30
    flags[0] = 1                    # at the edge: no left reference
    for i in range(5, 7):
        flags[i] = 1                # 2-frame run
    for i in range(10, 25):
        flags[i] = 1                # 15 frames > max_gap
    plans = plan_repairs(_det(flags), RepairConfig(max_gap=12, max_static_seconds=0))
    assert plans[0].status == "unrepaired:edge"
    assert (plans[5].a, plans[5].b, plans[5].status) == (4, 7, "interpolate")
    assert plans[12].status.startswith("unrepaired:run")


def test_long_masked_run_is_repairable_but_full_frame_is_not():
    flags = [0] * 60
    for i in range(10, 40):
        flags[i] = 1                # 30 frames, longer than max_gap
    det = _det(flags)
    det.masks[10:40, 0, 0] = True   # small mask (1 of 4 cells) -> masked blending
    plans = plan_repairs(det, RepairConfig(max_gap=12, max_masked_gap=120))
    assert plans[20].status == "interpolate"
    det.masks[10:40] = True         # full-frame masks -> limited by max_gap (static rule off)
    plans = plan_repairs(det, RepairConfig(max_gap=12, max_masked_gap=120, max_static_seconds=0))
    assert plans[20].status.startswith("unrepaired:run")


def test_pict_type_mapping():
    from video_repair.io import _pict_type
    assert [_pict_type(i) for i in (1, 2, 3)] == ["I", "P", "B"]


def test_long_full_frame_run_waits_for_static_check():
    flags = [0] * 60
    for i in range(10, 25):
        flags[i] = 1                # 15 frames, full-frame
    det = _det(flags)
    det.masks[10:25] = True
    plans = plan_repairs(det, RepairConfig(max_gap=12, max_static_seconds=1.0), fps=30.0)
    assert plans[12].status == "interpolate_if_static"
    plans = plan_repairs(det, RepairConfig(max_gap=12, max_static_seconds=0.3), fps=30.0)
    assert plans[12].status.startswith("unrepaired:run")


def test_keep_static_crossfades_and_restores_grain():
    import torch
    from video_repair.repair import Repairer
    torch.manual_seed(0)
    a = torch.full((1, 3, 64, 64), 0.5) + 0.01 * torch.randn(1, 3, 64, 64)
    b = torch.full((1, 3, 64, 64), 0.5) + 0.01 * torch.randn(1, 3, 64, 64)
    mid = torch.zeros(2, 3, 64, 64)                        # interpolator output, must be ignored
    wgt = torch.zeros(1, 1, 64, 64)                        # everything static
    out = Repairer._keep_static(mid, a, b, torch.tensor([0.25, 0.5]), wgt, sigma=0.01)
    assert abs(float(out.mean()) - 0.5) < 2e-3            # cross-fade of a and b, not the interpolator
    noise = (out[1] - 0.5).std()                          # t=0.5: grain restores ~sigma
    assert 0.007 < float(noise) < 0.013


def test_streak_hysteresis_catches_faded_tail_but_not_single_objects():
    from video_repair.detect import _hysteresis
    x = np.array([0.0, 0.002, 0.3, 0.05, 0.02, 0.009, 0.005, 0.002, 0.0, 0.003, 0.012, 0.002])
    on = _hysteresis(x, 0.02, 0.004)
    assert on.tolist() == [False, False, True, True, True, True, True, False, False, False, False, False]


def test_stripes_signal_flags_vertical_streaks():
    import torch
    from video_repair import signals
    rng = np.random.default_rng(0)
    img = rng.random((1, 1, 120, 160)).astype(np.float32) * 0.3 + 0.3     # textured, isotropic
    streaked = img.copy()
    streaked[..., 60:, :] = streaked[..., 59:60, :]                        # copy one row down
    r_clean = signals.stripes(torch.from_numpy(img), (6, 8))
    r_bad = signals.stripes(torch.from_numpy(streaked), (6, 8))
    assert float(r_clean.max()) < np.log(2)
    assert float((r_bad > np.log(3)).float().mean()) > 0.4
