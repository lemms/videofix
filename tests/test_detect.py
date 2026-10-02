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
    plans = plan_repairs(_det(flags), RepairConfig(max_gap=12))
    assert plans[0].status == "unrepaired:edge"
    assert (plans[5].a, plans[5].b, plans[5].status) == (4, 7, "interpolate")
    assert plans[12].status.startswith("unrepaired:run")
