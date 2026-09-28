"""Unified 19-joint skeleton and trajectory preprocessing (gap filling, resampling, vertical-axis alignment, smoothing)."""
import numpy as np
from scipy.interpolate import interp1d

# Unified 19-joint skeleton (H36M-17 + two toes)
JOINTS = ["pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
          "spine", "neck", "head", "head_top",
          "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
          "l_toe", "r_toe"]
J = {n: i for i, n in enumerate(JOINTS)}
BONES = [(0, 1), (1, 2), (2, 3), (3, 18), (0, 4), (4, 5), (5, 6), (6, 17), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
END_EFFECTORS = {"l_wrist": 13, "r_wrist": 16, "l_toe": 17, "r_toe": 18}


def interp_nan(x):
    """Linear interpolation of NaN / zero-dropout frames along axis 0. x: (T, ...)"""
    x = x.astype(np.float64).copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            flat[:, c] = 0.0
        elif bad.any():
            flat[bad, c] = np.interp(t[bad], t[~bad], col[~bad])
    return flat.reshape(x.shape)


def resample(x, fps_in, fps_out):
    """Resample (T, ...) trajectory to fps_out with linear interpolation."""
    T = len(x)
    if T < 2:
        return x
    dur = (T - 1) / fps_in
    n = max(2, int(round(dur * fps_out)) + 1)
    t_in = np.linspace(0, dur, T); t_out = np.linspace(0, dur, n)
    f = interp1d(t_in, x, axis=0, kind="linear")
    return f(t_out)


def resample_to_len(x, n):
    T = len(x)
    if T == n:
        return x
    t_in = np.linspace(0, 1, T); t_out = np.linspace(0, 1, n)
    return interp1d(t_in, x, axis=0, kind="linear")(t_out)


def vertical_axis(Jt):
    """Detect vertical axis index: axis with largest (head - ankle) mean difference."""
    d = Jt[:, J["head"]] - 0.5 * (Jt[:, J["l_ankle"]] + Jt[:, J["r_ankle"]])
    return int(np.argmax(np.abs(np.nanmean(d, axis=0)))), np.sign(np.nanmean(d, axis=0))


def to_z_up(Jt):
    """Permute axes so vertical is the last axis and points up. Returns (T,19,3)."""
    v, s = vertical_axis(Jt)
    order = [i for i in range(3) if i != v] + [v]
    out = Jt[:, :, order].copy()
    if s[v] < 0:
        out[:, :, 2] *= -1
    return out


def smooth(x, fps, win_s=0.06):
    """Moving-average smoothing along time (odd window)."""
    w = max(1, int(round(win_s * fps)))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return x
    k = np.ones(w) / w
    flat = x.reshape(len(x), -1)
    pad = w // 2
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        col = np.pad(flat[:, c], pad, mode="edge")
        out[:, c] = np.convolve(col, k, mode="valid")
    return out.reshape(x.shape)
