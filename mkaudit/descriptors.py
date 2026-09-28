"""The 54 interpretable kinematic descriptors of a movement segment (paper Sec. 1.2) and the sequence normalisation
used by the neural skeleton models."""
import numpy as np
from scipy.signal import find_peaks

from .skeleton import J, END_EFFECTORS, interp_nan, smooth, resample_to_len


def angle(a, b, c):
    """Angle at b (degrees) between ba and bc, arrays (T,3)."""
    u = a - b; v = c - b
    cosang = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def body_height(Jt):
    """Stature proxy: max over time of head-to-lowest-ankle vertical distance (z-up)."""
    h = Jt[:, J["head"], 2] - np.minimum(Jt[:, J["l_ankle"], 2], Jt[:, J["r_ankle"], 2])
    return float(np.nanpercentile(h, 95))


def log_dimensionless_jerk(traj, fps):
    v = np.gradient(traj, 1 / fps, axis=0)
    a = np.gradient(v, 1 / fps, axis=0)
    jerk = np.gradient(a, 1 / fps, axis=0)
    T = len(traj) / fps
    vpeak = np.max(np.linalg.norm(v, axis=1)) + 1e-9
    return float(-np.log((T ** 3 / vpeak ** 2) * np.trapezoid(np.sum(jerk ** 2, axis=1), dx=1 / fps) + 1e-12))


def descriptors(Jt, fps):
    """Kinematic descriptor vector for a segment. Jt: (T,19,3) metres, z-up. Returns dict."""
    Jt = smooth(interp_nan(Jt), fps)
    T = len(Jt)
    H = body_height(Jt)
    if not np.isfinite(H) or H < 0.5:
        H = 1.6
    f = {"duration_s": T / fps, "height_proxy_m": H}
    vel = np.gradient(Jt, 1 / fps, axis=0)
    speed = np.linalg.norm(vel, axis=2) / H  # heights per second
    pel = Jt[:, 0]
    peak_t = {}
    for name, idx in END_EFFECTORS.items():
        s = speed[:, idx]
        k = int(np.argmax(s))
        f[f"peak_speed_{name}"] = float(s[k])
        f[f"peak_time_frac_{name}"] = k / max(1, T - 1)
        peak_t[name] = k
    fastest = max(END_EFFECTORS, key=lambda n: f[f"peak_speed_{n}"])
    f["fastest_ee"] = fastest
    f["fastest_is_leg"] = float("toe" in fastest)
    f["fastest_is_left"] = float(fastest.startswith("l_"))
    kf = peak_t[fastest]
    # kick height relative to pelvis / shoulder (in heights)
    sho_z = 0.5 * (Jt[:, J["l_shoulder"], 2] + Jt[:, J["r_shoulder"], 2])
    for side in ("l", "r"):
        toe_z = Jt[:, J[f"{side}_toe"], 2]
        f[f"{side}_toe_max_rel_pelvis"] = float(np.max((toe_z - pel[:, 2]) / H))
        f[f"{side}_toe_max_rel_shoulder"] = float(np.max((toe_z - sho_z) / H))
    f["kick_height_rel_pelvis"] = max(f["l_toe_max_rel_pelvis"], f["r_toe_max_rel_pelvis"])
    f["kick_height_rel_shoulder"] = max(f["l_toe_max_rel_shoulder"], f["r_toe_max_rel_shoulder"])
    # joint angles
    for side in ("l", "r"):
        knee = angle(Jt[:, J[f"{side}_hip"]], Jt[:, J[f"{side}_knee"]], Jt[:, J[f"{side}_ankle"]])
        hip = angle(Jt[:, J[f"{side}_knee"]], Jt[:, J[f"{side}_hip"]], Jt[:, J[f"{side}_shoulder"]])
        elb = angle(Jt[:, J[f"{side}_shoulder"]], Jt[:, J[f"{side}_elbow"]], Jt[:, J[f"{side}_wrist"]])
        f[f"{side}_knee_min"] = float(np.min(knee)); f[f"{side}_knee_max"] = float(np.max(knee))
        f[f"{side}_knee_range"] = f[f"{side}_knee_max"] - f[f"{side}_knee_min"]
        f[f"{side}_hip_min"] = float(np.min(hip)); f[f"{side}_hip_range"] = float(np.max(hip) - np.min(hip))
        f[f"{side}_elbow_max"] = float(np.max(elb)); f[f"{side}_elbow_min"] = float(np.min(elb))
        f[f"{side}_knee_at_peak"] = float(knee[kf]); f[f"{side}_elbow_at_peak"] = float(elb[kf])
    # trunk lean from vertical
    trunk = Jt[:, J["neck"]] - pel
    lean = np.degrees(np.arccos(np.clip(trunk[:, 2] / (np.linalg.norm(trunk, axis=1) + 1e-9), -1, 1)))
    f["trunk_lean_mean"] = float(np.mean(lean)); f["trunk_lean_max"] = float(np.max(lean)); f["trunk_lean_at_peak"] = float(lean[kf])
    # pelvis-shoulder axial separation (horizontal plane)
    hipv = Jt[:, J["r_hip"], :2] - Jt[:, J["l_hip"], :2]
    shov = Jt[:, J["r_shoulder"], :2] - Jt[:, J["l_shoulder"], :2]
    ang = np.degrees(np.arctan2(shov[:, 1], shov[:, 0]) - np.arctan2(hipv[:, 1], hipv[:, 0]))
    ang = (ang + 180) % 360 - 180
    f["hip_shoulder_sep_max"] = float(np.max(np.abs(ang))); f["hip_shoulder_sep_at_peak"] = float(abs(ang[kf]))
    # pelvis (COM proxy) dynamics
    pv = np.linalg.norm(np.gradient(pel, 1 / fps, axis=0), axis=1) / H
    f["pelvis_peak_speed"] = float(np.max(pv)); f["pelvis_mean_speed"] = float(np.mean(pv))
    f["pelvis_vert_range"] = float((pel[:, 2].max() - pel[:, 2].min()) / H)
    f["pelvis_path_len"] = float(np.sum(np.linalg.norm(np.diff(pel[:, :2], axis=0), axis=1)) / H)
    # pelvis yaw rotation total (body turning)
    yaw = np.unwrap(np.arctan2(hipv[:, 1], hipv[:, 0]))
    f["pelvis_yaw_range_deg"] = float(np.degrees(yaw.max() - yaw.min()))
    # stance
    st = np.linalg.norm(Jt[:, J["l_ankle"], :2] - Jt[:, J["r_ankle"], :2], axis=1) / H
    f["stance_width_mean"] = float(np.mean(st)); f["stance_width_max"] = float(np.max(st))
    # smoothness / rhythm of fastest effector
    f["ldj_fastest"] = log_dimensionless_jerk(Jt[:, END_EFFECTORS[fastest]], fps)
    s = speed[:, END_EFFECTORS[fastest]]
    pk, _ = find_peaks(s, height=0.5 * s.max(), distance=max(1, int(0.25 * fps)))
    f["n_speed_peaks_fastest"] = int(len(pk))
    f["speed_mean_over_peak_fastest"] = float(np.mean(s) / (s.max() + 1e-9))
    # symmetry
    f["sym_wrist"] = abs(f["peak_speed_l_wrist"] - f["peak_speed_r_wrist"])
    f["sym_toe"] = abs(f["peak_speed_l_toe"] - f["peak_speed_r_toe"])
    # head stability
    hv = np.linalg.norm(np.gradient(Jt[:, J["head"]], 1 / fps, axis=0), axis=1) / H
    f["head_mean_speed"] = float(np.mean(hv))
    return f


def normalize_sequence(Jt, n_frames=64):
    """Sequence for neural models: z-up, pelvis-centred, height-scaled, resampled to n_frames. (n,19,3)"""
    Jt = smooth(interp_nan(Jt), 50)
    H = body_height(Jt)
    if not np.isfinite(H) or H < 0.5:
        H = 1.6
    X = (Jt - Jt[:, :1, :]) / H
    return resample_to_len(X, n_frames).astype(np.float32)
