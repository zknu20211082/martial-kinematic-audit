"""Kyokushin karate motion capture (Vicon C3D): mapping to the unified 19-joint skeleton and single-strike segmentation."""
import numpy as np
import ezc3d
from scipy.signal import find_peaks

from .config import DATA
from .skeleton import JOINTS, J, interp_nan, smooth
from .descriptors import body_height

# folder with the athlete sub-folders B0367 ... B0405 (the extracted Karate.zip; "Karate" is accepted as well)
ROOT = DATA / "Kyokushin" / "Karate (1)"
if not ROOT.is_dir() and (DATA / "Kyokushin" / "Karate").is_dir():
    ROOT = DATA / "Kyokushin" / "Karate"
TECH = {"S01": "Gyaku-Zuki", "S02": "Mae-Geri", "S03": "Mawashi-Geri-gedan", "S04": "Mawashi-Geri-jodan", "S05": "Ushiro-Mawashi-Geri"}
COND = {"E01": "air", "E02": "shield", "E03": "attacker", "E04": "defender"}
# figshare notice 2023-03-02: three misnamed files
RENAME = {"2017-03-03-B0388-S05-E01-T02": "2017-03-03-B0388-S05-E02-T01",
          "2017-03-03-B0388-S05-E02-T01": "2017-03-03-B0388-S05-E01-T02",
          "2017-03-07-B0396-S03-E02-T01": "2017-03-07-B0396-S03-E01-T03"}
FPS_OUT = 50
FILE_PATTERN = r"(\d{4}-\d{2}-\d{2})-(B\d{4})-(S\d{2})-(E\d{2})-(T\d{2})"

# Plug-in Gait virtual joint centres (preferred) and marker fallbacks
PIG_JOINTS = {
    "pelvis": ["PELO"], "r_hip": ["RFEP"], "r_knee": ["RFEO"], "r_ankle": ["RTIO"],
    "l_hip": ["LFEP"], "l_knee": ["LFEO"], "l_ankle": ["LTIO"],
    "spine": ["TRXO"], "neck": ["TRXO"], "head": ["HEDO"], "head_top": ["HEDO"],
    "l_shoulder": ["LHUP"], "l_elbow": ["LHUO"], "l_wrist": ["LRAO"],
    "r_shoulder": ["RHUP"], "r_elbow": ["RHUO"], "r_wrist": ["RRAO"],
    "l_toe": ["LTOE"], "r_toe": ["RTOE"],
}
MARKER_FALLBACK = {
    "pelvis": ["LASI", "RASI", "LPSI", "RPSI"], "r_hip": ["RASI", "RPSI"], "r_knee": ["RKNE"], "r_ankle": ["RANK"],
    "l_hip": ["LASI", "LPSI"], "l_knee": ["LKNE"], "l_ankle": ["LANK"],
    "spine": ["STRN", "T10"], "neck": ["CLAV", "C7"], "head": ["LFHD", "RFHD", "LBHD", "RBHD"], "head_top": ["LFHD", "RFHD", "LBHD", "RBHD"],
    "l_shoulder": ["LSHO"], "l_elbow": ["LELB"], "l_wrist": ["LWRA", "LWRB"],
    "r_shoulder": ["RSHO"], "r_elbow": ["RELB"], "r_wrist": ["RWRA", "RWRB"],
    "l_toe": ["LTOE"], "r_toe": ["RTOE"],
}


def load_subject_skeleton(path, code):
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    pts = c["data"]["points"][:3]  # (3, N, T) mm
    rate = float(c["header"]["points"]["frame_rate"])
    pref = f"{code}:" if any(l.startswith(code + ":") for l in labels) else ""
    idx = {}
    for i, l in enumerate(labels):
        if pref:
            if l.startswith(pref):
                idx[l[len(pref):]] = i
        elif ":" not in l:
            idx[l] = i
    T = pts.shape[2]
    out = np.full((T, len(JOINTS), 3), np.nan)
    used_virtual = 0
    for jn, ji in J.items():
        names = [n for n in PIG_JOINTS[jn] if n in idx]
        if names:
            used_virtual += 1
        else:
            names = [n for n in MARKER_FALLBACK[jn] if n in idx]
        if not names:
            continue
        arr = np.stack([pts[:, idx[n], :].T for n in names], 0)  # (k,T,3)
        arr[arr == 0] = np.nan  # Vicon marks missing as 0
        out[:, ji] = np.nanmean(arr, axis=0)
    events = c["parameters"].get("EVENT", {})
    ev = []
    if "TIMES" in events and len(events["TIMES"]["value"]):
        tv = np.array(events["TIMES"]["value"])
        ev = sorted((tv[0] * 60 + tv[1]).tolist()) if tv.ndim == 2 else sorted(tv.tolist())
    return out / 1000.0, rate, ev, used_virtual, len(labels)


def segment_strikes(Jt, fps, tech):
    """Return list of (start, end) frame windows around speed peaks of the striking effector."""
    Js = smooth(interp_nan(Jt), fps)
    H = body_height(Js)
    if not np.isfinite(H) or H < 0.5:
        H = 1.6
    vel = np.gradient(Js, 1 / fps, axis=0)
    ee = [13, 16] if tech == "S01" else [17, 18]
    sp = np.max(np.linalg.norm(vel[:, ee], axis=2), axis=1) / H
    thr = max(0.35 * sp.max(), 1.5)  # heights/s
    pk, _ = find_peaks(sp, height=thr, distance=int(0.8 * fps))
    win = int(0.5 * fps)
    segs = []
    for p in pk:
        s, e = max(0, p - win), min(len(Js), p + win)
        segs.append((int(s), int(e), int(p), float(sp[p])))
    return segs
