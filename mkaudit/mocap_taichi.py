"""UMONS-TAICHI: segmented Qualisys TSV and segmented Kinect V2 skeletons mapped to the unified 19-joint skeleton."""
import re

import numpy as np

from .config import DATA
from .skeleton import JOINTS, J

TD = DATA / "UMONS-TAICHI"
FPS_OUT = 50
Q_JOINTS = {
    "pelvis": ["L_IAS", "L_IPS", "R_IPS", "R_IAS"], "r_hip": ["R_FTC"], "r_knee": ["R_FLE", "R_FME"], "r_ankle": ["R_FAL", "R_TAM"],
    "l_hip": ["L_FTC"], "l_knee": ["L_FLE", "L_FME"], "l_ankle": ["L_FAL", "L_TAM"],
    "spine": ["STRN", "TV10"], "neck": ["CV7", "CLAV"], "head": ["LFHD", "RFHD", "LBHD", "RBHD"], "head_top": ["LFHD", "RFHD", "LBHD", "RBHD"],
    "l_shoulder": ["LAC"], "l_elbow": ["L_HLE", "L_HME"], "l_wrist": ["L_RSP", "L_USP"],
    "r_shoulder": ["RAC"], "r_elbow": ["R_HLE", "R_HME"], "r_wrist": ["R_RSP", "R_USP"],
    "l_toe": ["L_FM2"], "r_toe": ["R_FM2"],
}
# Kinect V2 SDK joint order
K_JOINTS = {"pelvis": [0], "r_hip": [16], "r_knee": [17], "r_ankle": [18], "l_hip": [12], "l_knee": [13], "l_ankle": [14],
            "spine": [1], "neck": [20], "head": [3], "head_top": [3], "l_shoulder": [4], "l_elbow": [5], "l_wrist": [6],
            "r_shoulder": [8], "r_elbow": [9], "r_wrist": [10], "l_toe": [15], "r_toe": [19]}
PAT = re.compile(r"P(\d{2})T(\d{2})C(\d{2})G(\d{2})D(\d{2})S(\d{2})")


def parse_name(stem):
    m = PAT.match(stem)
    p, t, c, g, d, s = m.groups()
    return {"pid": f"P{p}", "type": f"T{t}", "clip": int(c), "gesture": f"G{g}", "direction": f"D{d}", "instance": int(s)}


def read_tsv(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    names, start, freq = None, None, 179.0
    for i, l in enumerate(lines[:15]):
        parts = l.split("\t")
        if parts[0] == "FREQUENCY":
            freq = float(parts[1])
        if parts[0] == "MARKER_NAMES":
            names = [x for x in parts[1:] if x]; start = i + 1; break
    rows = [np.array([float(v) if v else np.nan for v in l.split("\t")]) for l in lines[start:] if l.strip()]
    ncol = 3 * len(names)
    arr = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        arr[i, :min(ncol, len(r))] = r[:ncol]
    arr = arr.reshape(len(rows), len(names), 3)
    arr[arr == 0] = np.nan
    return names, arr, freq


def to_unified(arr, names, mapping):
    idx = {n: i for i, n in enumerate(names)}
    T = len(arr)
    out = np.full((T, len(JOINTS), 3), np.nan)
    for jn, ji in J.items():
        ids = [idx[n] for n in mapping[jn] if n in idx]
        if ids:
            out[:, ji] = np.nanmean(arr[:, ids], axis=1)
    return out
