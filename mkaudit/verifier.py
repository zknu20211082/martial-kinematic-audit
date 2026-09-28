"""Kinematic Claim Verifier (KCV, paper Sec. 1.5).

* ``kcv_from_pose``: pseudo ground truth (and the evidence text for evidence prompting) for in-the-wild clips, derived
  from cached RTMPose 2D keypoints.
* ``match``: field-level verdict of one claim: 'pass' / 'fail' / 'unverifiable'.
* ``recover`` / ``rescore``: re-parse every field from the stored raw model output and re-score it with the verifier
  aligned to the prompt specification (the scoring used in the paper; see scripts/31_vlm_rescore.py).
* ``kf`` / ``hallucination_rate``: kinematic faithfulness KF = N_pass / (N_pass + N_fail) and hallucination rate
  HR = N_fail / N_all, pooled over the kinematic fields of all videos of a test set.
"""
import re

import numpy as np
import pandas as pd

from .config import CACHE
from .vlm import FIELDS

POSE = CACHE / "pose2d"  # cached RTMPose keypoints (scripts/03_extract_pose2d.py)
# fields of the re-scored audit (paper Table 4)
RESCORE_FIELDS = ["technique", "n_people", "striking_limb", "striking_limb_type", "peak_foot_height", "body_turn", "knee_extended_at_peak"]
KIN = [f for f in RESCORE_FIELDS if f not in ("technique", "striking_limb_type")]  # KF uses the original five kinematic fields
COVERED = ["n_people", "striking_limb", "peak_foot_height"]; HELD = ["body_turn", "knee_extended_at_peak"]
PRED_FIELDS = ["technique", "n_people", "striking_limb", "peak_foot_height", "body_turn", "knee_extended_at_peak"]


# ---------------- Kinematic Claim Verifier (wild clips, from RTMPose 2D) ----------------
def kcv_from_pose(clip):
    z = np.load(POSE / (clip + ".npz")); K, S = z["kpts"], z["scores"]
    T = len(K)
    pres = (S.mean(2) > 0.3)  # T,M
    n_people = int((pres.mean(0) > 0.4).sum())
    ev = {"n_people": n_people if n_people < 2 else "2+"}
    if pres[:, 0].mean() < 0.4:
        ev.update({"striking_limb": "unverifiable", "peak_foot_height": "unverifiable"}); return ev, {}
    k = K[:, 0]; s = S[:, 0]
    size = np.median(np.abs(k[:, 5, 1] - k[:, 15, 1]) + np.abs(k[:, 6, 1] - k[:, 16, 1])) / 2 + 1e-6  # shoulder-ankle
    conf = {}
    speeds = {}
    for name, j in (("left_arm", 9), ("right_arm", 10), ("left_leg", 15), ("right_leg", 16)):
        ok = s[:, j] > 0.5
        v = np.linalg.norm(np.diff(k[:, j], axis=0), axis=1) / size * float(z["fps"])
        v[~(ok[1:] & ok[:-1])] = np.nan
        speeds[name] = np.nanpercentile(v, 95) if np.isfinite(v).sum() > 3 else np.nan
    if all(np.isnan(v) for v in speeds.values()):
        ev["striking_limb"] = "unverifiable"
    else:
        best = max(speeds, key=lambda n: -1 if np.isnan(speeds[n]) else speeds[n])
        second = sorted([v for v in speeds.values() if np.isfinite(v)], reverse=True)
        margin = (second[0] - second[1]) / (second[0] + 1e-6) if len(second) > 1 else 1.0
        ev["striking_limb"] = best if margin > 0.15 else "unverifiable"
        conf["limb_margin"] = margin
    # foot height (image y grows downward)
    hip_y = np.nanmean(np.where(s[:, [11, 12]] > 0.5, k[:, [11, 12], 1], np.nan), axis=1)
    sho_y = np.nanmean(np.where(s[:, [5, 6]] > 0.5, k[:, [5, 6], 1], np.nan), axis=1)
    ank_y = np.nanmin(np.where(s[:, [15, 16]] > 0.5, k[:, [15, 16], 1], np.nan), axis=1)
    rel_hip = (hip_y - ank_y) / size; rel_sho = (sho_y - ank_y) / size
    if np.isfinite(rel_hip).sum() < 3:
        ev["peak_foot_height"] = "unverifiable"
    else:
        if np.nanmax(rel_sho) > 0.05:
            ev["peak_foot_height"] = "above_shoulder"
        elif np.nanmax(rel_hip) > 0.05:
            ev["peak_foot_height"] = "hip_to_shoulder"
        else:
            ev["peak_foot_height"] = "below_hip_or_no_kick"
        conf["foot_rel_hip_max"] = float(np.nanmax(rel_hip))
    # evidence text for grounded prompting
    evtxt = {"number of tracked people": ev["n_people"],
             "fastest-moving limb (performer's own side)": ev["striking_limb"],
             "highest foot position": ev["peak_foot_height"].replace("_", " ")}
    if ev["striking_limb"] != "unverifiable":
        evtxt["peak limb speeds (body-heights per second)"] = ", ".join(f"{n}={v:.1f}" for n, v in speeds.items() if np.isfinite(v))
    return ev, evtxt


def match(field, pred, truth):
    """Return 'pass' / 'fail' / 'unverifiable'."""
    if truth in (None, "unverifiable", "") or (isinstance(truth, float) and np.isnan(truth)):
        return "unverifiable"
    p = str(pred).strip().lower()
    t = str(truth).strip().lower()
    if field == "n_people":
        try:
            pi = int(re.search(r"\d+", p).group(0))
        except Exception:
            return "fail"
        return "pass" if (t == "2+" and pi >= 2) or (t != "2+" and pi == int(t)) else "fail"
    if field == "peak_foot_height" and t == "below_hip_or_no_kick":
        return "pass" if p in ("below_hip", "no_kick") else "fail"
    if field == "technique":
        return "pass" if p.replace(" ", "").replace("-", "").replace("_", "") == t.replace(" ", "").replace("-", "").replace("_", "") else "fail"
    return "pass" if p == t else "fail"


def summarize(df, name):
    """Per-condition summary with the scoring applied during generation (written as e3_summary_{wild,synth}.csv;
    superseded in the paper by the re-scored summary, see ``rescore``)."""
    out = []
    for cond, sub in df.groupby("condition"):
        r = {"set": name, "condition": cond, "n": len(sub)}
        for f in FIELDS:
            res = sub[f"res_{f}"]
            ver = res.isin(["pass", "fail"]).sum()
            r[f"{f}_acc"] = float((res == "pass").sum() / ver) if ver else np.nan
            r[f"{f}_verifiable"] = int(ver)
        kin = [f for f in FIELDS if f != "technique"]
        allres = pd.concat([sub[f"res_{f}"] for f in kin])
        ver = allres.isin(["pass", "fail"]).sum()
        r["KF_score"] = float((allres == "pass").sum() / ver) if ver else np.nan
        r["hallucination_rate"] = float((allres == "fail").sum() / len(allres))
        r["verifiable_rate"] = float(ver / len(allres))
        r["mean_gen_s"] = float(sub.gen_s.mean())
        out.append(r)
    return out


# ---------------- re-scoring aligned to the prompt specification ----------------
def limb_type(v):
    v = str(v).strip().lower()
    if v in ("left_arm", "right_arm"):
        return "arm"
    if v in ("left_leg", "right_leg"):
        return "leg"
    if v in ("unverifiable", "", "nan"):
        return "unverifiable"
    return v


def recover(raw, field):
    """Re-parse a field from the stored raw model output (robust to truncation after the field)."""
    m = re.search(r'"%s"\s*:\s*(?:"([^"]*)"|(-?\d+)|(null)|(true|false))' % field, str(raw))
    if not m:
        return ""
    if m.group(1) is not None:
        return m.group(1)
    if m.group(2) is not None:
        return m.group(2)
    if m.group(3):
        return "null"
    return m.group(4)


def rescore(df):
    """- knee_extended_at_peak: ground truth is "n/a" when the (true) striking limb is an arm (the prompt asks for "n/a" in that case);
    - adds striking_limb_type (arm / leg, side-agnostic) as an additional covered field for side-ambiguous views."""
    df = df.copy()
    # stored pred_* columns lost literal "n/a" answers on resume; recover every field from the raw output text
    for f in PRED_FIELDS:
        df[f"pred_{f}"] = [recover(r, f) for r in df["raw"]]
        if f != "knee_extended_at_peak":
            df[f"res_{f}"] = [match(f, p, t) for p, t in zip(df[f"pred_{f}"], df[f"truth_{f}"])]
    arm = df["truth_striking_limb"].astype(str).isin(["left_arm", "right_arm"])
    tk = df["truth_knee_extended_at_peak"].astype(str)
    df.loc[arm & tk.isin(["yes", "no"]), "truth_knee_extended_at_peak"] = "n/a"
    df["res_knee_extended_at_peak"] = [match("knee_extended_at_peak", p, t) for p, t in zip(df["pred_knee_extended_at_peak"], df["truth_knee_extended_at_peak"])]
    df["truth_striking_limb_type"] = df["truth_striking_limb"].map(limb_type)
    df["pred_striking_limb_type"] = df["pred_striking_limb"].map(limb_type)
    df["res_striking_limb_type"] = [match("striking_limb_type", p, t) for p, t in zip(df["pred_striking_limb_type"], df["truth_striking_limb_type"])]
    return df


def kf(sub, fields):
    """Kinematic faithfulness: passes / (passes + fails) pooled over `fields` (NaN if nothing is verifiable)."""
    res = pd.concat([sub[f"res_{f}"] for f in fields]); v = res.isin(["pass", "fail"]).sum()
    return (res == "pass").sum() / v if v else np.nan


def hallucination_rate(sub, fields):
    """Hallucination rate: fails / all claims (unverifiable claims included in the denominator) pooled over `fields`."""
    allres = pd.concat([sub[f"res_{f}"] for f in fields])
    return float((allres == "fail").sum() / len(allres))


def verifiable_rate(sub, fields):
    allres = pd.concat([sub[f"res_{f}"] for f in fields])
    return float(allres.isin(["pass", "fail"]).mean())
