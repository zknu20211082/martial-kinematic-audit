"""Synthetic supervision from motion capture (paper Sec. 1.4): multi-view pinhole projection of unified skeletons,
COCO-17 conversion, stick-figure rendering and exact kinematic ground truth for rendered clips."""
import math

import numpy as np
import cv2

from .skeleton import J, BONES
from .descriptors import descriptors

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
W, H, FPS = 320, 240, 25


def camera(az_deg, el_deg=12.0, dist=4.5, f=300.0):
    az, el = math.radians(az_deg), math.radians(el_deg)
    cpos = np.array([dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az), dist * math.sin(el)])
    target = np.array([0, 0, 0.9])
    fwd = target - cpos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd])  # rows: x right, y down, z forward
    return R, cpos, f


def project(P, R, cpos, f):
    """P: (T,V,3) world metres -> (T,V,2) pixels + depth."""
    Q = (P - cpos) @ R.T
    z = np.clip(Q[..., 2], 0.5, None)
    x = f * Q[..., 0] / z + W / 2; y = f * Q[..., 1] / z + H / 2
    return np.stack([x, y], -1), z


def world_from_sequence(X, H_m=1.7):
    """Undo normalisation: X (T,19,3) pelvis-centred height-scaled -> metres, pelvis at (0,0,~0.95), feet near ground."""
    P = X * H_m
    zmin = np.percentile(np.minimum(P[:, J["l_ankle"], 2], P[:, J["r_ankle"], 2]), 5)
    P[:, :, 2] -= zmin - 0.08
    P[:, :, :2] -= P[:1, :1, :2]  # first-frame pelvis at origin (keep translation)
    return P


COCO_FROM_UNIFIED = {0: "head", 1: "head", 2: "head", 3: "head", 4: "head", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
                     9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip", 13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle"}


def to_coco17(P2):
    """(T,19,2) unified 2D -> (T,17,2) COCO order (head joints approximated by head centre)."""
    return np.stack([P2[:, J[COCO_FROM_UNIFIED[i]]] for i in range(17)], 1)


def draw_frame(p2, depth, bg):
    img = bg.copy()
    col = (30, 30, 30)
    for a, b in BONES:
        pa, pb = tuple(np.round(p2[a]).astype(int)), tuple(np.round(p2[b]).astype(int))
        cv2.line(img, pa, pb, col, 5, cv2.LINE_AA)
    for v in range(len(p2)):
        cv2.circle(img, tuple(np.round(p2[v]).astype(int)), 5, (200, 60, 40), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(p2[J["head"]]).astype(int)), 11, (60, 60, 60), -1, cv2.LINE_AA)
    return img


def gt_labels(P, fps=50):
    """Exact kinematic ground truth for a world-space sequence (T,19,3)."""
    d = descriptors(P, fps)
    fast = d["fastest_ee"]
    limb = {"l_wrist": "left_arm", "r_wrist": "right_arm", "l_toe": "left_leg", "r_toe": "right_leg"}[fast]
    kh = d["kick_height_rel_pelvis"]; ks = d["kick_height_rel_shoulder"]
    if d["fastest_is_leg"] < 0.5 and kh < 0.15:
        foot = "no_kick"
    elif ks > 0:
        foot = "above_shoulder"
    elif kh > 0:
        foot = "hip_to_shoulder"
    else:
        foot = "below_hip"
    turn = "yes" if d["pelvis_yaw_range_deg"] > 90 else "no"
    knee_side = "l" if fast.startswith("l") else "r"
    knee_ext = "yes" if d[f"{knee_side}_knee_at_peak"] > 150 else "no"
    return {"striking_limb": limb, "peak_foot_height": foot, "body_turn": turn, "knee_extended_at_peak": knee_ext,
            "kick_height_rel_pelvis": kh, "pelvis_yaw_range_deg": d["pelvis_yaw_range_deg"], "peak_speed": d[f"peak_speed_{fast}"]}
