"""2D-skeleton helpers for recognition on in-the-wild video (paper Sec. 1.4): COCO-17 topology, loading and
normalising cached RTMPose keypoints, training-time augmentation and test-time perturbations."""
import numpy as np
import torch

from .config import CACHE
from .skeleton import resample_to_len

COCO_BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
T_FRAMES = 48
POSE = CACHE / "pose2d"; VMAE = CACHE / "videomae"


def load_pose(clip, hw_norm=True):
    z = np.load(POSE / (clip + ".npz"))
    K, S, hw = z["kpts"], z["scores"], z["hw"]  # (T,M,17,2),(T,M,17)
    T, M = K.shape[:2]
    X = np.zeros((T, M, 17, 3), np.float32)
    for m in range(M):
        k = K[:, m]; s = S[:, m]
        valid = s > 0.3
        if valid.sum() == 0:
            continue
        # centre on mean valid keypoint per clip, scale by clip-level person size
        cx = k[..., 0][valid].mean(); cy = k[..., 1][valid].mean()
        size = max(k[..., 1][valid].max() - k[..., 1][valid].min(), 1.0)
        X[:, m, :, 0] = (k[..., 0] - cx) / size; X[:, m, :, 1] = -(k[..., 1] - cy) / size; X[:, m, :, 2] = s
        X[:, m][~valid] = 0
    X = resample_to_len(X, T_FRAMES).astype(np.float32)  # (T,M,17,3)
    return np.transpose(X, (3, 0, 2, 1))  # C,T,V,M


def augment2d(xb, rng):
    """xb: tensor (N,3,T,V,M). random horizontal flip (swap L/R joints), scale, temporal shift, jitter."""
    x = xb.clone()
    N = x.shape[0]
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    perm = list(range(17))
    for a, b in flip_pairs:
        perm[a], perm[b] = b, a
    for i in range(N):
        if rng.random() < 0.5:
            x[i, 0] = -x[i, 0]; x[i] = x[i][:, :, perm, :]
        x[i, :2] *= rng.uniform(0.85, 1.15)
        sh = rng.integers(-4, 5); x[i] = torch.roll(x[i], int(sh), dims=1)
        x[i, :2] += torch.randn_like(x[i, :2]) * 0.01
    return x


def perturb(X, kind, rng):
    """E4 robustness perturbations on skeleton tensors (N,3,T,V,M)."""
    x = X.clone()
    if kind == "flip":
        perm = list(range(17))
        for a, b in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
            perm[a], perm[b] = b, a
        x[:, 0] = -x[:, 0]; x = x[:, :, :, perm, :]
    elif kind.startswith("occl"):
        p = float(kind[4:]) / 100
        mask = torch.tensor(rng.random((x.shape[0], 1, x.shape[2], x.shape[3], 1)) < p)
        x = x.masked_fill(mask, 0.0)
    elif kind.startswith("crop"):
        f = float(kind[4:]) / 100; T = x.shape[2]; L = max(8, int(T * f)); st = (T - L) // 2
        seg = x[:, :, st:st + L]
        x = torch.nn.functional.interpolate(seg.permute(0, 1, 3, 4, 2).reshape(x.shape[0], -1, L), size=T, mode="linear").reshape(x.shape[0], 3, x.shape[3], x.shape[4], T).permute(0, 1, 4, 2, 3)
    elif kind.startswith("jitter"):
        s = float(kind[6:]) / 100
        x[:, :2] += torch.randn_like(x[:, :2]) * s
    elif kind.startswith("second"):  # drop second person
        x[:, :, :, :, 1:] = 0
    return x
