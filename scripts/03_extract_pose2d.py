"""2D pose extraction (RTMPose via rtmlib, ONNX CUDA) for every clip in video_index.csv. Saves one npz per clip.

Paper: Sec. 1.4 (RTMPose-m + YOLOX-m, every 2nd frame, at most 2 persons by box area). The cached keypoints feed the
skeleton stream (20, 22), the KCV pseudo ground truth and evidence text (30) and the counts in 53.
Output: <cache_dir>/pose2d/<clip>.npz. Requires a CUDA GPU (onnxruntime-gpu).

Usage: python scripts/03_extract_pose2d.py [limit] [shard] [nshard]
       limit = max. number of clips (0 = all); shard/nshard split the clip list for parallel processes."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

from mkaudit.cuda_dlls import expose_torch_cuda_dlls
expose_torch_cuda_dlls()  # Windows: must run before onnxruntime / rtmlib are imported
import argparse, time
import numpy as np, pandas as pd, av, cv2
from tqdm import tqdm
from rtmlib import Body
from mkaudit.config import CACHE, DD

OUT = CACHE / "pose2d"
MAX_PERSONS = 2
STRIDE = 2  # every 2nd frame (~12-15 fps)


def read_frames(path, stride=STRIDE):
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 30.0
        for i, fr in enumerate(c.decode(video=0)):
            if i % stride == 0:
                frames.append(fr.to_ndarray(format="bgr24"))
    return frames, fps / stride


def main(limit=None, shard=0, nshard=1):
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(DD / "video_index.csv")
    body = Body(mode="balanced", backend="onnxruntime", device="cuda")
    todo = [r for r in idx.itertuples() if not (OUT / (r.clip + ".npz")).exists()]
    todo = todo[shard::nshard]
    if limit:
        todo = todo[:limit]
    t0 = time.time(); nfr = 0
    for r in tqdm(todo, desc="pose"):
        try:
            frames, fps = read_frames(r.path)
        except Exception as ex:
            print("decode fail", r.clip, ex); continue
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        K = np.zeros((len(frames), MAX_PERSONS, 17, 2), np.float32); S = np.zeros((len(frames), MAX_PERSONS, 17), np.float32)
        for t, img in enumerate(frames):
            kp, sc = body(img)  # (N,17,2),(N,17)
            if kp is None or len(kp) == 0:
                continue
            area = (kp[:, :, 0].max(1) - kp[:, :, 0].min(1)) * (kp[:, :, 1].max(1) - kp[:, :, 1].min(1)) * (sc.mean(1) > 0.3)
            order = np.argsort(-area)[:MAX_PERSONS]
            for p, o in enumerate(order):
                K[t, p] = kp[o]; S[t, p] = sc[o]
        nfr += len(frames)
        np.savez_compressed(OUT / (r.clip + ".npz"), kpts=K, scores=S, fps=fps, hw=np.array([h, w]))
    dt = time.time() - t0
    print(f"clips {len(todo)} frames {nfr} time {dt:.0f}s fps {nfr / max(dt, 1):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("limit", nargs="?", type=int, default=0, help="process at most this many clips (0 = all)")
    ap.add_argument("shard", nargs="?", type=int, default=0, help="index of this shard")
    ap.add_argument("nshard", nargs="?", type=int, default=1, help="number of shards")
    a = ap.parse_args()
    main(limit=a.limit if a.limit else None, shard=a.shard, nshard=a.nshard)
