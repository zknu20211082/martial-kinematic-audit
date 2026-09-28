"""Synthetic supervision from motion capture: multi-view pinhole projection of unified skeletons,
(1) projected 2D skeleton datasets for the viewpoint experiment, (2) rendered stick-figure videos with
exact kinematic ground truth for the video-LLM faithfulness benchmark.

Paper: Sec. 1.4 and Table 1 (36 192 projected 2D skeletons at 8 azimuths; 200 stick-figure videos).
Outputs: <work_dir>/synth_viewpoint_karate.npz (feeds 21, 22), <cache_dir>/synth_videos/*.mp4 and
<work_dir>/synth_videos.csv (feed 30). The 200 videos and their ground-truth table used in the paper are shipped in
data/synthetic_benchmark/.

Usage: python scripts/05_render_synthetic.py [all|viewpoint|videos]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, json
import numpy as np, pandas as pd, av
from tqdm import tqdm
from mkaudit.config import DD, CACHE, ensure_dirs, metadata_path
from mkaudit.skeleton import resample_to_len
from mkaudit.synth import AZ, W, H, FPS, camera, project, world_from_sequence, to_coco17, draw_frame, gt_labels

VID = CACHE / "synth_videos"


def build_viewpoint_dataset():
    """Karate strikes -> COCO-17 2D projections at 8 azimuths (normalised like the video pipeline)."""
    z = np.load(DD / "karate_sequences.npz"); keys, X = z["keys"], z["X"]
    df = pd.read_csv(DD / "karate_strikes.csv")
    key2row = {f"{f}#{k}": i for i, (f, k) in enumerate(zip(df.file, df.strike_idx))}
    out, meta = [], []
    for i, k in enumerate(tqdm(keys, desc="viewpoint-proj")):
        r = df.iloc[key2row[k]]
        P = world_from_sequence(X[i])
        for az in AZ:
            R, c, f = camera(az)
            p2, _ = project(P, R, c, f)
            p2 = to_coco17(p2)
            cx, cy = p2[..., 0].mean(), p2[..., 1].mean(); size = max(p2[..., 1].max() - p2[..., 1].min(), 1.0)
            n = np.zeros((len(p2), 17, 3), np.float32)
            n[..., 0] = (p2[..., 0] - cx) / size; n[..., 1] = -(p2[..., 1] - cy) / size; n[..., 2] = 1.0
            n = resample_to_len(n, 48).astype(np.float32)
            out.append(np.transpose(n, (2, 0, 1))[..., None])  # C,T,V,M=1
            meta.append({"key": k, "code": r.code, "tech": r.tech, "cond": r.cond, "az": az})
    np.savez_compressed(DD / "synth_viewpoint_karate.npz", X=np.stack(out), meta=json.dumps(meta))
    print("viewpoint dataset", len(out))


def render_videos(n_karate=120, n_taichi=80, seed=0):
    VID.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    bg = np.full((H, W, 3), 235, np.uint8)
    for y in range(H):  # simple floor gradient
        bg[y] = 235 - int(40 * y / H)
    zk = np.load(DD / "karate_sequences.npz"); dfk = pd.read_csv(DD / "karate_strikes.csv")
    key2row = {f"{f}#{k}": i for i, (f, k) in enumerate(zip(dfk.file, dfk.strike_idx))}
    # stratified by technique
    per_t = n_karate // 5
    sel = []
    for t in sorted(dfk.tech.unique()):
        idx = [i for i, k in enumerate(zk["keys"]) if dfk.iloc[key2row[k]].tech == t]
        sel += list(rng.choice(idx, min(per_t, len(idx)), replace=False))
    items = [("karate", zk["X"][i], dfk.iloc[key2row[zk["keys"][i]]].tech_name, zk["keys"][i]) for i in sel]
    zt = np.load(DD / "taichi_qualisys_sequences.npz"); dft = pd.read_csv(DD / "taichi_qualisys.csv")
    gcls = pd.read_csv(metadata_path("UMONS-TAICHI", "gesture_classes.csv")).set_index("gesture_id")
    f2i = {k: i for i, k in enumerate(zt["keys"])}
    tsel = dft.sample(n_taichi, random_state=seed)
    items += [("taichi", zt["X"][f2i[r.file]], gcls.loc[r.gesture, "name_en"], r.file) for r in tsel.itertuples()]
    for k, (src, X, label, key) in enumerate(tqdm(items, desc="render")):
        P = world_from_sequence(X)
        az = int(rng.choice(AZ)); R, c, f = camera(az, el_deg=rng.uniform(5, 20))
        p2, dep = project(P, R, c, f)
        n_out = max(8, int(round(len(P) / 50 * FPS)))
        p2r = resample_to_len(p2, n_out)
        name = f"synth_{k:04d}_{src}.mp4"
        with av.open(str(VID / name), "w") as cont:
            st = cont.add_stream("libx264", rate=FPS); st.width, st.height, st.pix_fmt = W, H, "yuv420p"; st.options = {"crf": "28"}
            for t in range(n_out):
                fr = av.VideoFrame.from_ndarray(draw_frame(p2r[t], None, bg), format="bgr24")
                for pkt in st.encode(fr):
                    cont.mux(pkt)
            for pkt in st.encode():
                cont.mux(pkt)
        gt = gt_labels(P); gt.update({"video": name, "source": src, "label": label, "key": key, "azimuth": az, "n_people": 1})
        rows.append(gt)
    pd.DataFrame(rows).to_csv(DD / "synth_videos.csv", index=False)
    print("rendered", len(rows)); print(pd.DataFrame(rows).groupby(["source", "peak_foot_height"]).size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["all", "viewpoint", "videos"])
    what = ap.parse_args().what
    ensure_dirs(DD)
    if what in ("all", "viewpoint"):
        build_viewpoint_dataset()
    if what in ("all", "videos"):
        render_videos()
