"""E7: does motion-capture multi-view synthetic supervision help skeleton recognition on in-the-wild video?
Pre-train the ST-GCN skeleton stream on 2D projections of the karate mocap (single view vs. eight views),
then fine-tune on HMDB51-MA / UCF101-MA official splits with full or reduced labels, and compare with training
from scratch under identical hyper-parameters. Also repeats the two-class cross-dataset transfer.

Paper: Table 3(b) and Sec. 2.4. Needs <work_dir>/synth_viewpoint_karate.npz (05), the caches of 03/04 and a CUDA GPU.
Outputs in <results_dir>: e7_synth_pretrain.csv (resumable), e7_synth_pretrain_cross.csv, e7_synth_pretrain.json;
pre-trained weights are cached in <cache_dir>/pretrain/."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import time, json, math
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from mkaudit.config import DD, RES, CACHE, ensure_dirs
from mkaudit.stgcn import STGCN, train_model, predict
from mkaudit.recognition import COCO_BONES, augment2d, load_pose, POSE, VMAE
from mkaudit.utils import save_json

PRE = CACHE / "pretrain"


def pretrain(views, epochs=20):
    path = PRE / f"stgcn_synth_{views}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu"), None
    z = np.load(DD / "synth_viewpoint_karate.npz"); X = torch.tensor(z["X"]); meta = pd.DataFrame(json.loads(str(z["meta"])))
    techs = sorted(meta.tech.unique()); y = meta.tech.map({t: i for i, t in enumerate(techs)}).values
    keep = np.ones(len(meta), bool) if views == "8view" else (meta.az.values == 0)
    t0 = time.time()
    m = STGCN(3, 17, COCO_BONES, len(techs), n_persons=1, base=64, dropout=0.3)
    m = train_model(m, X[keep], y[keep], task="cls", epochs=epochs, bs=64, lr=2e-3, aug=augment2d, seed=0)
    sd = {k: v.cpu() for k, v in m.state_dict().items() if not k.startswith("fc.")}
    torch.save(sd, path)
    print(f"pretrained {views}: n={int(keep.sum())} in {time.time() - t0:.0f}s")
    return sd, time.time() - t0


def new_model(n_cls, init):
    m = STGCN(3, 17, COCO_BONES, n_cls, n_persons=2, base=64, dropout=0.3)
    if init is not None:
        missing, unexpected = m.load_state_dict(init, strict=False)
        assert all(k.startswith("fc.") for k in missing) and not unexpected, (missing, unexpected)
    return m


def load_dataset(ds, idx):
    sub = idx[idx.dataset == ds].reset_index(drop=True)
    ok = [(POSE / (c + ".npz")).exists() and (VMAE / (c + ".npz")).exists() for c in sub["clip"]]
    sub = sub[ok].reset_index(drop=True)
    classes = sorted(sub.cls.unique()); y = sub.cls.map({c: i for i, c in enumerate(classes)}).values
    X = torch.tensor(np.stack([load_pose(c) for c in sub["clip"]]))
    F = np.stack([np.load(VMAE / (c + ".npz"))["feat"].mean(0) for c in sub["clip"]])
    return sub, classes, X, F, y


def subsample(tr_idx, y, frac, seed):
    if frac >= 1:
        return tr_idx
    rng = np.random.default_rng(seed); keep = []
    for c in np.unique(y[tr_idx]):
        ids = tr_idx[y[tr_idx] == c]
        keep += list(rng.choice(ids, max(2, math.ceil(frac * len(ids))), replace=False))
    return np.array(sorted(keep))


def main():
    ensure_dirs(RES, PRE)
    inits = {"scratch": None}
    for v in ("1view", "8view"):
        inits[f"pretrain-{v}"], _ = pretrain(v)
    idx = pd.read_csv(DD / "video_index.csv")
    rows, packs = [], {}
    out_csv = RES / "e7_synth_pretrain.csv"
    if out_csv.exists():   # resume
        rows = pd.read_csv(out_csv).to_dict("records")
    done = {(r["dataset"], int(r["split"]), float(r["label_frac"]), r["init"]) for r in rows}
    for ds in ("HMDB51-MA", "UCF101-MA"):
        sub, classes, X, F, y = packs[ds] = load_dataset(ds, idx)
        for s in (1, 2, 3):
            tr_all = np.where((sub[f"split{s}"] == 1).values)[0]; te = np.where((sub[f"split{s}"] == 2).values)[0]
            for frac in (1.0, 0.25):
                tr = subsample(tr_all, y, frac, seed=100 + s)
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(F[tr], y[tr]); p_vm = clf.predict_proba(F[te])
                for name, init in inits.items():
                    if (ds, s, frac, name) in done:
                        continue
                    t0 = time.time()
                    m = train_model(new_model(len(classes), init), X[tr], y[tr], task="cls", epochs=40, bs=32, lr=2e-3, aug=augment2d, seed=s)
                    p_sk = torch.softmax(torch.tensor(predict(m, X[te])), 1).numpy()
                    accs = {}
                    for stream, p in (("skeleton", p_sk), ("fusion", 0.5 * p_sk + 0.5 * p_vm), ("videomae", p_vm)):
                        if stream == "videomae" and name != "scratch":
                            continue
                        accs[stream] = accuracy_score(y[te], p.argmax(1))
                        rows.append({"dataset": ds, "split": s, "label_frac": frac, "n_train": len(tr), "n_test": len(te), "init": name, "stream": stream,
                                     "top1": accs[stream], "mean_class_acc": balanced_accuracy_score(y[te], p.argmax(1)), "train_s": time.time() - t0})
                    print(ds, s, frac, name, {k: round(v, 3) for k, v in accs.items()}, flush=True)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
    # cross-dataset (punch vs fencing), skeleton stream only
    shared = {"HMDB51-MA": {"punch": 0, "fencing": 1}, "UCF101-MA": {"Punch": 0, "Fencing": 1}}
    cross = []
    for src, dst in (("UCF101-MA", "HMDB51-MA"), ("HMDB51-MA", "UCF101-MA")):
        def sel(ds):
            sub, classes, X, F, y = packs[ds]; m = sub.cls.isin(shared[ds]).values
            return X[m], F[m], sub.cls[m].map(shared[ds]).values
        Xs, Fs, ys = sel(src); Xd, Fd, yd = sel(dst)
        p_vm = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(Fs, ys).predict_proba(Fd)
        for name, init in inits.items():
            m = train_model(new_model(2, init), Xs, ys, task="cls", epochs=30, bs=32, lr=2e-3, aug=augment2d, seed=0)
            p_sk = torch.softmax(torch.tensor(predict(m, Xd)), 1).numpy()
            for stream, p in (("skeleton", p_sk), ("fusion", 0.5 * p_sk + 0.5 * p_vm)):
                cross.append({"train": src, "test": dst, "init": name, "stream": stream, "top1": accuracy_score(yd, p.argmax(1)), "n_test": len(yd)})
            print(cross[-2], cross[-1], flush=True)
    pd.DataFrame(cross).to_csv(RES / "e7_synth_pretrain_cross.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["dataset", "label_frac", "stream", "init"]).top1.agg(["mean", "std"]).round(4)
    print(summ.to_string())
    # paired split-level comparison vs scratch
    out = {"per_setting": json.loads(summ.reset_index().to_json(orient="records")), "cross": cross}
    save_json(out, RES / "e7_synth_pretrain.json")


if __name__ == "__main__":
    main()
