"""E1: fine-grained martial-arts action recognition on HMDB51-MA / UCF101-MA (official splits).
Streams: (a) 2D-skeleton ST-GCN (RTMPose), (b) VideoMAE-K400 linear probe, (c) late fusion, (d) cross-dataset,
(e) robustness of the skeleton stream to test-time perturbations.

Paper: Table 3(a), Fig. 3(a)-(c), Sec. 2.3. Needs the caches of 03 and 04 and a CUDA GPU.
Outputs in <results_dir>: e1_recognition.csv, e1_cross_dataset.csv, e1_confusion_{HMDB51-MA,UCF101-MA}[_skeleton|_videomae].csv."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from mkaudit.config import DD, RES, HF_CACHE, ensure_dirs
from mkaudit.stgcn import STGCN, train_model, predict
from mkaudit.recognition import COCO_BONES, POSE, VMAE, load_pose, augment2d, perturb


def run_dataset(ds, idx, splits=(1, 2, 3), epochs=40):
    sub = idx[idx.dataset == ds].reset_index(drop=True)
    ok = [(POSE / (c + ".npz")).exists() and (VMAE / (c + ".npz")).exists() for c in sub["clip"]]
    print(ds, "clips with both caches:", sum(ok), "/", len(sub))
    sub = sub[ok].reset_index(drop=True)
    classes = sorted(sub.cls.unique()); c2i = {c: i for i, c in enumerate(classes)}
    y = sub.cls.map(c2i).values
    X = torch.tensor(np.stack([load_pose(c) for c in sub["clip"]]))
    F = np.stack([np.load(VMAE / (c + ".npz"))["feat"].mean(0) for c in sub["clip"]])
    L = np.stack([np.load(VMAE / (c + ".npz"))["logits"].mean(0) for c in sub["clip"]])
    out, probs_store = [], {}
    rng = np.random.default_rng(0)
    for s in splits:
        tr = (sub[f"split{s}"] == 1).values; te = (sub[f"split{s}"] == 2).values
        # (a) skeleton ST-GCN
        t0 = time.time()
        model = STGCN(3, 17, COCO_BONES, len(classes), n_persons=2, base=64, dropout=0.3)
        model = train_model(model, X[tr], y[tr], task="cls", epochs=epochs, bs=32, lr=2e-3, aug=augment2d, seed=s)
        t_train = time.time() - t0
        t0 = time.time(); logit_sk = predict(model, X[te]); t_inf = (time.time() - t0) / te.sum()
        p_sk = torch.softmax(torch.tensor(logit_sk), 1).numpy()
        # (b) VideoMAE linear probe
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
        clf.fit(F[tr], y[tr]); p_vm = clf.predict_proba(F[te])
        # (c) fusion
        p_fu = 0.5 * p_sk + 0.5 * p_vm
        for name, p in (("skeleton-STGCN", p_sk), ("VideoMAE-probe", p_vm), ("fusion", p_fu)):
            pr = p.argmax(1)
            out.append({"dataset": ds, "split": s, "stream": name, "top1": accuracy_score(y[te], pr), "mean_class_acc": balanced_accuracy_score(y[te], pr),
                        "n_test": int(te.sum()), "n_train": int(tr.sum())})
            probs_store[(s, name)] = (p, y[te])
        out[-3].update({"train_time_s": t_train, "infer_ms_per_clip": 1000 * t_inf})
        # (e) robustness of the skeleton stream (split s)
        for kind in ("flip", "occl20", "occl40", "occl60", "crop75", "crop50", "jitter03", "jitter06", "second"):
            pr = predict(model, perturb(X[te], kind, rng)).argmax(1)
            out.append({"dataset": ds, "split": s, "stream": f"skeleton-STGCN/{kind}", "top1": accuracy_score(y[te], pr), "mean_class_acc": balanced_accuracy_score(y[te], pr), "n_test": int(te.sum())})
        # confusion matrix for split 1 fusion
        if s == 1:
            cm = confusion_matrix(y[te], p_fu.argmax(1))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(RES / f"e1_confusion_{ds}.csv")
            pd.DataFrame(confusion_matrix(y[te], p_sk.argmax(1)), index=classes, columns=classes).to_csv(RES / f"e1_confusion_{ds}_skeleton.csv")
            pd.DataFrame(confusion_matrix(y[te], p_vm.argmax(1)), index=classes, columns=classes).to_csv(RES / f"e1_confusion_{ds}_videomae.csv")
    # (b2) Kinetics-400 zero-shot transfer via label keyword mapping
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", cache_dir=str(HF_CACHE))
    id2label = cfg.id2label
    kw = {"punch": ["punching person", "punching bag"], "kick": ["side kick", "high kick", "drop kicking"], "hit": ["punching person", "slapping"],
          "sword": ["sword fighting"], "sword_exercise": ["sword fighting"], "draw_sword": ["sword fighting"], "fencing": ["fencing"],
          "Punch": ["punching person"], "BoxingPunchingBag": ["punching bag"], "BoxingSpeedBag": ["punching bag"], "Fencing": ["fencing"],
          "TaiChi": ["tai chi"], "SumoWrestling": ["wrestling"], "Nunchucks": ["sword fighting", "capoeira"]}
    cls_ids = {c: [i for i, n in id2label.items() if any(k in n.lower() for k in kw[c])] for c in classes}
    zs = np.stack([L[:, ids].max(1) if ids else np.full(len(L), -1e9) for c, ids in cls_ids.items()], 1)
    pr = zs.argmax(1)
    out.append({"dataset": ds, "split": 0, "stream": "K400-zero-shot(VideoMAE)", "top1": accuracy_score(y, pr), "mean_class_acc": balanced_accuracy_score(y, pr), "n_test": len(y),
                "note": json.dumps({c: [id2label[i] for i in ids] for c, ids in cls_ids.items()})})
    return out, (sub, classes, X, F, y)


def cross_dataset(packs):
    """Train on one dataset, test on the other for shared semantic classes (punch, fencing)."""
    out = []
    shared = {"HMDB51-MA": {"punch": "punch", "fencing": "fencing"}, "UCF101-MA": {"Punch": "punch", "Fencing": "fencing"}}
    def subset(ds):
        sub, classes, X, F, y = packs[ds]
        m = sub.cls.isin(shared[ds].keys()).values
        lab = sub.cls[m].map(shared[ds]).map({"punch": 0, "fencing": 1}).values
        return X[m], F[m], lab
    for src, dst in (("UCF101-MA", "HMDB51-MA"), ("HMDB51-MA", "UCF101-MA")):
        Xs, Fs, ys = subset(src); Xd, Fd, yd = subset(dst)
        model = STGCN(3, 17, COCO_BONES, 2, n_persons=2, base=64, dropout=0.3)
        model = train_model(model, Xs, ys, task="cls", epochs=30, bs=32, lr=2e-3, aug=augment2d, seed=0)
        p_sk = torch.softmax(torch.tensor(predict(model, Xd)), 1).numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(Fs, ys); p_vm = clf.predict_proba(Fd)
        for name, p in (("skeleton-STGCN", p_sk), ("VideoMAE-probe", p_vm), ("fusion", 0.5 * p_sk + 0.5 * p_vm)):
            out.append({"train": src, "test": dst, "stream": name, "top1": accuracy_score(yd, p.argmax(1)), "mean_class_acc": balanced_accuracy_score(yd, p.argmax(1)), "n_test": len(yd)})
    return out


def main():
    ensure_dirs(RES)
    idx = pd.read_csv(DD / "video_index.csv")
    res, packs = [], {}
    for ds in ("HMDB51-MA", "UCF101-MA"):
        o, pk = run_dataset(ds, idx); res += o; packs[ds] = pk
    df = pd.DataFrame(res); df.to_csv(RES / "e1_recognition.csv", index=False)
    print(df[~df.stream.str.contains("/")].groupby(["dataset", "stream"])[["top1", "mean_class_acc"]].mean().round(3))
    print(df[df.stream.str.contains("/")].groupby(["dataset", "stream"])[["top1"]].mean().round(3))
    cd = pd.DataFrame(cross_dataset(packs)); cd.to_csv(RES / "e1_cross_dataset.csv", index=False); print(cd.round(3))


if __name__ == "__main__":
    main()
