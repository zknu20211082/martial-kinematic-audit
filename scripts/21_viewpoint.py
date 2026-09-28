"""Viewpoint experiment: technique recognition from 2D projections of karate mocap at 8 camera azimuths.
Protocols: cross-view (train on cardinal views, test on diagonal views and vice versa), leave-one-view-out, all-view mixed.
All protocols are additionally subject-disjoint (train/test athletes never overlap).

Paper: Fig. 3(d) and Sec. 2.4. Needs <work_dir>/synth_viewpoint_karate.npz from 05 and a CUDA GPU.
Output: <results_dir>/e4_viewpoint.csv."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import json
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from mkaudit.config import DD, RES, ensure_dirs
from mkaudit.stgcn import STGCN, train_model, predict
from mkaudit.recognition import COCO_BONES


def main():
    ensure_dirs(RES)
    z = np.load(DD / "synth_viewpoint_karate.npz"); X = torch.tensor(z["X"]); meta = pd.DataFrame(json.loads(str(z["meta"])))
    techs = sorted(meta.tech.unique()); y = meta.tech.map({t: i for i, t in enumerate(techs)}).values
    subjects = np.array(sorted(meta.code.unique())); rng = np.random.default_rng(0); rng.shuffle(subjects)
    test_subj = set(subjects[: len(subjects) // 3])
    is_test_subj = meta.code.isin(test_subj).values
    az = meta.az.values
    out = []
    def fit_eval(tr, te, name, epochs=25):
        m = STGCN(3, 17, COCO_BONES, len(techs), n_persons=1, base=32, dropout=0.3)
        m = train_model(m, X[tr], y[tr], task="cls", epochs=epochs, bs=64, lr=2e-3, aug=None, seed=0)
        pr = predict(m, X[te]).argmax(1)
        r = {"protocol": name, "n_train": int(tr.sum()), "n_test": int(te.sum()), "top1": accuracy_score(y[te], pr), "mean_class_acc": balanced_accuracy_score(y[te], pr)}
        for a in sorted(np.unique(az[te])):
            mm = az[te] == a
            r[f"acc_az{a}"] = accuracy_score(y[te][mm], pr[mm])
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}); out.append(r); return m
    card = np.isin(az, [0, 90, 180, 270]); diag = ~card
    fit_eval(~is_test_subj & card, is_test_subj & diag, "train-cardinal/test-diagonal")
    fit_eval(~is_test_subj & diag, is_test_subj & card, "train-diagonal/test-cardinal")
    fit_eval(~is_test_subj, is_test_subj, "all-views-mixed (subject-disjoint)")
    for a in (0, 90, 45):
        fit_eval(~is_test_subj & (az == a), is_test_subj, f"train-single-view-{a}/test-all-views")
    fit_eval(~is_test_subj & (az != 135), is_test_subj & (az == 135), "leave-one-view-out-135")
    pd.DataFrame(out).to_csv(RES / "e4_viewpoint.csv", index=False)


if __name__ == "__main__":
    main()
