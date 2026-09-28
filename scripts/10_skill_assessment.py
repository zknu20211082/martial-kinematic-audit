"""E2: skill assessment from motion capture / Kinect (Taichi expert score, Karate grade) with leave-one-subject-out.
Also mines the kinematic knowledge base: per-descriptor Spearman correlation with skill (KKB rules).

Paper: Table 2 (within-gesture z-scored descriptor rows, LightGBM and ST-GCN rows; karate rows), Secs. 2.1-2.2.
Outputs in <results_dir>: e2_skill_{which}.csv/.json, kkb_taichi_rules.csv, kkb_karate_rules.csv,
e2_karate_breakdown.csv; in <work_dir>: pred_taichi_*_stgcn.npy, pred_karate_stgcn.npy.
The paper's result files come from two separate runs: `taichi` (-> e2_skill_taichi.*) and `karate`
(-> e2_skill_karate.*). ST-GCN parts need a CUDA GPU.

Usage: python scripts/10_skill_assessment.py [all|taichi|karate]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols, loso_regress, loso_stgcn, report_reg, mine_rules, within_gesture_norm
from mkaudit.utils import save_json

warnings.filterwarnings("ignore")


def run_taichi():
    results = []
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    cat_map = {"Novice": 0, "Intermediate": 1, "Advanced": 2, "Expert": 3}
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv")
        df = df[df.pid.isin(part.index)].reset_index(drop=True)
        y = part.loc[df.pid, "skill_mean"].values.astype(float)
        ycat = part.loc[df.pid, "category"].map(cat_map).values
        groups = df.pid.values
        cols = feat_cols(df)
        # add gesture one-hot
        G = pd.get_dummies(df.gesture, prefix="g").astype(float)
        dfz = within_gesture_norm(df, cols, "gesture")
        dfz = pd.concat([dfz, G], axis=1); cols_z = cols + list(G.columns)
        if sensor == "qualisys":
            mine_rules(dfz, cols, y, "pid", "kkb_taichi_rules.csv")
        for model in ("ridge", "lgbm"):
            pred = loso_regress(dfz, y, groups, cols_z, model)
            results.append(report_reg(f"Taichi/{sensor}/descriptors+{model}/LOSO", y, pred, groups, {"dataset": "taichi", "sensor": sensor, "model": model}))
        # category classification (4-class ordinal) via LOSO logistic regression on descriptors
        X = np.nan_to_num(dfz[cols_z].values.astype(float))
        pc = np.zeros(len(X), int)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            m.fit(X[tr], ycat[tr]); pc[te] = m.predict(X[te])
        acc = accuracy_score(ycat, pc); within1 = float(np.mean(np.abs(pc - ycat) <= 1))
        subj = pd.DataFrame({"g": groups, "y": ycat, "p": pc}).groupby("g").agg(lambda v: v.mode().iloc[0])
        print(f"Taichi/{sensor}/category-4class/LOSO acc_seg={acc:.3f} within1={within1:.3f} acc_subject={accuracy_score(subj.y, subj.p):.3f}")
        results.append({"setting": f"Taichi/{sensor}/category4/logreg/LOSO", "dataset": "taichi", "sensor": sensor, "model": "logreg",
                        "acc_segment": acc, "acc_within1": within1, "acc_subject": float(accuracy_score(subj.y, subj.p))})
        # ST-GCN on raw sequences (LOSO, 12 folds)
        z = np.load(DD / f"taichi_{sensor}_sequences.npz")
        key2i = {k: i for i, k in enumerate(z["keys"])}
        Xs = z["X"][[key2i[f] for f in df.file]]
        t0 = time.time()
        pred = loso_stgcn(Xs, y, groups, "reg", n_folds=None, epochs=30)
        results.append(report_reg(f"Taichi/{sensor}/ST-GCN/LOSO", y, pred, groups, {"dataset": "taichi", "sensor": sensor, "model": "stgcn", "train_time_s": time.time() - t0}))
        np.save(DD / f"pred_taichi_{sensor}_stgcn.npy", pred)
    # matched-segment comparison qualisys vs kinect (same segment ids)
    return results


def run_karate():
    results = []
    part = pd.read_csv(metadata_path("Kyokushin", "participants.csv")).set_index("code")
    df = pd.read_csv(DD / "karate_strikes.csv")
    df = df[df.code.isin(part.index)].reset_index(drop=True)
    y = part.loc[df.code, "grade_ordinal"].values.astype(float)
    age = part.loc[df.code, "age"].values.astype(float)
    yadv = (y >= 6).astype(int)  # advanced = 4th kyu and above
    groups = df.code.values
    cols = feat_cols(df)
    T = pd.get_dummies(df.tech, prefix="t").astype(float); C = pd.get_dummies(df.cond, prefix="c").astype(float)
    dfz = within_gesture_norm(df, cols, "tech")
    dfz = pd.concat([dfz, T, C], axis=1); cols_z = cols + list(T.columns) + list(C.columns)
    mine_rules(dfz, cols, y, "code", "kkb_karate_rules.csv")
    for model in ("ridge", "lgbm"):
        pred = loso_regress(dfz, y, groups, cols_z, model)
        r = report_reg(f"Karate/descriptors+{model}/LOSO grade", y, pred, groups, {"dataset": "karate", "model": model})
        r["spearman_pred_vs_age"] = float(spearmanr(pred, age).correlation)
        r["spearman_grade_vs_age_subject"] = float(spearmanr(part.grade_ordinal, part.age).correlation)
        results.append(r)
    # beginner vs advanced classification
    X = np.nan_to_num(dfz[cols_z].values.astype(float))
    pp = np.zeros(len(X))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        m.fit(X[tr], yadv[tr]); pp[te] = m.predict_proba(X[te])[:, 1]
    subj = pd.DataFrame({"g": groups, "y": yadv, "p": pp}).groupby("g").mean()
    r = {"setting": "Karate/beginner-vs-advanced/logreg/LOSO", "dataset": "karate", "model": "logreg",
         "auc_segment": float(roc_auc_score(yadv, pp)), "acc_segment": float(accuracy_score(yadv, pp > 0.5)),
         "auc_subject": float(roc_auc_score(subj.y, subj.p)), "acc_subject": float(accuracy_score(subj.y, subj.p > 0.5))}
    print(f"Karate/beginner-vs-advanced/LOSO AUC_seg={r['auc_segment']:.3f} acc_seg={r['acc_segment']:.3f} AUC_subj={r['auc_subject']:.3f} acc_subj={r['acc_subject']:.3f}")
    results.append(r)
    # per-technique / per-condition breakdown of ridge predictions
    pred = loso_regress(dfz, y, groups, cols_z, "ridge")
    bd = []
    for (t, c), sub in pd.DataFrame({"tech": df.tech_name, "cond": df.cond_name, "y": y, "p": pred}).groupby(["tech", "cond"]):
        if len(sub) > 20:
            bd.append({"tech": t, "cond": c, "n": len(sub), "spearman": spearmanr(sub.y, sub.p).correlation})
    pd.DataFrame(bd).to_csv(RES / "e2_karate_breakdown.csv", index=False)
    # ST-GCN, subject-grouped 6-fold
    z = np.load(DD / "karate_sequences.npz"); key2i = {k: i for i, k in enumerate(z["keys"])}
    keys = [f"{f}#{k}" for f, k in zip(df.file, df.strike_idx)]
    Xs = z["X"][[key2i[k] for k in keys]]
    t0 = time.time()
    pred = loso_stgcn(Xs, y, groups, "reg", n_folds=6, epochs=25)
    r = report_reg("Karate/ST-GCN/6-fold-by-subject grade", y, pred, groups, {"dataset": "karate", "model": "stgcn", "train_time_s": time.time() - t0})
    r["spearman_pred_vs_age"] = float(spearmanr(pred, age).correlation)
    results.append(r)
    np.save(DD / "pred_karate_stgcn.npy", pred)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["all", "taichi", "karate"])
    which = ap.parse_args().which
    ensure_dirs(DD, RES)
    res = []
    if which in ("all", "taichi"):
        res += run_taichi()
    if which in ("all", "karate"):
        res += run_karate()
    pd.DataFrame(res).to_csv(RES / f"e2_skill_{which}.csv", index=False)
    save_json(res, RES / f"e2_skill_{which}.json")
