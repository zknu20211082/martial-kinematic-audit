"""Age/anthropometry confound analysis for the Karate grade task, and the best Taichi configuration (raw descriptors).

Paper: Table 2 (raw-descriptor Taichi rows), Fig. 2(a)-(d) inputs, Sec. 2.2 (age confound).
Outputs in <results_dir>: e2_karate_confound.csv, kkb_karate_rules_age_controlled.csv (p_partial is added by 50),
e2_skill_taichi_raw.csv, kkb_taichi_rules_raw.csv, e2_taichi_breakdown_{qualisys,kinect}.csv;
in <work_dir>: pred_taichi_{qualisys,kinect}_ridge_raw.npy (used by 51 and 52). CPU only."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols, loso_regress, within_gesture_norm, report_reg, mine_rules
from mkaudit.stats import partial_spearman

warnings.filterwarnings("ignore")


def karate():
    part = pd.read_csv(metadata_path("Kyokushin", "participants.csv")).set_index("code")
    df = pd.read_csv(DD / "karate_strikes.csv"); df = df[df.code.isin(part.index)].reset_index(drop=True)
    y = part.loc[df.code, "grade_ordinal"].values.astype(float); age = part.loc[df.code, "age"].values.astype(float); groups = df.code.values
    cols = feat_cols(df); T = pd.get_dummies(df.tech, prefix="t").astype(float); C = pd.get_dummies(df.cond, prefix="c").astype(float)
    oh = list(T.columns) + list(C.columns)
    out = []
    for name, dfx in (("within-tech z-score", pd.concat([within_gesture_norm(df, cols, "tech"), T, C], axis=1)), ("raw descriptors", pd.concat([df, T, C], axis=1))):
        pred = loso_regress(dfx, y, groups, cols + oh)
        sub = pd.DataFrame({"g": groups, "y": y, "p": pred, "age": age}).groupby("g").mean()
        r = {"features": name, "rho_subject": float(spearmanr(sub.y, sub.p).correlation),
             "rho_pred_vs_age_subject": float(spearmanr(sub.p, sub.age).correlation),
             "rho_grade_vs_age_subject": float(spearmanr(sub.y, sub.age).correlation),
             "partial_rho_subject_ctrl_age": partial_spearman(sub.p, sub.y, sub.age)}
        # children only (age <= 14): 21 athletes, grades 4-9 kyu
        kid = (age <= 14)
        pk = loso_regress(dfx[kid].reset_index(drop=True), y[kid], groups[kid], cols + oh)
        sk = pd.DataFrame({"g": groups[kid], "y": y[kid], "p": pk, "age": age[kid]}).groupby("g").mean()
        r["children_n_subjects"] = int(len(sk)); r["children_rho_subject"] = float(spearmanr(sk.y, sk.p).correlation)
        r["children_partial_rho_ctrl_age"] = partial_spearman(sk.p, sk.y, sk.age)
        # training-years as alternative expertise target
        yrs = part.loc[df.code, "training_years"].values.astype(float)
        py = loso_regress(dfx, yrs, groups, cols + oh)
        sy = pd.DataFrame({"g": groups, "y": yrs, "p": py, "age": age}).groupby("g").mean()
        r["years_rho_subject"] = float(spearmanr(sy.y, sy.p).correlation); r["years_partial_rho_ctrl_age"] = partial_spearman(sy.p, sy.y, sy.age)
        print(r); out.append(r)
    # descriptor-level partial correlations with grade controlling for age (subject means)
    sub = df.groupby("code")[cols].mean(); sub["grade"] = part.loc[sub.index, "grade_ordinal"]; sub["age"] = part.loc[sub.index, "age"]
    rows = []
    for c in cols:
        if sub[c].std() < 1e-9:
            continue
        rows.append({"descriptor": c, "rho_grade": float(spearmanr(sub[c], sub.grade).correlation), "rho_age": float(spearmanr(sub[c], sub.age).correlation),
                     "partial_rho_grade_ctrl_age": partial_spearman(sub[c], sub.grade, sub.age)})
    rr = pd.DataFrame(rows).sort_values("partial_rho_grade_ctrl_age", key=np.abs, ascending=False)
    rr.to_csv(RES / "kkb_karate_rules_age_controlled.csv", index=False)
    print(rr.head(10).round(3).to_string(index=False))
    pd.DataFrame(out).to_csv(RES / "e2_karate_confound.csv", index=False)


def taichi_raw():
    """Best Taichi configuration: raw descriptors + gesture one-hot (no within-gesture normalisation)."""
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    out = []
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        y = part.loc[df.pid, "skill_mean"].values.astype(float); groups = df.pid.values
        cols = feat_cols(df); G = pd.get_dummies(df.gesture, prefix="g").astype(float); dfx = pd.concat([df, G], axis=1)
        for model in ("ridge", "lgbm"):
            pred = loso_regress(dfx, y, groups, cols + list(G.columns), model)
            out.append(report_reg(f"Taichi/{sensor}/raw descriptors+{model}/LOSO", y, pred, groups, {"dataset": "taichi", "sensor": sensor, "model": model, "features": "raw"}))
            if model == "ridge":
                np.save(DD / f"pred_taichi_{sensor}_ridge_raw.npy", pred)
        if sensor == "qualisys":
            mine_rules(dfx, cols, y, "pid", "kkb_taichi_rules_raw.csv")
        # per-gesture breakdown
        pred = loso_regress(dfx, y, groups, cols + list(G.columns), "ridge")
        bd = [{"gesture": g, "n": len(s), "spearman_segment": spearmanr(s.y, s.p).correlation} for g, s in pd.DataFrame({"gesture": df.gesture, "y": y, "p": pred}).groupby("gesture")]
        pd.DataFrame(bd).to_csv(RES / f"e2_taichi_breakdown_{sensor}.csv", index=False)
    pd.DataFrame(out).to_csv(RES / "e2_skill_taichi_raw.csv", index=False)


if __name__ == "__main__":
    ensure_dirs(DD, RES)
    karate(); taichi_raw()
