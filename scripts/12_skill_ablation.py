"""E5 ablations for skill assessment: descriptor groups, within-gesture normalisation, training-data scale, sensor fusion.

Paper: Table 2 (karate row "without height & duration"), Sec. 2.2 (adults-only analyses) and the ablation numbers
quoted in Sec. 2.1. Outputs in <results_dir>: e5_ablation_taichi.csv, e5_ablation_karate.csv. CPU only.

Usage: python scripts/12_skill_ablation.py [all|taichi|karate]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols, loso_regress, within_gesture_norm, report_reg

warnings.filterwarnings("ignore")

GROUPS = {
    "speed": lambda c: "speed" in c or "peak_time" in c or "path_len" in c,
    "angles": lambda c: any(k in c for k in ("knee", "hip_", "elbow", "sep")),
    "posture": lambda c: any(k in c for k in ("trunk", "stance", "toe_max", "kick_height", "vert_range", "yaw")),
    "smoothness": lambda c: any(k in c for k in ("ldj", "n_speed_peaks", "speed_mean_over", "sym_", "head_mean")),
}


def taichi():
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    out = []
    dfs = {}
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        dfs[sensor] = df
        y = part.loc[df.pid, "skill_mean"].values.astype(float); groups = df.pid.values
        cols = feat_cols(df); G = pd.get_dummies(df.gesture, prefix="g").astype(float)
        dfz = pd.concat([within_gesture_norm(df, cols, "gesture"), G], axis=1)
        base = report_reg(f"Taichi/{sensor}/all descriptors (ridge)", y, loso_regress(dfz, y, groups, cols + list(G.columns)), groups, {"sensor": sensor, "ablation": "full"})
        out.append(base)
        for g, fn in GROUPS.items():
            sel = [c for c in cols if fn(c)]
            out.append(report_reg(f"Taichi/{sensor}/only-{g} ({len(sel)})", y, loso_regress(dfz, y, groups, sel + list(G.columns)), groups, {"sensor": sensor, "ablation": f"only_{g}", "n_feat": len(sel)}))
            rest = [c for c in cols if not fn(c)]
            out.append(report_reg(f"Taichi/{sensor}/without-{g}", y, loso_regress(dfz, y, groups, rest + list(G.columns)), groups, {"sensor": sensor, "ablation": f"without_{g}", "n_feat": len(rest)}))
        raw = pd.concat([df, G], axis=1)
        out.append(report_reg(f"Taichi/{sensor}/no within-gesture normalisation", y, loso_regress(raw, y, groups, cols + list(G.columns)), groups, {"sensor": sensor, "ablation": "no_gesture_norm"}))
        out.append(report_reg(f"Taichi/{sensor}/no gesture one-hot", y, loso_regress(dfz, y, groups, cols), groups, {"sensor": sensor, "ablation": "no_gesture_onehot"}))
        # training-data scale: subsample training segments (LOSO, keep all test)
        rng = np.random.default_rng(0)
        for frac in (0.1, 0.25, 0.5):
            pred = np.zeros(len(df))
            X = np.nan_to_num(dfz[cols + list(G.columns)].values.astype(float))
            from sklearn.linear_model import Ridge; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
            for gg in np.unique(groups):
                tr = np.where(groups != gg)[0]; te = groups == gg
                tr = rng.choice(tr, max(20, int(len(tr) * frac)), replace=False)
                m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
            out.append(report_reg(f"Taichi/{sensor}/train-fraction {frac}", y, pred, groups, {"sensor": sensor, "ablation": f"train_frac_{frac}"}))
    # sensor fusion on matched segments
    q, k = dfs["qualisys"], dfs["kinect"]
    common = sorted(set(q.file) & set(k.file))
    qm = q.set_index("file").loc[common].reset_index(); km = k.set_index("file").loc[common].reset_index()
    y = part.loc[qm.pid, "skill_mean"].values.astype(float); groups = qm.pid.values
    cols = feat_cols(qm); G = pd.get_dummies(qm.gesture, prefix="g").astype(float)
    qz = within_gesture_norm(qm, cols, "gesture"); kz = within_gesture_norm(km, cols, "gesture")
    fused = pd.concat([qz[cols].add_prefix("q_"), kz[cols].add_prefix("k_"), G], axis=1)
    out.append(report_reg("Taichi/matched/qualisys only", y, loso_regress(pd.concat([qz, G], axis=1), y, groups, cols + list(G.columns)), groups, {"sensor": "matched_qualisys", "ablation": "matched"}))
    out.append(report_reg("Taichi/matched/kinect only", y, loso_regress(pd.concat([kz, G], axis=1), y, groups, cols + list(G.columns)), groups, {"sensor": "matched_kinect", "ablation": "matched"}))
    out.append(report_reg("Taichi/matched/qualisys+kinect fusion", y, loso_regress(fused, y, groups, list(fused.columns)), groups, {"sensor": "fusion", "ablation": "matched", "n_matched": len(common)}))
    pd.DataFrame(out).to_csv(RES / "e5_ablation_taichi.csv", index=False)


def karate():
    part = pd.read_csv(metadata_path("Kyokushin", "participants.csv")).set_index("code")
    df = pd.read_csv(DD / "karate_strikes.csv"); df = df[df.code.isin(part.index)].reset_index(drop=True)
    y = part.loc[df.code, "grade_ordinal"].values.astype(float); groups = df.code.values
    cols = feat_cols(df); T = pd.get_dummies(df.tech, prefix="t").astype(float); C = pd.get_dummies(df.cond, prefix="c").astype(float)
    dfz = pd.concat([within_gesture_norm(df, cols, "tech"), T, C], axis=1); oh = list(T.columns) + list(C.columns)
    out = [report_reg("Karate/all descriptors (ridge)", y, loso_regress(dfz, y, groups, cols + oh), groups, {"ablation": "full"})]
    for g, fn in GROUPS.items():
        sel = [c for c in cols if fn(c)]; rest = [c for c in cols if not fn(c)]
        out.append(report_reg(f"Karate/only-{g} ({len(sel)})", y, loso_regress(dfz, y, groups, sel + oh), groups, {"ablation": f"only_{g}", "n_feat": len(sel)}))
        out.append(report_reg(f"Karate/without-{g}", y, loso_regress(dfz, y, groups, rest + oh), groups, {"ablation": f"without_{g}", "n_feat": len(rest)}))
    # confound controls: drop anthropometric/duration features; adults only (age >= 17) where grade is not tied to growth
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
    anthro = [c for c in cols if c in ("height_proxy_m", "duration_s")]
    cols_na = [c for c in cols if c not in anthro]
    r = report_reg("Karate/without height & duration", y, loso_regress(dfz, y, groups, cols_na + oh), groups, {"ablation": "no_anthropometric"})
    age = part.loc[df.code, "age"].values.astype(float); r["spearman_pred_vs_age"] = float(spearmanr(loso_regress(dfz, y, groups, cols_na + oh), age).correlation); out.append(r)
    adult = (age >= 17)
    sub = dfz[adult].reset_index(drop=True); ya = y[adult]; ga = groups[adult]
    r = report_reg("Karate/adults only (age>=17), without height & duration", ya, loso_regress(sub, ya, ga, cols_na + oh), ga, {"ablation": "adults_only", "n_subjects": int(len(np.unique(ga)))}); out.append(r)
    ydan = (ya >= 10).astype(int)
    X = np.nan_to_num(sub[cols_na + oh].values.astype(float)); pp = np.zeros(len(X))
    for gg in np.unique(ga):
        tr, te = ga != gg, ga == gg
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], ydan[tr]); pp[te] = m.predict_proba(X[te])[:, 1]
    sj = pd.DataFrame({"g": ga, "y": ydan, "p": pp}).groupby("g").mean()
    out.append({"setting": "Karate/adults only: dan vs kyu (logreg, LOSO)", "ablation": "adults_dan_vs_kyu", "n": int(len(X)), "n_subjects": int(len(np.unique(ga))),
                "auc_segment": float(roc_auc_score(ydan, pp)), "auc_subject": float(roc_auc_score(sj.y, sj.p))})
    print(out[-1])
    # per-condition training (does the opponent condition help?)
    for cond in ("E01", "E02", "E03", "E04"):
        m = (df.cond == cond).values
        sub = dfz[m].reset_index(drop=True)
        out.append(report_reg(f"Karate/condition {cond} only", y[m], loso_regress(sub, y[m], groups[m], cols + oh), groups[m], {"ablation": f"cond_{cond}"}))
    pd.DataFrame(out).to_csv(RES / "e5_ablation_karate.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["all", "taichi", "karate"])
    w = ap.parse_args().which
    ensure_dirs(RES)
    if w in ("all", "taichi"):
        taichi()
    if w in ("all", "karate"):
        karate()
