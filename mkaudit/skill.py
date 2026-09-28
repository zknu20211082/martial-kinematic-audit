"""Skill assessment and kinematic-knowledge-base (KKB) rule mining with leave-one-subject-out validation
(paper Secs. 1.3, 2.1 and 2.2)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

from .stgcn import STGCN, train_model, predict, augment3d  # imports torch before lightgbm, as in the original code
import lightgbm as lgb

from .config import RES
from .skeleton import BONES
from .stats import bootstrap_ci

NON_FEAT = {"file", "code", "tech", "tech_name", "cond", "cond_name", "trial", "strike_idx", "peak_frame", "pid", "type", "clip",
            "gesture", "direction", "instance", "sensor", "src_fps", "fastest_ee", "peak_speed_strike"}


def feat_cols(df):
    """The 54 numeric kinematic descriptor columns of a descriptor table."""
    return [c for c in df.columns if c not in NON_FEAT and np.issubdtype(df[c].dtype, np.number)]


def loso_regress(df, y, groups, cols, model="ridge"):
    """Leave-one-subject-out regression; returns out-of-fold predictions."""
    pred = np.zeros(len(df))
    X = df[cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if model == "ridge":
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        else:
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1, random_state=0)
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def loso_stgcn(X, y, groups, task="reg", n_folds=None, epochs=30, seed=0):
    """Subject-grouped CV with ST-GCN. n_folds=None -> LOSO."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    folds = [ug] if n_folds == 1 else (np.array_split(ug, n_folds) if n_folds else [[g] for g in ug])
    pred = np.zeros(len(X)) if task == "reg" else np.zeros((len(X), int(y.max()) + 1))
    for fg in folds:
        te = np.isin(groups, fg); tr = ~te
        model = STGCN(3, 19, BONES, 1 if task == "reg" else int(y.max()) + 1, base=32)
        ytr = y[tr]
        if task == "reg":
            mu, sd = ytr.mean(), ytr.std() + 1e-6
            model = train_model(model, X[tr], (ytr - mu) / sd, task="reg", epochs=epochs, aug=augment3d, seed=seed)
            pred[te] = predict(model, X[te]).squeeze(1) * sd + mu
        else:
            model = train_model(model, X[tr], ytr, task="cls", epochs=epochs, aug=augment3d, seed=seed)
            pred[te] = predict(model, X[te])
    return pred


def report_reg(name, y, pred, groups, extra=None):
    r = {"setting": name, "n": int(len(y)), "n_subjects": int(len(np.unique(groups)))}
    r["spearman_segment"] = float(spearmanr(y, pred).correlation)
    r["spearman_segment_ci"] = bootstrap_ci(lambda a, b: spearmanr(a, b).correlation, y, pred)
    r["mae_segment"] = float(mean_absolute_error(y, pred))
    gp = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").mean()
    r["spearman_subject"] = float(spearmanr(gp.y, gp.p).correlation)
    r["pearson_subject"] = float(pearsonr(gp.y, gp.p)[0])
    r["mae_subject"] = float(mean_absolute_error(gp.y, gp.p))
    if extra:
        r.update(extra)
    print(f"{name:55s} rho_seg={r['spearman_segment']:.3f} [{r['spearman_segment_ci'][0]:.2f},{r['spearman_segment_ci'][1]:.2f}] "
          f"MAE={r['mae_segment']:.3f} | rho_subj={r['spearman_subject']:.3f} MAE_subj={r['mae_subject']:.3f}")
    return r


def mine_rules(df, cols, y, group_col, out_name):
    """KKB mining: Spearman of each descriptor with skill, within-gesture-normalised, plus expert-vs-novice effect size.
    Writes <results_dir>/<out_name>."""
    rows = []
    for c in cols:
        x = df[c].values.astype(float)
        if np.nanstd(x) < 1e-9:
            continue
        rho, p = spearmanr(x, y, nan_policy="omit")
        # subject-level (mean per subject) to avoid pseudo-replication
        gp = pd.DataFrame({"g": df[group_col], "x": x, "y": y}).groupby("g").mean()
        rho_s, p_s = spearmanr(gp.x, gp.y)
        hi, lo = x[y >= np.nanpercentile(y, 66)], x[y <= np.nanpercentile(y, 33)]
        d = (np.nanmean(hi) - np.nanmean(lo)) / (np.sqrt((np.nanvar(hi) + np.nanvar(lo)) / 2) + 1e-9)
        rows.append({"descriptor": c, "spearman_segment": rho, "p_segment": p, "spearman_subject": rho_s, "p_subject": p_s,
                     "cohen_d_high_vs_low": d, "mean_high": np.nanmean(hi), "mean_low": np.nanmean(lo)})
    r = pd.DataFrame(rows).sort_values("spearman_subject", key=np.abs, ascending=False)
    r.to_csv(RES / out_name, index=False)
    print(f"KKB rules -> {out_name}; top 8 by |rho_subject|:")
    print(r.head(8)[["descriptor", "spearman_subject", "p_subject", "cohen_d_high_vs_low"]].to_string(index=False))
    return r


def within_gesture_norm(df, cols, key):
    z = df.copy()
    for c in cols:
        z[c] = df.groupby(key)[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    return z
