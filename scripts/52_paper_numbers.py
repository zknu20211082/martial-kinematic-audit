"""Collect every number quoted in the paper into results/paper_numbers.json (single source of truth for the manuscript).

Paper: Table 1 (dataset statistics), Table 2, Table 3(a), Table 4 and the numbers quoted in Secs. 1-2.
Inputs: <work_dir> tables (01, 02, 05, 11), <results_dir> outputs of 10-13, 20, 21, 31, 40, 50 and
paper_group_ablation_raw.json written by 51_make_figures.py. Output: <results_dir>/paper_numbers.json. CPU only."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols
from mkaudit.stats import bootstrap_ci
from mkaudit.utils import save_json

warnings.filterwarnings("ignore")


def loso_pred(df, cols_, y, groups):
    X = np.nan_to_num(df[cols_].values.astype(float)); pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        pred[te] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr]).predict(X[te])
    return pred


def subj(groups, y, p):
    s = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").mean()
    return float(spearmanr(s.y, s.p).correlation), float(np.mean(np.abs(s.y - s.p)))


def main():
    ensure_dirs(RES)
    N = {}
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    dq = pd.read_csv(DD / "taichi_qualisys.csv")
    cols = feat_cols(dq)
    N["n_descriptors"] = len(cols); N["descriptor_list"] = cols
    N["taichi_participants"] = {"n": len(part), "male": int((part.gender == "M").sum()), "female": int((part.gender == "F").sum()),
                                "age_min": int(part.age.min()), "age_max": int(part.age.max()), "practice_min": float(part.practice_years.min()),
                                "practice_max": float(part.practice_years.max()), "score_min": float(part.skill_mean.min()), "score_max": float(part.skill_mean.max())}
    kp = pd.read_csv(metadata_path("Kyokushin", "participants.csv"))
    N["karate_participants"] = {"n": len(kp), "male": int((kp.gender == "M").sum()), "female": int((kp.gender == "F").sum()), "age_min": int(kp.age.min()), "age_max": int(kp.age.max()),
                                "n_adv_4kyu_plus": int((kp.grade_ordinal >= 6).sum()), "n_adult_17plus": int((kp.age >= 17).sum()), "n_child_14minus": int((kp.age <= 14).sum()),
                                "n_dan": int((kp.grade_ordinal >= 10).sum())}
    ks = pd.read_csv(DD / "karate_strikes.csv")
    N["karate_segments"] = {"total": len(ks), "by_cond": ks.groupby("cond_name").size().to_dict(), "by_tech": ks.groupby("tech_name").size().to_dict(), "files": int(ks.file.nunique())}
    kf = pd.read_csv(DD / "karate_files.csv"); N["karate_rates"] = kf.rate.value_counts().to_dict(); N["karate_files_all_virtual"] = float((kf.virtual_joints == 19).mean())
    N["taichi_segments"] = {"qualisys": len(dq), "kinect": len(pd.read_csv(DD / "taichi_kinect.csv"))}

    rng = np.random.default_rng(0)
    raw = {}
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        y = part.loc[df.pid, "skill_mean"].values.astype(float); groups = df.pid.values
        G = pd.get_dummies(df.gesture, prefix="g").astype(float); dfx = pd.concat([df, G], axis=1); cc = cols + list(G.columns)
        pred = np.load(DD / f"pred_taichi_{sensor}_ridge_raw.npy")
        r = {"seg_rho": float(spearmanr(y, pred).correlation), "seg_rho_ci": bootstrap_ci(lambda a, b: spearmanr(a, b).correlation, y, pred),
             "seg_mae": float(np.mean(np.abs(y - pred)))}
        r["subj_rho"], r["subj_mae"] = subj(groups, y, pred)
        s = pd.DataFrame({"g": groups, "p": pred}).groupby("g").p.mean()
        r["category_mean_pred"] = s.groupby(part.loc[s.index, "category"]).mean().round(3).to_dict()
        r["category_mean_true"] = part.groupby("category").skill_mean.mean().round(3).to_dict()
        # training fraction (raw config)
        tf = {}
        X = np.nan_to_num(dfx[cc].values.astype(float))
        for frac in (0.1, 0.25, 0.5):
            vals = []
            for rep in range(5):
                p2 = np.zeros(len(y))
                for g in np.unique(groups):
                    tr = np.where(groups != g)[0]; te = groups == g
                    tr = rng.choice(tr, max(20, int(len(tr) * frac)), replace=False)
                    p2[te] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr]).predict(X[te])
                vals.append(subj(groups, y, p2)[0])
            tf[str(frac)] = [float(np.mean(vals)), float(np.std(vals))]
        r["train_fraction_subj_rho_mean_sd"] = tf
        raw[sensor] = r
        # per gesture segment-level rho
        r["per_gesture_seg_rho"] = {g: float(spearmanr(sub.y, sub.p).correlation) for g, sub in pd.DataFrame({"gesture": df.gesture, "y": y, "p": pred}).groupby("gesture")}
    N["taichi_raw"] = raw
    # matched-sensor fusion (raw)
    q = pd.read_csv(DD / "taichi_qualisys.csv"); k = pd.read_csv(DD / "taichi_kinect.csv")
    common_ = sorted(set(q.file) & set(k.file)); qm = q.set_index("file").loc[common_].reset_index(); km = k.set_index("file").loc[common_].reset_index()
    y = part.loc[qm.pid, "skill_mean"].values.astype(float); groups = qm.pid.values
    G = pd.get_dummies(qm.gesture, prefix="g").astype(float)
    fused = pd.concat([qm[cols].add_prefix("q_"), km[cols].add_prefix("k_"), G], axis=1)
    N["taichi_matched_raw"] = {"n": len(common_),
                               "qualisys": subj(groups, y, loso_pred(pd.concat([qm, G], axis=1), cols + list(G.columns), y, groups))[0],
                               "kinect": subj(groups, y, loso_pred(pd.concat([km, G], axis=1), cols + list(G.columns), y, groups))[0],
                               "fusion": subj(groups, y, loso_pred(fused, list(fused.columns), y, groups))[0]}
    N["group_ablation_raw"] = json.load(open(RES / "paper_group_ablation_raw.json", encoding="utf-8"))
    # skill tables
    for f in ("e2_skill_taichi.csv", "e2_skill_taichi_raw.csv", "e2_skill_karate.csv", "e2_stgcn_recheck.csv", "e2_karate_confound.csv", "e5_ablation_karate.csv"):
        N[f.replace(".csv", "")] = json.loads(pd.read_csv(RES / f).to_json(orient="records"))
    # E1
    e1 = pd.read_csv(RES / "e1_recognition.csv"); m = e1[~e1.stream.str.contains("/")]
    N["e1_mean"] = json.loads(m.groupby(["dataset", "stream"]).agg(top1=("top1", "mean"), sd=("top1", "std"), mca=("mean_class_acc", "mean"), n_test=("n_test", "mean"), n_train=("n_train", "mean")).reset_index().to_json(orient="records"))
    N["e1_split1"] = json.loads(m[m.split == 1][["dataset", "stream", "top1", "n_test"]].to_json(orient="records"))
    pe = e1[e1.stream.str.contains("/")].copy(); pe["k"] = pe.stream.str.split("/").str[1]
    N["e1_perturb"] = json.loads(pe.groupby(["dataset", "k"]).top1.mean().unstack(0).reset_index().to_json(orient="records"))
    N["e1_cross"] = json.loads(pd.read_csv(RES / "e1_cross_dataset.csv").to_json(orient="records"))
    per = {}
    cm = pd.read_csv(RES / "e1_confusion_HMDB51-MA.csv", index_col=0); classes = list(cm.index)
    for nm, f in (("skeleton", "e1_confusion_HMDB51-MA_skeleton.csv"), ("videomae", "e1_confusion_HMDB51-MA_videomae.csv"), ("fusion", "e1_confusion_HMDB51-MA.csv")):
        c = pd.read_csv(RES / f, index_col=0).loc[classes, classes].values
        per[nm] = dict(zip(classes, (np.diag(c) / c.sum(1)).round(3).tolist()))
    N["e1_hmdb_per_class_recall_split1"] = per
    cmu = pd.read_csv(RES / "e1_confusion_UCF101-MA.csv", index_col=0)
    N["e1_ucf_per_class_recall_split1_fusion"] = dict(zip(cmu.index, (np.diag(cmu.values) / cmu.values.sum(1)).round(3).tolist()))
    N["e4_viewpoint"] = json.loads(pd.read_csv(RES / "e4_viewpoint.csv").to_json(orient="records"))
    N["e3"] = json.loads(pd.read_csv(RES / "e3_summary_rescored.csv").to_json(orient="records"))
    st = json.load(open(RES / "paper_stats.json", encoding="utf-8"))
    N["e3_mcnemar"] = st["audit_rescored"]; N["taichi_perm"] = {k: st[k] for k in ("taichi_qualisys", "taichi_kinect")}; N["karate_age"] = st["karate"]
    N["e6"] = json.loads(pd.read_csv(RES / "e6_efficiency.csv").to_json(orient="records"))
    N["synth"] = {"n_videos": int(len(pd.read_csv(DD / "synth_videos.csv"))), "by_source": pd.read_csv(DD / "synth_videos.csv").groupby("source").size().to_dict()}
    z = np.load(DD / "synth_viewpoint_karate.npz"); N["synth_viewpoint_n"] = int(z["X"].shape[0])
    idx = pd.read_csv(DD / "video_index.csv")
    N["video_index"] = {ds: {"n": int((idx.dataset == ds).sum()), **{f"split{s}": idx[idx.dataset == ds][f"split{s}"].value_counts().to_dict() for s in (1, 2, 3)}} for ds in idx.dataset.unique()}
    save_json(N, RES / "paper_numbers.json")
    print(json.dumps({k: v for k, v in N.items() if k in ("n_descriptors", "taichi_participants", "karate_participants", "karate_segments", "karate_rates", "taichi_segments", "taichi_raw", "taichi_matched_raw", "e1_split1", "e1_hmdb_per_class_recall_split1", "e1_ucf_per_class_recall_split1_fusion", "synth", "synth_viewpoint_n", "video_index")}, ensure_ascii=False, indent=1, default=str))
    e3 = pd.DataFrame(N["e3"]); print(e3[["set", "condition", "n", "KF", "KF_ci", "tech_ci", "verifiable_rate", "halluc_rate", "top_pred_label", "top_pred_share", "pred_entropy_bits"]].to_string())


if __name__ == "__main__":
    main()
