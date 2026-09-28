"""Extra statistics for the paper: permutation tests (Taichi LOSO), age-partial correlations (Karate),
bootstrap CIs and exact McNemar tests (video-LLM audit, scoring at generation time), recognition split statistics.

Paper: Table 2 (permutation p-values, partial correlation, adult subset), Fig. 2(a)(b)(e)(f), Secs. 2.1-2.2.
Inputs: <work_dir> descriptor tables, <results_dir> outputs of 11, 20 and 30. Outputs in <results_dir>:
paper_stats.json, paper_karate_subject_preds.csv, and the column p_partial added to kkb_karate_rules_age_controlled.csv.
The re-scored audit statistics (keys 'audit_rescored', 'audit_summary_rescored') are added by 31_vlm_rescore.py and
are kept when this script rewrites paper_stats.json. CPU only (the 2 x 1000 permutations take a few minutes)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import json, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols, loso_regress, within_gesture_norm
from mkaudit.stats import boot_ci, mcnemar_exact as mcnemar, partial_spearman_test as partial, partial_p_value
from mkaudit.verifier import kf
from mkaudit.utils import save_json

warnings.filterwarnings("ignore")
N_PERM = 1000


def loso_ridge_fast(X, y, groups):
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def subj_rho(groups, y, p):
    s = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").mean()
    r = spearmanr(s.y, s.p)
    return float(r.correlation), float(r.pvalue)


def main():
    ensure_dirs(RES)
    OUT = {}
    rng = np.random.default_rng(2026)

    # ---------------- (a) Taichi permutation tests ----------------
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        groups = df.pid.values
        cols = feat_cols(df); G = pd.get_dummies(df.gesture, prefix="g").astype(float)
        X = np.nan_to_num(pd.concat([df[cols], G], axis=1).values.astype(float))
        pids = np.array(sorted(part.index)); skill = part.loc[pids, "skill_mean"].values
        y = part.loc[df.pid, "skill_mean"].values.astype(float)
        t0 = time.time()
        p_obs = loso_ridge_fast(X, y, groups)
        rho_obs, p_param = subj_rho(groups, y, p_obs)
        seg_rho = float(spearmanr(y, p_obs).correlation)
        null = []
        for _ in range(N_PERM):
            perm = dict(zip(pids, rng.permutation(skill)))
            yp = np.array([perm[g] for g in groups])
            null.append(subj_rho(groups, yp, loso_ridge_fast(X, yp, groups))[0])
        null = np.array(null)
        p_perm = (1 + np.sum(null >= rho_obs)) / (1 + N_PERM)
        OUT[f"taichi_{sensor}"] = {"rho_subject": rho_obs, "p_spearman_subject": p_param, "rho_segment": seg_rho, "p_permutation": float(p_perm),
                                   "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)), "n_perm": N_PERM, "secs": time.time() - t0}
        print(sensor, OUT[f"taichi_{sensor}"])
        # per-category mean predicted score
        s = pd.DataFrame({"g": groups, "p": p_obs}).groupby("g").p.mean()
        OUT[f"taichi_{sensor}"]["category_means"] = s.groupby(part.loc[s.index, "category"]).mean().to_dict()

    # ---------------- (b) Karate age-partial correlations ----------------
    kp = pd.read_csv(metadata_path("Kyokushin", "participants.csv")).set_index("code")
    df = pd.read_csv(DD / "karate_strikes.csv"); df = df[df.code.isin(kp.index)].reset_index(drop=True)
    y = kp.loc[df.code, "grade_ordinal"].values.astype(float); groups = df.code.values
    cols = feat_cols(df); T = pd.get_dummies(df.tech, prefix="t").astype(float); C = pd.get_dummies(df.cond, prefix="c").astype(float)
    dfz = pd.concat([within_gesture_norm(df, cols, "tech"), T, C], axis=1)
    pred = loso_regress(dfz, y, groups, cols + list(T.columns) + list(C.columns), "ridge")
    s = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").mean()
    s["age"] = kp.loc[s.index, "age"]; s["grade"] = kp.loc[s.index, "grade"]; s["years"] = kp.loc[s.index, "training_years"]
    s.to_csv(RES / "paper_karate_subject_preds.csv")

    r_all = spearmanr(s.y, s.p); r_age_pred = spearmanr(s.p, s.age); r_age_grade = spearmanr(s.y, s.age)
    pr, pp = partial(s.p, s.y, s.age)
    adults = s[s.age >= 17]; kids = s[s.age <= 14]
    OUT["karate"] = {"rho_subject": float(r_all.correlation), "p": float(r_all.pvalue), "rho_pred_age": float(r_age_pred.correlation), "rho_grade_age": float(r_age_grade.correlation),
                     "partial_rho_ctrl_age": pr, "partial_p": pp, "n_adults": int(len(adults)), "rho_adults": float(spearmanr(adults.y, adults.p).correlation),
                     "p_adults": float(spearmanr(adults.y, adults.p).pvalue), "n_children": int(len(kids)), "rho_children": float(spearmanr(kids.y, kids.p).correlation),
                     "p_children": float(spearmanr(kids.y, kids.p).pvalue)}
    rules = pd.read_csv(RES / "kkb_karate_rules_age_controlled.csv")
    n = 37
    rules["p_partial"] = [partial_p_value(r, n) for r in rules.partial_rho_grade_ctrl_age]
    rules.to_csv(RES / "kkb_karate_rules_age_controlled.csv", index=False)
    OUT["karate"]["rules_age_controlled_top"] = rules.head(6)[["descriptor", "rho_grade", "rho_age", "partial_rho_grade_ctrl_age", "p_partial"]].round(4).to_dict("records")
    print("karate", {k: v for k, v in OUT["karate"].items() if k != "rules_age_controlled_top"})
    print(rules.head(6).round(3).to_string(index=False))

    # Taichi rule p-values (subject level, n=12) for the raw-descriptor rules
    tr = pd.read_csv(RES / "kkb_taichi_rules_raw.csv")
    OUT["taichi_rules_top"] = tr.head(10)[["descriptor", "spearman_subject", "p_subject", "cohen_d_high_vs_low", "mean_high", "mean_low"]].round(4).to_dict("records")

    # ---------------- (c) Video-LLM audit statistics (scoring at generation time) ----------------
    FIELDS = ["technique", "n_people", "striking_limb", "peak_foot_height", "body_turn", "knee_extended_at_peak"]
    COVERED = ["n_people", "striking_limb", "peak_foot_height"]; HELD = ["body_turn", "knee_extended_at_peak"]
    r = pd.concat([pd.read_csv(RES / "e3_vlm_synth.csv"), pd.read_csv(RES / "e3_vlm_wild.csv")])
    audit = []
    for src, sub in r.groupby("source"):
        z = sub[sub.condition == "zeroshot"].set_index("clip"); g = sub[sub.condition == "grounded"].set_index("clip")
        common = z.index.intersection(g.index); z = z.loc[common]; g = g.loc[common]
        row = {"set": src, "n_pairs": int(len(common))}
        for name, d in (("zeroshot", z.reset_index()), ("grounded", g.reset_index())):
            allk = [f for f in FIELDS if f != "technique"]
            row[f"{name}_KF"] = float(kf(d, allk)); row[f"{name}_KF_ci"] = tuple(boot_ci(d, lambda s_: kf(s_, allk), rng))
            row[f"{name}_KF_covered"] = float(kf(d, COVERED)); row[f"{name}_KF_held"] = float(kf(d, HELD))
            row[f"{name}_tech_acc"] = float((d.res_technique == "pass").mean())
            row[f"{name}_tech_ci"] = tuple(boot_ci(d, lambda s_: (s_.res_technique == "pass").mean(), rng))
            vc = d.pred_technique.astype(str).value_counts(normalize=True)
            row[f"{name}_top_pred_label"] = str(vc.index[0]); row[f"{name}_top_pred_share"] = float(vc.iloc[0])
            pdist = vc.values; row[f"{name}_pred_entropy_bits"] = float(-(pdist * np.log2(pdist)).sum())
            allres = pd.concat([d[f"res_{f}"] for f in allk])
            row[f"{name}_halluc_rate"] = float((allres == "fail").sum() / len(allres))
        for f in FIELDS:
            m = z[f"res_{f}"].isin(["pass", "fail"]) & g[f"res_{f}"].isin(["pass", "fail"])
            if m.sum() == 0:
                continue
            a = (z[f"res_{f}"][m] == "pass").values; b = (g[f"res_{f}"][m] == "pass").values
            n01, n10, p = mcnemar(a, b)
            row[f"mcnemar_{f}"] = {"n": int(m.sum()), "zs_acc": float(a.mean()), "gr_acc": float(b.mean()), "only_grounded_correct": n01, "only_zeroshot_correct": n10, "p": p}
        audit.append(row)
        print(src, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("mcnemar")})
        for k, v in row.items():
            if k.startswith("mcnemar"):
                print("   ", k, v)
    OUT["audit"] = audit

    # ---------------- (d) E1 per-split & fusion gains ----------------
    e1 = pd.read_csv(RES / "e1_recognition.csv"); m = e1[~e1.stream.str.contains("/")]
    OUT["e1"] = m.groupby(["dataset", "stream"]).agg(top1=("top1", "mean"), sd=("top1", "std"), mca=("mean_class_acc", "mean")).reset_index().round(4).to_dict("records")
    OUT["e1_split1"] = m[m.split == 1][["dataset", "stream", "top1", "n_test"]].round(4).to_dict("records")
    ps = RES / "paper_stats.json"
    if ps.exists():  # keep the re-scored audit written by 31_vlm_rescore.py
        with open(ps, encoding="utf-8") as fh:
            prev = json.load(fh)
        for k in ("audit_rescored", "audit_summary_rescored"):
            if k in prev and k not in OUT:
                OUT[k] = prev[k]
    save_json(OUT, ps)
    print("saved paper_stats.json")


if __name__ == "__main__":
    main()
