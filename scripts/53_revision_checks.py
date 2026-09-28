"""Answer the reviewer's verification items (A1-A10) directly from the experiment records: record counts, split
counts, Benjamini-Hochberg FDR of the KKB rules, audit claim counts (pass / fail / unverifiable), parse failures.

Paper: Sec. 1.1 (file and segment counts), Secs. 2.1-2.2 (FDR q-values), Sec. 1.5 and Table 4 notes (verifiable claim
counts, 0 parse failures), Table 3 notes (HMDB51 split-1 counts). Outputs in <results_dir>: revision_checks.json,
kkb_karate_rules_age_controlled_fdr.csv, kkb_taichi_rules_raw_fdr.csv. CPU only."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import json, glob, os
import numpy as np, pandas as pd
from mkaudit.config import DATA, DD, RES, CACHE, ensure_dirs
from mkaudit.stats import bh_fdr as bh
from mkaudit.utils import save_json


def main():
    ensure_dirs(RES)
    OUT = {}
    # ---------------- A1: karate record counts ----------------
    kf = pd.read_csv(DD / "karate_files.csv")
    ks = pd.read_csv(DD / "karate_strikes.csv")
    err = kf[kf["error"].notna()] if "error" in kf else kf.iloc[0:0]
    ok = kf[kf["error"].isna()] if "error" in kf else kf
    seg_files = set(ks["file"])
    ok = ok.assign(has_seg=ok["canon"].isin(seg_files))
    OUT["A1"] = {"files_total": int(len(kf)), "excluded": err[["file", "error"]].to_dict("records"), "valid": int(len(ok)),
                 "rate_counts_valid": ok["rate"].value_counts().to_dict(), "files_with_segments": int(ok["has_seg"].sum()),
                 "files_without_segments": int((~ok["has_seg"]).sum()),
                 "no_seg_by_cond": ok[~ok["has_seg"]].groupby("cond").size().to_dict(),
                 "no_seg_by_tech": ok[~ok["has_seg"]].groupby("tech").size().to_dict(),
                 "segments": int(len(ks)), "segments_by_cond": ks.groupby("cond").size().to_dict()}
    # ---------------- A2: HMDB/UCF split-1 test counts ----------------
    idx = pd.read_csv(DD / "video_index.csv")
    HSPLIT = DATA / "hmdb51" / "test_train_splits" / "testTrainMulti_7030_splits"
    official = {}
    for cls in ["punch", "kick", "hit", "sword", "sword_exercise", "draw_sword", "fencing"]:
        rows = [l.split() for l in open(HSPLIT / f"{cls}_test_split1.txt", encoding="utf-8", errors="ignore") if l.strip()]
        lab = [int(r[1]) for r in rows if len(r) >= 2]
        names = [r[0] for r in rows if len(r) >= 2]
        local = set(idx[(idx.dataset == "HMDB51-MA") & (idx.cls == cls)]["clip"])
        official[cls] = {"listed": len(names), "test_listed": lab.count(2), "train_listed": lab.count(1),
                         "test_listed_missing_locally": sum(1 for n, l in zip(names, lab) if l == 2 and n not in local)}
    pose_ok = {os.path.basename(f)[:-4] for f in glob.glob(str(CACHE / "pose2d" / "*.npz"))}
    vm_ok = {os.path.basename(f)[:-4] for f in glob.glob(str(CACHE / "videomae" / "*.npz"))}
    h1 = idx[(idx.dataset == "HMDB51-MA") & (idx.split1 == 2)]
    u1 = idx[(idx.dataset == "UCF101-MA") & (idx.split1 == 2)]
    OUT["A2"] = {"hmdb_official_split1": official,
                 "hmdb_split1_test_local": int(len(h1)),
                 "hmdb_split1_test_with_pose_and_videomae": int(h1["clip"].isin(pose_ok & vm_ok).sum()),
                 "hmdb_split1_test_missing_cache": h1[~h1["clip"].isin(pose_ok & vm_ok)]["clip"].tolist(),
                 "hmdb_all_missing_cache": idx[(idx.dataset == "HMDB51-MA") & ~idx["clip"].isin(pose_ok & vm_ok)]["clip"].tolist(),
                 "ucf_split1_test_local": int(len(u1)), "ucf_split1_test_with_caches": int(u1["clip"].isin(pose_ok & vm_ok).sum())}
    # ---------------- A3: partial correlations, FDR ----------------
    kr = pd.read_csv(RES / "kkb_karate_rules_age_controlled.csv")
    kr["q_fdr"] = bh(kr["p_partial"].values)
    tr = pd.read_csv(RES / "kkb_taichi_rules_raw.csv").dropna(subset=["p_subject"])
    tr["q_fdr"] = bh(tr["p_subject"].values)
    st = json.load(open(RES / "paper_stats.json", encoding="utf-8"))
    OUT["A3"] = {"overall_partial": {"rho": st["karate"]["partial_rho_ctrl_age"], "p": st["karate"]["partial_p"]},
                 "karate_rules_top": kr.head(6)[["descriptor", "partial_rho_grade_ctrl_age", "p_partial", "q_fdr"]].round(5).to_dict("records"),
                 "karate_n_tests": int(len(kr)), "karate_min_q": float(kr.q_fdr.min()),
                 "taichi_rules_top10": tr.head(10)[["descriptor", "spearman_subject", "p_subject", "q_fdr"]].round(5).to_dict("records"),
                 "taichi_n_tests": int(len(tr)), "taichi_n_q_lt_0.05": int((tr.q_fdr < 0.05).sum()), "taichi_n_p_lt_0.01": int((tr.p_subject < 0.01).sum()),
                 "taichi_top10_max_q": float(tr.head(10).q_fdr.max())}
    kr.to_csv(RES / "kkb_karate_rules_age_controlled_fdr.csv", index=False); tr.to_csv(RES / "kkb_taichi_rules_raw_fdr.csv", index=False)
    # ---------------- A4/A5/A9/A10: VLM audit counts ----------------
    syn = pd.read_csv(RES / "e3_vlm_synth_rescored.csv", keep_default_na=False, na_values=[""])
    wild = pd.read_csv(RES / "e3_vlm_wild_rescored.csv", keep_default_na=False, na_values=[""])
    k = syn[(syn.source == "synthetic-karate")]
    OUT["A4"] = {cond: pd.crosstab(g.truth_knee_extended_at_peak, g.pred_knee_extended_at_peak.astype(str)).to_dict() for cond, g in k.groupby("condition")}
    OUT["A4_n_arm_truth"] = int((k[k.condition == "zeroshot"].truth_striking_limb.isin(["left_arm", "right_arm"])).sum())
    OUT["A4_n_punch_videos"] = int((k[k.condition == "zeroshot"].label == "Gyaku-Zuki").sum())
    OUT["A4_arm_truth_by_label"] = k[(k.condition == "zeroshot") & k.truth_striking_limb.isin(["left_arm", "right_arm"])].label.value_counts().to_dict()
    FIELDS = ["n_people", "striking_limb", "peak_foot_height", "body_turn", "knee_extended_at_peak"]
    cnt = {}
    for (src, cond), g in pd.concat([syn, wild]).groupby(["source", "condition"]):
        d = {"n": int(len(g))}
        for f in FIELDS:
            r = g[f"res_{f}"]
            d[f] = {"pass": int((r == "pass").sum()), "fail": int((r == "fail").sum()), "unverifiable": int((r == "unverifiable").sum())}
        tp = sum(d[f]["pass"] for f in FIELDS); tf = sum(d[f]["fail"] for f in FIELDS)
        d["KF_micro"] = tp / (tp + tf); d["HR"] = tf / (len(g) * len(FIELDS))
        # macro variant for comparison
        per = []
        for _, row in g.iterrows():
            ps = sum(row[f"res_{f}"] == "pass" for f in FIELDS); fs = sum(row[f"res_{f}"] == "fail" for f in FIELDS)
            if ps + fs:
                per.append(ps / (ps + fs))
        d["KF_macro"] = float(np.mean(per))
        tech = g["pred_technique"].astype(str)
        d["parse_fail_technique_empty"] = int(g["pred_technique"].isna().sum() + (tech.str.strip() == "").sum())
        d["raw_without_json"] = int((~g["raw"].astype(str).str.contains(r"\{")).sum())
        cnt[f"{src}|{cond}"] = d
    OUT["A5_counts"] = cnt
    OUT["A9_npeople_truth"] = {f"{a}|{b}": int(v) for (a, b), v in wild[wild.condition == "zeroshot"].groupby("source").truth_n_people.value_counts().items()}
    # ---------------- A7/A8 environment & model ids ----------------
    import torch, transformers
    OUT["A8"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    OUT["A7"] = "MCG-NJU/videomae-base-finetuned-kinetics (VideoMAE V1, ViT-B, Kinetics-400 fine-tuned)"
    # parameter ratio
    e6 = pd.read_csv(RES / "e6_efficiency.csv")
    OUT["param_ratio"] = {"fusion_incl_pose_M": 2.04 + 86.54 + 39.0, "fusion_excl_pose_M": 2.04 + 86.54, "qwen_M": 8290.0}
    OUT["param_ratio"]["ratio_incl_pose"] = OUT["param_ratio"]["fusion_incl_pose_M"] / 8290.0
    OUT["param_ratio"]["ratio_excl_pose"] = OUT["param_ratio"]["fusion_excl_pose_M"] / 8290.0
    save_json(OUT, RES / "revision_checks.json")
    print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
