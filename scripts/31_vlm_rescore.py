"""Re-score the stored VLM outputs with the verifier aligned to the prompt specification (the scoring used in the paper):
- knee_extended_at_peak: ground truth is "n/a" when the (true) striking limb is an arm (the prompt asks for "n/a" in that case);
- adds striking_limb_type (arm / leg, side-agnostic) as an additional covered field for side-ambiguous views.

Paper: Table 4, Fig. 4(b)-(d), Secs. 2.5-2.6 (KF, hallucination rate, bootstrap CIs, exact McNemar tests).
Inputs:  <in-dir>/e3_vlm_synth.csv, <in-dir>/e3_vlm_wild.csv (raw outputs written by 30; shipped in results/).
Outputs: <out-dir>/e3_vlm_{synth,wild}_rescored.csv, <out-dir>/e3_summary_rescored.csv and <out-dir>/paper_stats.json
         (keys 'audit_rescored' and 'audit_summary_rescored' are added to an existing paper_stats.json found in
         <out-dir>, else in <in-dir>; otherwise a new file with only these keys is written).
CPU only, a few seconds.

Usage: python scripts/31_vlm_rescore.py [--in-dir DIR] [--out-dir DIR]   (both default to <results_dir>)"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from mkaudit.config import RES
from mkaudit.verifier import RESCORE_FIELDS as FIELDS, KIN, COVERED, HELD, rescore, kf, hallucination_rate, verifiable_rate
from mkaudit.stats import boot_ci, mcnemar_exact as mcnemar
from mkaudit.utils import save_json


def main(in_dir=RES, out_dir=RES):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frames = []
    for name in ("e3_vlm_synth.csv", "e3_vlm_wild.csv"):
        raw = pd.read_csv(in_dir / name, keep_default_na=False, na_values=[""])  # keep literal "n/a" answers as strings
        d = rescore(raw); d.to_csv(out_dir / name.replace(".csv", "_rescored.csv"), index=False); frames.append(d)
    r = pd.concat(frames)
    summ, audit = [], []
    for (src, cond), g in r.groupby(["source", "condition"]):
        row = {"set": src, "condition": cond, "n": len(g)}
        for f in FIELDS:
            res = g[f"res_{f}"]; v = res.isin(["pass", "fail"]).sum()
            row[f"{f}_acc"] = float((res == "pass").sum() / v) if v else np.nan; row[f"{f}_n"] = int(v)
        row["KF"] = float(kf(g, KIN)); row["KF_covered"] = float(kf(g, COVERED)); row["KF_held"] = float(kf(g, HELD))
        row["halluc_rate"] = hallucination_rate(g, KIN); row["verifiable_rate"] = verifiable_rate(g, KIN)
        row["KF_ci"] = boot_ci(g, lambda s: kf(s, KIN), rng); row["tech_ci"] = boot_ci(g, lambda s: (s.res_technique == "pass").mean(), rng)
        vc = g.pred_technique.astype(str).value_counts(normalize=True)
        row["top_pred_label"] = vc.index[0]; row["top_pred_share"] = float(vc.iloc[0]); row["pred_entropy_bits"] = float(-(vc.values * np.log2(vc.values)).sum())
        row["n_classes"] = int(g.label.nunique()); row["mean_gen_s"] = float(g.gen_s.mean())
        summ.append(row)
    S = pd.DataFrame(summ); S.to_csv(out_dir / "e3_summary_rescored.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["set", "condition", "n", "technique_acc", "n_people_acc", "striking_limb_acc", "striking_limb_type_acc", "peak_foot_height_acc", "body_turn_acc",
             "knee_extended_at_peak_acc", "KF", "KF_covered", "KF_held", "halluc_rate", "top_pred_share", "pred_entropy_bits"]].round(3).to_string(index=False))
    for src, sub in r.groupby("source"):
        z = sub[sub.condition == "zeroshot"].set_index("clip"); g = sub[sub.condition == "grounded"].set_index("clip")
        c = z.index.intersection(g.index); z, g = z.loc[c], g.loc[c]
        row = {"set": src, "n_pairs": int(len(c))}
        for f in FIELDS:
            m = z[f"res_{f}"].isin(["pass", "fail"]) & g[f"res_{f}"].isin(["pass", "fail"])
            if m.sum() == 0:
                continue
            a = (z[f"res_{f}"][m] == "pass").values; b = (g[f"res_{f}"][m] == "pass").values
            n01, n10, p = mcnemar(a, b)
            row[f] = {"n": int(m.sum()), "zs": float(a.mean()), "gr": float(b.mean()), "gr_only": n01, "zs_only": n10, "p": p}
        audit.append(row)
        print(src, {k: v for k, v in row.items() if k not in ("set",)})
    # knee answers breakdown for synthetic sets
    syn = r[r.source.str.startswith("synthetic")]
    print(pd.crosstab([syn.source, syn.condition, syn.truth_knee_extended_at_peak], syn.pred_knee_extended_at_peak.astype(str)))
    ps_out = out_dir / "paper_stats.json"
    ps_in = ps_out if ps_out.exists() else in_dir / "paper_stats.json"
    st = {}
    if ps_in.exists():
        with open(ps_in, encoding="utf-8") as fh:
            st = json.load(fh)
    st["audit_rescored"] = audit; st["audit_summary_rescored"] = json.loads(S.to_json(orient="records"))
    save_json(st, ps_out)
    print("updated", ps_out.name, "in", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(RES), help="folder with e3_vlm_synth.csv and e3_vlm_wild.csv (default: results_dir)")
    ap.add_argument("--out-dir", default=str(RES), help="folder for the re-scored files (default: results_dir)")
    a = ap.parse_args()
    main(a.in_dir, a.out_dir)
