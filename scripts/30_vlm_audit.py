"""E3: kinematic faithfulness of a video LLM (Qwen2.5-VL-7B-Instruct) on wild martial-arts clips and on synthetic
stick-figure clips with exact ground truth. Two prompting conditions: zero-shot vs. evidence-grounded (pose-derived
kinematic evidence injected in the prompt). A rule-based Kinematic Claim Verifier (KCV) scores the JSON fields.

Paper: Sec. 1.5 (generation setup), raw material for Table 4 and Fig. 4. The paper re-scores the stored outputs with
31_vlm_rescore.py; the summaries written here (e3_summary_{wild,synth}.csv) use the scoring applied at generation time.
Outputs in <results_dir>: e3_vlm_wild.csv, e3_vlm_synth.csv (raw model outputs, resumable), e3_summary_{which}.csv.
The paper used two runs: `synth` and `wild`. Needs a CUDA GPU with >= 24 GB (bf16 7B model); wild clips also need
the pose cache of 03.

Usage: python scripts/30_vlm_audit.py [all|wild|synth] [limit] [--model ID_OR_DIR] [--synth-dir DIR] [--synth-csv CSV]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, time
import pandas as pd, torch
from tqdm import tqdm
from mkaudit.config import DD, RES, CACHE, SYNTH_BENCHMARK_DIR, ensure_dirs
from mkaudit.vlm import QWEN_ID, HMDB_CLASSES, UCF_CLASSES, KARATE_CLASSES, FIELDS, read_frames, prompt_text, parse_json, load_qwen
from mkaudit.verifier import POSE, kcv_from_pose, match, summarize


@torch.no_grad()
def run(model, proc, items, out_csv, conditions=("zeroshot", "grounded")):
    rows = []
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); rows = prev.to_dict("records"); done = set(zip(prev["clip"], prev["condition"]))
    for it in tqdm(items, desc=out_csv.stem):
        try:
            frames = read_frames(it["path"])
        except Exception as ex:
            print("decode fail", it["clip"], ex); continue
        for cond in conditions:
            if (it["clip"], cond) in done:
                continue
            ev = it.get("evidence_text") if cond == "grounded" else None
            if cond == "grounded" and not ev:
                continue
            msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_text(it["classes"], ev)}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(model.device)
            t0 = time.time()
            gen = model.generate(**inputs, max_new_tokens=220, do_sample=False)
            out = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            js = parse_json(out)
            row = {"clip": it["clip"], "source": it["source"], "condition": cond, "label": it["label"], "raw": out.replace("\n", " ")[:600], "gen_s": time.time() - t0}
            for f in FIELDS:
                row[f"pred_{f}"] = js.get(f, "")
                row[f"truth_{f}"] = it["truth"].get(f, "unverifiable")
                row[f"res_{f}"] = match(f, js.get(f, ""), it["truth"].get(f, "unverifiable"))
            row["explanation"] = js.get("explanation", "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def main(which="all", limit=None, model_id=QWEN_ID, revision=None, synth_dir=None, synth_csv=None):
    ensure_dirs(RES)
    model, proc = load_qwen(model_id, revision=revision)
    summary = []
    if which in ("all", "wild"):
        idx = pd.read_csv(DD / "video_index.csv")
        test = idx[idx.split1 == 2].reset_index(drop=True)
        items = []
        for r in test.itertuples():
            if not (POSE / (r.clip + ".npz")).exists():
                continue
            ev, evtxt = kcv_from_pose(r.clip)
            truth = {"technique": r.cls, **ev}
            items.append({"clip": r.clip, "path": r.path, "source": r.dataset, "label": r.cls,
                          "classes": HMDB_CLASSES if r.dataset == "HMDB51-MA" else UCF_CLASSES, "truth": truth, "evidence_text": evtxt})
        if limit:
            items = items[:limit]
        df = run(model, proc, items, RES / "e3_vlm_wild.csv")
        for ds in df.source.unique():
            summary += summarize(df[df.source == ds], ds)
    if which in ("all", "synth"):
        sv = pd.read_csv(synth_csv)
        items = []
        for r in sv.itertuples():
            classes = KARATE_CLASSES if r.source == "karate" else sorted(sv[sv.source == "taichi"].label.unique().tolist())
            truth = {"technique": r.label, "n_people": 1, "striking_limb": r.striking_limb, "peak_foot_height": r.peak_foot_height,
                     "body_turn": r.body_turn, "knee_extended_at_peak": r.knee_extended_at_peak}
            evtxt = {"number of tracked people": 1, "fastest-moving limb (performer's own side)": r.striking_limb,
                     "highest foot position": r.peak_foot_height.replace("_", " "), "peak limb speed (body-heights per second)": f"{r.peak_speed:.1f}"}
            items.append({"clip": r.video, "path": str(pathlib.Path(synth_dir) / r.video), "source": f"synthetic-{r.source}", "label": r.label,
                          "classes": classes, "truth": truth, "evidence_text": evtxt})
        if limit:
            items = items[:limit]
        df = run(model, proc, items, RES / "e3_vlm_synth.csv")
        for ds in df.source.unique():
            summary += summarize(df[df.source == ds], ds)
    pd.DataFrame(summary).to_csv(RES / f"e3_summary_{which}.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["all", "wild", "synth"])
    ap.add_argument("limit", nargs="?", type=int, default=None, help="only the first N clips (debugging)")
    ap.add_argument("--model", default=QWEN_ID, help="Hub id or local snapshot directory of Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--revision", default=None, help="Hub revision (the paper used cc594898137f460bfe9f0759e9844b3ce807cfb5)")
    ap.add_argument("--synth-dir", default=None, help="folder with the synthetic videos "
                    "(default: <cache_dir>/synth_videos if it exists, else data/synthetic_benchmark/videos)")
    ap.add_argument("--synth-csv", default=None, help="ground-truth table of the synthetic videos "
                    "(default: <work_dir>/synth_videos.csv if it exists, else data/synthetic_benchmark/synth_videos.csv)")
    a = ap.parse_args()
    synth_dir = a.synth_dir or (CACHE / "synth_videos" if (CACHE / "synth_videos").is_dir() else SYNTH_BENCHMARK_DIR / "videos")
    synth_csv = a.synth_csv or (DD / "synth_videos.csv" if (DD / "synth_videos.csv").exists() else SYNTH_BENCHMARK_DIR / "synth_videos.csv")
    main(a.which, a.limit, a.model, a.revision, synth_dir, synth_csv)
