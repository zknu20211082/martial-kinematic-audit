# Reproducing the paper

## 1. Setup

1. Install the environment (README, Installation) and download the datasets (`data/README.md`).
2. Choose the storage locations in `configs/paths.yaml` or with environment variables. For a full reproduction it is
   convenient to write results to a new folder, so that the shipped `results/` stay untouched for comparison:

   ```bash
   export MKA_DATA_ROOT=/path/to/datasets
   export MKA_RESULTS_DIR=$PWD/outputs/results
   export MKA_FIGURES_DIR=$PWD/outputs/figures
   ```

   PowerShell: `$env:MKA_DATA_ROOT = "<path-to-datasets>"` and so on. All steps must see the same locations, because later
   steps read the outputs of earlier ones.
3. Run the scripts from the repository root, in the order below. Steps 03, 04, 22 and 30 resume where they stopped
   (already cached clips / finished rows are skipped).

## 2. Pipeline

Runtimes are approximate, measured in the original run on one NVIDIA RTX 3090 (24 GB) with a desktop CPU.

| Step | Command | Main inputs | Main outputs | Runtime | GPU |
|---|---|---|---|---|---|
| 00 | `python scripts/00_build_video_index.py` | HMDB51 / UCF101 videos, HMDB51 split files | `work/video_index.csv` | < 1 min | no |
| 01 | `python scripts/01_preprocess_karate.py` | 1411 karate C3D files | `work/karate_strikes.csv`, `karate_files.csv`, `karate_sequences.npz` | ~15 min | no |
| 02 | `python scripts/02_preprocess_taichi.py` | UMONS-TAICHI segmented TSV / Kinect | `work/taichi_{qualisys,kinect}.csv`, `*_sequences.npz` | ~5 min | no |
| 03 | `python scripts/03_extract_pose2d.py` | video index, videos | `cache/pose2d/*.npz` (1757 clips) | ~70 min (126k frames at ~33 ms); shardable: `03_extract_pose2d.py 0 <k> <n>` | yes |
| 04 | `python scripts/04_extract_videomae.py` | video index, videos | `cache/videomae/*.npz` | ~20 min | yes |
| 05 | `python scripts/05_render_synthetic.py` | karate / Taichi sequences | `work/synth_viewpoint_karate.npz` (36 192 skeletons), `cache/synth_videos/*.mp4`, `work/synth_videos.csv` | ~3 min | no |
| 10 | `python scripts/10_skill_assessment.py taichi` then `... karate` | descriptor tables, sequences, metadata | `e2_skill_{taichi,karate}.*`, `kkb_*_rules.csv`, `e2_karate_breakdown.csv` | ~20 min + ~12 min | ST-GCN parts |
| 11 | `python scripts/11_age_confound.py` | descriptor tables, metadata | `e2_karate_confound.csv`, `kkb_karate_rules_age_controlled.csv`, `e2_skill_taichi_raw.csv`, `kkb_taichi_rules_raw.csv`, `e2_taichi_breakdown_*.csv`, `work/pred_taichi_*_ridge_raw.npy` | ~1 min | no |
| 12 | `python scripts/12_skill_ablation.py` | descriptor tables, metadata | `e5_ablation_{taichi,karate}.csv` | a few min | no |
| 13 | `python scripts/13_stgcn_seed_recheck.py` | Taichi sequences | `e2_stgcn_recheck.csv` | ~27 min per sensor | yes |
| 20 | `python scripts/20_recognition.py` | caches of 03 and 04 | `e1_recognition.csv`, `e1_cross_dataset.csv`, `e1_confusion_*.csv` | ~5 min | yes |
| 21 | `python scripts/21_viewpoint.py` | `synth_viewpoint_karate.npz` | `e4_viewpoint.csv` | ~10-15 min | yes |
| 22 | `python scripts/22_synthetic_pretraining.py` | `synth_viewpoint_karate.npz`, caches of 03 and 04 | `e7_synth_pretrain.csv`, `e7_synth_pretrain_cross.csv`, `cache/pretrain/*.pt` | ~20-25 min | yes |
| 23 | `python scripts/23_synthetic_pretraining_summary.py` | outputs of 22 | `e7_synth_pretrain.json` | seconds | no |
| 30 | `python scripts/30_vlm_audit.py synth` then `... wild` | synthetic videos + `synth_videos.csv`; video index, videos, pose cache | `e3_vlm_synth.csv`, `e3_vlm_wild.csv`, `e3_summary_{synth,wild}.csv` | ~40 min + ~70 min | yes (24 GB) |
| 31 | `python scripts/31_vlm_rescore.py` | `e3_vlm_{synth,wild}.csv` | `e3_vlm_*_rescored.csv`, `e3_summary_rescored.csv`, `paper_stats.json` (audit keys) | ~15 s | no |
| 40 | `python scripts/40_efficiency.py` | video index, one clip, `e3_vlm_*.csv` | `e6_efficiency.csv` | ~2 min | yes |
| 50 | `python scripts/50_paper_stats.py` | outputs of 11, 20, 30 | `paper_stats.json`, `paper_karate_subject_preds.csv`, `p_partial` in `kkb_karate_rules_age_controlled.csv` | ~2 min | no |
| 51 | `python scripts/51_make_figures.py` | outputs of 11, 20, 21, 31, 50; synthetic videos | `figures/fig{1..4}_*.pdf/.png`, `paper_group_ablation_raw.json` | ~1 min | no |
| 52 | `python scripts/52_paper_numbers.py` | all of the above | `paper_numbers.json` | ~1 min | no |
| 53 | `python scripts/53_revision_checks.py` | outputs of 01, 11, 31, 40, 50, caches of 03 / 04 | `revision_checks.json`, `kkb_*_fdr.csv` | < 1 min | no |

Paths without a folder are in `<results_dir>`; `work/` and `cache/` stand for `<work_dir>` and `<cache_dir>`.

Order notes:

- 10 was run twice for the paper, once with `taichi` and once with `karate`; the downstream scripts read
  `e2_skill_taichi.*` and `e2_skill_karate.*` (the argument `all` writes `e2_skill_all.*` instead).
- 30 was run as `synth` and `wild`. To audit the shipped benchmark without rebuilding it, skip 01-05: when
  `<cache_dir>/synth_videos` does not exist, 30 and 51 use `data/synthetic_benchmark/`
  (or pass `--synth-dir` / `--synth-csv`). The wild part needs 00 and 03.
- 31 and 50 both write `paper_stats.json`; 31 adds the keys `audit_rescored` / `audit_summary_rescored` and 50 keeps
  them, so either order works.
- 51 must run before 52 (it writes `paper_group_ablation_raw.json`); 50 must run before 51 and 53.

## 3. Which step feeds which table and figure

| Paper item | Result files | Steps |
|---|---|---|
| Table 1 (datasets) | `paper_numbers.json` (`karate_*`, `taichi_*`, `video_index`, `synth*`), `revision_checks.json` (A1, A2) | 00-05, 52, 53 |
| Table 2 (skill assessment) | `e2_skill_taichi_raw.csv`, `e2_skill_taichi.csv`, `e2_stgcn_recheck.csv`, `e2_skill_karate.csv`, `e5_ablation_karate.csv`, `paper_stats.json` | 10, 11, 12, 13, 50 |
| Table 3(a) (recognition) | `e1_recognition.csv`, `e1_cross_dataset.csv`, `e3_summary_rescored.csv` (Qwen2.5-VL row) | 20, 31 |
| Table 3(b) (synthetic pre-training) | `e7_synth_pretrain.csv`, `e7_synth_pretrain_cross.csv`, `e7_synth_pretrain.json` | 22, 23 |
| Table 4 (video-LLM audit) | `e3_summary_rescored.csv`, `revision_checks.json` (verifiable counts) | 30, 31, 53 |
| Fig. 1 (framework) | none (drawn from fixed text) | 51 |
| Fig. 2 (knowledge mining, skill) | `pred_taichi_*_ridge_raw.npy`, `paper_stats.json`, `paper_group_ablation_raw.json`, `kkb_taichi_rules_raw.csv`, `paper_karate_subject_preds.csv` | 11, 50, 51 |
| Fig. 3 (recognition, robustness, viewpoint) | `e1_confusion_HMDB51-MA*.csv`, `e1_recognition.csv`, `e4_viewpoint.csv` | 20, 21, 51 |
| Fig. 4 (audit) | synthetic videos, `e3_vlm_synth_rescored.csv`, `e3_summary_rescored.csv` | 05, 30, 31, 51 |
| Sec. 2.1 text | `kkb_taichi_rules_raw_fdr.csv`, `paper_numbers.json` (`taichi_raw`), `e2_taichi_breakdown_*.csv` | 11, 52, 53 |
| Sec. 2.2 text | `paper_stats.json` (`karate`), `kkb_karate_rules_age_controlled_fdr.csv`, `e5_ablation_karate.csv`, `e2_karate_confound.csv` | 11, 12, 50, 53 |
| Sec. 2.6 text (McNemar, knee answers) | `paper_stats.json` (`audit_rescored`), `revision_checks.json` (A4) | 31, 53 |
| Sec. 2.7 (deployment cost) | `e6_efficiency.csv` | 40 |

## 4. Determinism

- Preprocessing (01, 02, 05), the ridge / logistic regressions, rule mining, the permutation and bootstrap statistics
  (fixed seeds) and the re-scoring (31) are deterministic on a given software stack; 31 reproduces the shipped
  `e3_summary_rescored.csv` exactly, including the bootstrap intervals. LightGBM uses a fixed seed as well.
- ST-GCN training (10, 13, 20-22) uses fixed seeds but cuDNN kernels are not bit-for-bit deterministic, so re-runs can
  differ slightly from the shipped numbers.
- Qwen2.5-VL uses greedy decoding; outputs can still vary slightly across GPUs, drivers and library versions.
  Re-scoring the shipped raw outputs (31) is the exact reference.
