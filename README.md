# Kinematic Knowledge Mining and Hallucination Auditing of Multimodal Large Language Models for Martial-Arts Videos on Social Media



[English](#english) | [中文](#中文)

---

## English

### Overview

Motion capture of Kyokushin karate (Vicon, 37 athletes, 4524 single strikes) and Yang-style Taijiquan (Qualisys and
Kinect V2, 12 practitioners, 13 gestures) is mapped to a unified 19-joint skeleton and described by 54 interpretable
kinematic descriptors. Expertise rules are mined at subject level with leave-one-subject-out (LOSO) validation and
permutation tests, and the karate grade labels are tested for an age confound with partial correlations. For
in-the-wild video, a 2D-skeleton stream (RTMPose + ST-GCN) and an appearance stream (VideoMAE linear probe) are fused
and evaluated on the martial-arts subsets of HMDB51 and UCF101 (official splits, cross-dataset transfer, test-time
perturbations). The motion capture is projected into eight camera views, giving 36 192 synthetic 2D skeletons for
viewpoint experiments and pre-training, and 200 rendered stick-figure videos with exact kinematic ground truth. A
kinematic claim verifier (KCV) checks the structured answers of Qwen2.5-VL-7B-Instruct field by field (technique,
number of people, striking limb, peak foot height, body turn, knee extension) under zero-shot prompting and under
prompting with pose-derived kinematic evidence, and reports kinematic faithfulness (KF) and hallucination rate (HR).

### Repository structure

```
martial-kinematic-audit/
├── README.md
├── LICENSE                    MIT (code)
├── CITATION.cff
├── requirements.txt           pinned versions used for the paper
├── environment.yml            conda environment (Python 3.13) installing requirements.txt
├── configs/
│   └── paths.yaml             storage locations, overridable with MKA_* environment variables
├── mkaudit/                   shared library code
│   ├── config.py              path configuration
│   ├── skeleton.py            unified 19-joint skeleton, gap filling, resampling, smoothing
│   ├── descriptors.py         54 kinematic descriptors, sequence normalisation
│   ├── mocap_karate.py        Kyokushin C3D parsing, single-strike segmentation
│   ├── mocap_taichi.py        UMONS-TAICHI Qualisys / Kinect parsing
│   ├── synth.py               multi-view projection, stick-figure rendering, exact ground truth
│   ├── stgcn.py               ST-GCN model, training and prediction
│   ├── recognition.py         2D pose loading, augmentation, test-time perturbations
│   ├── videomae.py            VideoMAE loading (includes the attention-bias fix for transformers 5.x)
│   ├── vlm.py                 Qwen2.5-VL prompt template, frame sampling, JSON parsing
│   ├── verifier.py            kinematic claim verifier, re-scoring, KF and HR
│   ├── skill.py               LOSO skill regression, rule mining
│   ├── stats.py               bootstrap CIs, exact McNemar, Benjamini-Hochberg, age-partial correlation
│   └── cuda_dlls.py, utils.py
├── scripts/                   numbered entry points 00-53, one per pipeline step
├── data/
│   ├── README.md              where to download each dataset and the expected folder layout
│   ├── metadata/              participant and gesture tables used by the pipeline
│   └── synthetic_benchmark/   200 stick-figure videos + exact kinematic ground truth (synth_videos.csv)
├── results/                   result tables behind the paper, including all raw VLM outputs (see results/README.md)
└── docs/
    ├── prompts.md             exact zero-shot and evidence prompts, answer schema, parsing
    └── reproduce.md           step-by-step pipeline, runtimes, step -> table / figure map
```

### Installation

With conda:

```bash
conda env create -f environment.yml
conda activate mkaudit
```

Or with pip in an existing Python 3.13 environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper, including the CUDA 12.6 builds of PyTorch (installed from the
PyTorch wheel index). rtmlib declares a dependency on the CPU package `onnxruntime`; for GPU pose extraction keep only
`onnxruntime-gpu`:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

The scripts can be run from the repository root without installing the package (each script adds the repository
root to `sys.path`).

### Data preparation

Download the datasets and place them under the data root as described in [data/README.md](data/README.md). By default
the pipeline expects them in `./data/raw`; other locations are set in `configs/paths.yaml` or with environment
variables, for example

```bash
export MKA_DATA_ROOT=/path/to/datasets        # PowerShell: $env:MKA_DATA_ROOT = "<path-to-datasets>"
```

`MKA_WORK_DIR`, `MKA_CACHE_DIR`, `MKA_HF_CACHE`, `MKA_RESULTS_DIR` and `MKA_FIGURES_DIR` work the same way.
Hugging Face models are downloaded on first use into `<cache_dir>/hf` (or `MKA_HF_CACHE`).

### Quick start: reproduce the audit table without a GPU

`results/e3_vlm_synth.csv` and `results/e3_vlm_wild.csv` contain the raw answers of Qwen2.5-VL-7B-Instruct for all
1316 generations (400 on the 200 synthetic videos, 916 on the 459 in-the-wild split-1 test videos; zero-shot and
evidence prompts). Re-scoring them with the verifier reproduces the audit table (Table 4) in about 15 seconds on a
CPU. Only numpy, pandas, scipy, scikit-learn and PyYAML are needed for this step.

```bash
python scripts/31_vlm_rescore.py --out-dir outputs/quickstart
```

The script writes `e3_summary_rescored.csv`, the two `*_rescored.csv` files and `paper_stats.json` (McNemar tests
added) to `outputs/quickstart`; they are identical to the files in `results/`. Headline numbers
(`results/e3_summary_rescored.csv`; `zeroshot` = zero-shot prompt, `grounded` = prompt with kinematic evidence):

| Test set | Prompt | Videos | KF (95% CI) | HR | Technique accuracy | Share of most frequent predicted technique |
|---|---|---|---|---|---|---|
| synthetic-karate | zeroshot | 120 | 0.483 (0.438-0.527) | 0.517 | 0.200 | 1.000 |
| synthetic-karate | grounded | 120 | 0.763 (0.742-0.785) | 0.237 | 0.375 | 0.625 |
| synthetic-taichi | zeroshot | 80 | 0.675 (0.623-0.728) | 0.325 | 0.113 | 0.738 |
| synthetic-taichi | grounded | 80 | 0.905 (0.880-0.930) | 0.095 | 0.138 | 0.487 |
| HMDB51-MA | zeroshot | 203 | 0.603 (0.562-0.643) | 0.182 | 0.315 | 0.488 |
| HMDB51-MA | grounded | 202 | 0.955 (0.935-0.972) | 0.021 | 0.282 | 0.460 |
| UCF101-MA | zeroshot | 256 | 0.773 (0.748-0.799) | 0.107 | 0.742 | 0.258 |
| UCF101-MA | grounded | 255 | 0.964 (0.948-0.977) | 0.017 | 0.722 | 0.271 |

KF = passed / (passed + failed) kinematic claims and HR = failed / all kinematic claims, pooled over the five
kinematic fields of all videos of a test set. In-the-wild ground truth is derived from 2D pose, so body turn and knee
extension are unverifiable there, and under evidence prompting the fields covered by the evidence share their source
with the pseudo ground truth (see the paper, Sec. 1.5). The prompts are listed in [docs/prompts.md](docs/prompts.md).

### Full reproduction

[docs/reproduce.md](docs/reproduce.md) lists every step with its inputs, outputs, approximate runtime on one RTX 3090
and the tables and figures it feeds. In short:

| Steps | What | Needs |
|---|---|---|
| 00-02 | video index; karate and Taichi preprocessing, 54 descriptors | datasets, CPU |
| 03-05 | 2D pose (RTMPose), VideoMAE features, multi-view projections and stick-figure videos | GPU for 03 and 04 |
| 10-13 | skill assessment, rule mining, age confound, ablations, ST-GCN seed re-check | CPU (GPU for ST-GCN parts) |
| 20-23 | recognition, viewpoint generalisation, synthetic pre-training | GPU |
| 30-31 | Qwen2.5-VL audit and re-scoring | GPU with 24 GB for 30; CPU for 31 |
| 40 | latency and parameter counts | GPU |
| 50-53 | statistics, figures, numbers quoted in the paper, reviewer checks | CPU |

### Hardware and software

All experiments ran on one NVIDIA RTX 3090 (24 GB) under Windows 11 with Python 3.13, PyTorch 2.12.0 (CUDA 12.6),
Transformers 5.13.1 and onnxruntime-gpu 1.22.0. The 2D pose, VideoMAE, ST-GCN, video-LLM and latency steps need a
CUDA 12.x GPU; all statistics, the re-scoring and the figures run on a CPU. The figures use Arial and SimHei (the
labels are Chinese, as in the manuscript); outside Windows pass the font files in `MKA_FONT_FILES`. GPU training is
not bit-for-bit deterministic, so re-running the neural models can change their numbers slightly; the descriptor
pipeline, the regression and statistics steps and the re-scoring are deterministic.

### Citation

```bibtex
@article{xxx2026kinematic,
  title   = {Kinematic Knowledge Mining and Hallucination Auditing of Multimodal Large Language Models for Martial-Arts Videos on Social Media},
  author  = {×××},
  journal = {×××},
  year    = {2026}
}
```

See also [CITATION.cff](CITATION.cff).

### License

- Code: MIT License (see [LICENSE](LICENSE)).
- Datasets keep their own licenses: Kyokushin karate motion capture CC0; UMONS-TAICHI CC BY-NC-SA 4.0; HMDB51 and
  UCF101 under the terms of their providers. The raw data are not redistributed here.
- `data/synthetic_benchmark/`: clips rendered from the karate data (`*_karate.mp4`) follow CC0; clips rendered from
  UMONS-TAICHI (`*_taichi.mp4`) and their ground-truth rows follow CC BY-NC-SA 4.0 (non-commercial use only,
  share alike, attribution to UMONS-TAICHI).
- `data/metadata/` tables are transcribed from the dataset papers.
- Pre-trained models keep their licenses: Qwen2.5-VL-7B-Instruct (Apache-2.0), MCG-NJU/videomae-base-finetuned-kinetics
  (CC BY-NC 4.0), rtmlib / RTMPose / YOLOX (see https://github.com/Tau-J/rtmlib).

---

## 中文

### 简介

本仓库提供论文《社交媒体武术视频的运动学知识挖掘与多模态大模型幻觉审计》的代码、提示词、合成基准与结果文件。
极真空手道光学动捕（Vicon，37名运动员，4524次单次击打）与杨式太极拳动捕（Qualisys与Kinect V2，12名练习者，13类动作）
被映射到统一的19关节骨架，并计算54个可解释的运动学描述符；在受试者层面用留一受试者交叉验证与置换检验挖掘专长规则，
并用偏相关检验空手道等级标签的年龄混杂。面向网络视频，骨架流（RTMPose + ST-GCN）与外观流（VideoMAE线性探针）
后期融合，在HMDB51与UCF101的武术子集上按官方划分评测，并做跨库迁移与测试时扰动实验。动捕数据按8个相机方位投影，
得到36 192条合成二维骨架（用于视角实验与预训练）以及200条带精确运动学真值的火柴人视频。运动学声明校验器（KCV）
逐字段核验Qwen2.5-VL-7B-Instruct的结构化回答（技术类别、画面人数、打击肢体、足部最高位置、转体、膝伸展），
比较零样本提示与注入姿态运动学证据的提示，报告运动学忠实度（KF）与幻觉率（HR）。

### 目录结构

见上文英文部分的目录树：`mkaudit/` 为共享代码库，`scripts/` 为按编号排列的各步骤入口（00~53），
`data/` 含数据下载说明、元数据表与合成基准，`results/` 为论文所用的全部结果文件（含大模型原始输出），
`docs/` 含提示词说明与复现流程。

### 安装

```bash
conda env create -f environment.yml
conda activate mkaudit
```

或在已有的Python 3.13环境中执行 `python -m pip install -r requirements.txt`。requirements.txt固定了论文使用的版本，
PyTorch为CUDA 12.6版本（从PyTorch官方索引安装）。rtmlib会依赖CPU版 `onnxruntime`，做GPU姿态提取时请只保留
`onnxruntime-gpu`：

```bash
python -m pip uninstall -y onnxruntime
python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

脚本可以直接在仓库根目录运行，无需安装本包。

### 数据准备

按 [data/README.md](data/README.md) 下载各数据集并放到数据根目录下。默认数据根目录为 `./data/raw`，也可以在
`configs/paths.yaml` 中修改，或用环境变量覆盖，例如 `MKA_DATA_ROOT`；`MKA_WORK_DIR`、`MKA_CACHE_DIR`、
`MKA_HF_CACHE`、`MKA_RESULTS_DIR`、`MKA_FIGURES_DIR` 同理。Hugging Face模型在首次使用时下载到 `<cache_dir>/hf`。

### 快速开始：无需GPU复现审计结果表

`results/e3_vlm_synth.csv` 与 `results/e3_vlm_wild.csv` 保存了Qwen2.5-VL-7B-Instruct全部1316次生成的原始回答。
用校验器重新评分即可在CPU上约15秒复现审计结果表（表4），只需要numpy、pandas、scipy、scikit-learn与PyYAML：

```bash
python scripts/31_vlm_rescore.py --out-dir outputs/quickstart
```

输出的 `e3_summary_rescored.csv`、两个 `*_rescored.csv` 与 `paper_stats.json` 与 `results/` 中的文件一致。
主要数值见上文英文部分的表格（`zeroshot` 为零样本提示，`grounded` 为证据提示）。KF为可核验运动学声明中通过的比例，
HR为全部运动学声明中被证伪的比例，均按一个测试集内全部视频的5个运动学字段汇总。提示词全文见
[docs/prompts.md](docs/prompts.md)。

### 完整复现

[docs/reproduce.md](docs/reproduce.md) 给出每一步的输入输出、在单张RTX 3090上的大致耗时，以及它对应的论文图表。
00~02为数据预处理，03~05为姿态、VideoMAE特征与合成数据（03、04需要GPU），10~13为技能评估与知识挖掘，
20~23为识别、视角与合成预训练（需要GPU），30~31为大模型审计与重新评分（30需要24 GB显存的GPU），
40为效率测试，50~53为统计、作图、论文数值汇总与审稿核查（CPU即可）。

### 硬件与软件

全部实验在单张NVIDIA RTX 3090（24 GB）、Windows 11上完成，软件为Python 3.13、PyTorch 2.12.0（CUDA 12.6）、
Transformers 5.13.1与onnxruntime-gpu 1.22.0。姿态提取、VideoMAE、ST-GCN、视频大模型与效率测试需要CUDA 12.x GPU，
统计分析、重新评分与作图可在CPU上运行。作图使用Arial与SimHei字体，非Windows系统请通过 `MKA_FONT_FILES` 指定字体文件。
GPU训练不是逐位确定的，重跑神经网络模型时数值可能略有变化；描述符、回归与统计步骤以及重新评分是确定的。

### 引用

见上文英文部分的BibTeX条目与 [CITATION.cff](CITATION.cff)（作者信息待定，以×××表示）。

### 许可

- 代码采用MIT许可（见 [LICENSE](LICENSE)）。
- 各数据集沿用其原有许可：极真空手道动捕为CC0；UMONS-TAICHI为CC BY-NC-SA 4.0；HMDB51与UCF101遵循其发布方的条款。
  本仓库不再分发原始数据。
- `data/synthetic_benchmark/` 中由空手道数据渲染的视频（`*_karate.mp4`）沿用CC0；由UMONS-TAICHI渲染的视频
  （`*_taichi.mp4`）及其真值行沿用CC BY-NC-SA 4.0，仅限非商业用途，需相同方式共享并注明UMONS-TAICHI来源。
- 预训练模型沿用各自许可：Qwen2.5-VL-7B-Instruct为Apache-2.0，MCG-NJU/videomae-base-finetuned-kinetics为CC BY-NC 4.0，
  rtmlib / RTMPose / YOLOX见 https://github.com/Tau-J/rtmlib 。
