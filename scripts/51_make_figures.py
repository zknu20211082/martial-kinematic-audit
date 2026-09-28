"""Composite paper figures (vector PDF + 300-dpi PNG): Fig.1 framework, Fig.2 knowledge mining & skill,
Fig.3 recognition & robustness, Fig.4 video-LLM kinematic audit.

Paper: Figs. 1-4 (labels are in Chinese, as in the manuscript). Inputs: <results_dir> (outputs of 11, 20, 21, 31, 50),
<work_dir>/taichi_*.csv and pred_taichi_*_ridge_raw.npy (Fig. 2), the synthetic videos (Fig. 4a).
Side effect: writes <results_dir>/paper_group_ablation_raw.json (Fig. 2c numbers, read by 52_paper_numbers.py).
Fonts: Arial and SimHei must be available to matplotlib. On Windows they are taken from the system font folder;
elsewhere pass the .ttf files in MKA_FONT_FILES (separated by the OS path separator).

Usage: python scripts/51_make_figures.py [1 2 3 4] [--out-dir DIR] [--synth-dir DIR]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import av
from mkaudit.config import DD, RES, CACHE, FIG, SYNTH_BENCHMARK_DIR, ensure_dirs, metadata_path
from mkaudit.skill import feat_cols

PF = FIG                            # output folder (set in main)
SYNTH_DIR = CACHE / "synth_videos"  # rendered synthetic videos (set in main)
STATS = None                        # results/paper_stats.json (loaded in main)


def setup_fonts():
    """Register Arial + SimHei (Chinese labels) with matplotlib: files listed in MKA_FONT_FILES, and on Windows the
    system copies."""
    files = [p for p in os.environ.get("MKA_FONT_FILES", "").split(os.pathsep) if p.strip()]
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        files += [os.path.join(windir, "Fonts", n) for n in ("simhei.ttf", "arial.ttf", "arialbd.ttf")]
    for f in files:
        if os.path.exists(f):
            fm.fontManager.addfont(f)


setup_fonts()
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": ["Arial", "SimHei"], "axes.unicode_minus": False,
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": "#8f8e88", "axes.linewidth": 0.6, "axes.labelcolor": "#2b2b2b", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "legend.frameon": False,
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
ORD = {"Novice": "#86b6ef", "Intermediate": "#3987e5", "Advanced": "#1c5cab", "Expert": "#0d366b"}
CAT_ZH = {"Novice": "新手", "Intermediate": "中级", "Advanced": "高级", "Expert": "专家"}
INK, MUTED = "#1f1f1d", "#6b6a64"


def panel(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name):
    fig.savefig(PF / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PF / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


# =====================================================================================
# Figure 1: framework
# =====================================================================================
def fig1():
    fig = plt.figure(figsize=(6.7, 3.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    LAY = {"data": ("#eef4fc", "#2a78d6"), "know": ("#fdf1ea", "#eb6834"), "task": ("#eaf7f1", "#128a60")}

    def box(x, y, w, h, title, body, lay, fs=6.5):
        fc, ec = LAY[lay]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=fs + 0.7, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top", fontsize=fs, color="#333330", linespacing=1.4)

    def arrow(p, q, label=None, rad=0.0, lx=0.0, ly=0.0, color="#55544f", ha="center"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))
        if label:
            ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=5.9, color=color if color != "#55544f" else MUTED, ha=ha, va="center")

    for x, w, t, lay in ((1, 26, "多源数据层", "data"), (36, 26, "运动学知识层", "know"), (71, 28, "可信理解任务层", "task")):
        ax.text(x + w / 2, 54.3, t, ha="center", va="center", fontsize=8.2, fontweight="bold", color=LAY[lay][1])
    # data layer
    box(1, 38, 26, 13.5, "光学动捕：极真空手道", "37名运动员 · 1411条记录\n4524次单次击打 · 9级至4段\n逆突/前踢/回旋踢/后旋踢", "data")
    box(1, 20, 26, 15, "光学动捕：杨式太极拳", "12名练习者 · 13类动作\n2149段 · 3位教师0~10分评级\n同步Kinect V2骨架1815段", "data")
    box(1, 1, 26, 15.5, "网络武术视频", "HMDB51-MA / UCF101-MA\n14类 · 1760条剪辑\n低分辨率 · 多视角 · 多人", "data")
    # knowledge layer
    box(36, 40, 26, 11.5, "统一骨架与分段", "19关节统一拓扑 · 50 Hz重采样\n末端速度峰值切分单次击打\nVicon/Qualisys/Kinect统一坐标", "know")
    box(36, 24, 26, 13, "运动学知识库 KKB", "54维可解释运动学描述符\n受试者级规则挖掘 · 年龄偏相关\n错误发现率校正 · 带量纲规则", "know")
    box(36, 7, 26, 14, "动捕驱动的合成监督", "8方位投影：36 192条二维骨架\n火柴人视频200条 · 精确真值\n单视角与八视角合成预训练", "know")
    # task layer
    box(71, 40, 28, 11.5, "技能评估与规则解释", "留一受试者交叉验证\n受试者级置换检验\n屈髋屈膝 · 动作舒展 · 速度匀缓", "task")
    box(71, 21, 28, 16, "视频大模型运动学审计", "Qwen2.5-VL → 结构化运动学声明\n声明校验器KCV：KF与幻觉率\n零样本提示与证据提示对比\n逐字段核验：人数/肢体/足高/转体/膝伸展", "task")
    box(71, 1, 28, 15.5, "骨架与外观融合识别", "RTMPose → ST-GCN骨架流\nVideoMAE → 线性探针外观流\n后期融合 · 合成预训练 · 鲁棒性", "task")
    # arrows: data -> knowledge
    arrow((27.3, 44.8), (35.7, 45.8))
    arrow((27.3, 28.5), (35.7, 42.0), rad=0.12)
    arrow((49.0, 39.7), (49.0, 37.3)); arrow((49.0, 23.7), (49.0, 21.3))
    # knowledge -> tasks
    arrow((62.3, 33.5), (70.7, 44.0), label="规则与评分", rad=-0.08, lx=0.0, ly=7.6)
    arrow((62.3, 29.0), (70.7, 29.0), label="运动学证据", ly=1.3, color="#c0501f")
    arrow((62.3, 17.5), (70.7, 24.5), label="精确真值", rad=0.0, lx=-1.9, ly=2.4)
    arrow((62.3, 11.0), (70.7, 11.0), label="多视角预训练", ly=1.3)
    # wild video lane under the knowledge column
    arrow((27.3, 3.2), (70.7, 3.2), label="二维姿态序列 + 外观特征", ly=1.2)
    # recognition pipeline provides pose evidence to the audit
    arrow((85.0, 16.8), (85.0, 20.7), label="姿态证据", lx=1.2, ha="left", color="#c0501f")
    save(fig, "fig1_framework")


# =====================================================================================
# Figure 2: knowledge mining & skill assessment
# =====================================================================================
ZH_DESC = {
    "l_hip_min": "左髋角最小值", "r_hip_min": "右髋角最小值", "speed_mean_over_peak_fastest": "末端速度均峰比", "l_knee_at_peak": "峰值时刻左膝角",
    "r_knee_at_peak": "峰值时刻右膝角", "r_toe_max_rel_shoulder": "右足最高点", "l_toe_max_rel_shoulder": "左足最高点",
    "l_knee_max": "左膝角最大值", "l_elbow_at_peak": "峰值时刻左肘角", "r_knee_range": "右膝角活动范围", "r_hip_range": "右髋角活动范围",
    "l_hip_range": "左髋角活动范围", "kick_height_rel_shoulder": "踢腿高度(相对肩)", "r_knee_min": "右膝角最小值",
    "sym_toe": "双足峰速不对称度", "ldj_fastest": "末端轨迹平滑度LDJ", "hip_shoulder_sep_at_peak": "峰值时刻髋肩分离角",
    "peak_time_frac_r_wrist": "右腕峰速出现时刻",
}
UNIT = {"l_hip_min": "°", "r_hip_min": "°", "l_knee_at_peak": "°", "r_knee_at_peak": "°", "l_knee_max": "°", "l_elbow_at_peak": "°", "r_knee_range": "°",
        "r_hip_range": "°", "l_hip_range": "°", "r_knee_min": "°", "speed_mean_over_peak_fastest": "", "r_toe_max_rel_shoulder": "H", "l_toe_max_rel_shoulder": "H",
        "kick_height_rel_shoulder": "H"}
GROUPS = {
    "速度": lambda c: "speed" in c or "peak_time" in c or "path_len" in c,
    "关节角度": lambda c: any(k in c for k in ("knee", "hip_", "elbow", "sep")),
    "姿态": lambda c: any(k in c for k in ("trunk", "stance", "toe_max", "kick_height", "vert_range", "yaw")),
    "平滑/对称": lambda c: any(k in c for k in ("ldj", "n_speed_peaks", "speed_mean_over", "sym_", "head_mean")),
}


def loso_subject_rho(df, cols, y, groups):
    X = np.nan_to_num(df[cols].values.astype(float)); pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        pred[te] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr]).predict(X[te])
    s = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").mean()
    return spearmanr(s.y, s.p).correlation


def fig2():
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    fig = plt.figure(figsize=(6.7, 5.25))
    outer = GridSpec(2, 1, figure=fig, hspace=0.58, left=0.075, right=0.99, top=0.955, bottom=0.09)
    row1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.62, width_ratios=[1, 1, 1.22])
    row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.75, width_ratios=[1.6, 1.08, 1.0])
    # (a)(b) scatter
    abl = {}
    for i, sensor in enumerate(("qualisys", "kinect")):
        ax = fig.add_subplot(row1[0, i])
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        pred = np.load(DD / f"pred_taichi_{sensor}_ridge_raw.npy"); y = part.loc[df.pid, "skill_mean"].values
        g = pd.DataFrame({"pid": df.pid, "y": y, "p": pred}).groupby("pid").agg(y=("y", "first"), p=("p", "mean"), sd=("p", "std"))
        ax.plot([4.5, 10], [4.5, 10], color="#b9b8b1", lw=0.8, ls=(0, (3, 2)), zorder=1)
        for c, col in ORD.items():
            m = part.loc[g.index, "category"] == c
            ax.errorbar(g.y[m], g.p[m], yerr=g.sd[m], fmt="o", ms=4.2, color=col, ecolor=col, elinewidth=0.7, capsize=0, mec="white", mew=0.5, label=CAT_ZH[c], zorder=3)
        st = STATS[f"taichi_{sensor}"]
        ax.text(0.04, 0.97, f"受试者级 ρ = {st['rho_subject']:.2f}\n置换检验 p = {st['p_permutation']:.3f}\n段级 ρ = {st['rho_segment']:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.3, color=INK, linespacing=1.35)
        ax.set_xlim(4.5, 10); ax.set_ylim(4.5, 10); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([5, 6, 7, 8, 9, 10]); ax.set_yticks([5, 6, 7, 8, 9, 10])
        ax.set_xlabel("三位教师评分均值"); ax.set_ylabel("留一受试者预测评分")
        ax.set_title("太极·Qualisys光学动捕" if sensor == "qualisys" else "太极·Kinect V2深度骨架", fontsize=7.6)
        panel(ax, "(a)" if i == 0 else "(b)", x=-0.3)
        if i == 1:
            ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1, fontsize=6.0, labelspacing=0.25)
        # group ablation on raw descriptors
        cols = feat_cols(df); G = pd.get_dummies(df.gesture, prefix="g").astype(float); dfx = pd.concat([df, G], axis=1); gc = list(G.columns)
        groups = df.pid.values
        abl[sensor] = {"全部描述符": loso_subject_rho(dfx, cols + gc, y, groups)}
        for gname, fn in GROUPS.items():
            abl[sensor][f"仅{gname}"] = loso_subject_rho(dfx, [c for c in cols if fn(c)] + gc, y, groups)
        abl[sensor]["去除关节角度"] = loso_subject_rho(dfx, [c for c in cols if not GROUPS["关节角度"](c)] + gc, y, groups)
    json.dump(abl, open(RES / "paper_group_ablation_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # (c) group ablation
    ax = fig.add_subplot(row1[0, 2])
    labels = ["全部描述符", "仅关节角度", "去除关节角度", "仅姿态", "仅平滑/对称", "仅速度"]
    yy = np.arange(len(labels))[::-1]
    for k, (sensor, col, nm) in enumerate((("qualisys", BLUE, "Qualisys"), ("kinect", ORANGE, "Kinect V2"))):
        vals = [abl[sensor][l] for l in labels]
        ax.barh(yy + (0.19 if k == 0 else -0.19), vals, height=0.36, color=col, label=nm)
        for yv, v in zip(yy + (0.19 if k == 0 else -0.19), vals):
            ax.text(v + (0.02 if v >= 0 else -0.02), yv, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=5.6, color=INK)
    ax.axvline(0, color="#8f8e88", lw=0.6); ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.5); ax.set_xlim(-0.5, 1.15)
    ax.set_xlabel("受试者级 Spearman ρ"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    ax.set_title("描述符组消融（留一受试者）", fontsize=7.6); panel(ax, "(c)", x=-0.52)
    # (d) KKB rules
    ax = fig.add_subplot(row2[0, 0])
    r = pd.read_csv(RES / "kkb_taichi_rules_raw.csv").head(10).iloc[::-1]
    cols_ = [BLUE if v > 0 else RED for v in r.spearman_subject]
    ax.barh(np.arange(len(r)), r.spearman_subject, color=cols_, height=0.62)
    ax.set_yticks(np.arange(len(r))); ax.set_yticklabels([ZH_DESC.get(d, d) for d in r.descriptor], fontsize=6.3)
    for i, row in enumerate(r.itertuples()):
        u = UNIT.get(row.descriptor, "")
        fmt = (lambda v: f"{v:.0f}{u}") if u == "°" else ((lambda v: f"{v:+.2f}H") if u == "H" else (lambda v: f"{v:.2f}"))
        txt = f"高分组 {fmt(row.mean_high)} / 低分组 {fmt(row.mean_low)}"
        x = row.spearman_subject
        ax.text(0.03 if x < 0 else -0.03, i, txt, va="center", ha="left" if x < 0 else "right", fontsize=5.3, color="#3a3a36")
    ax.axvline(0, color="#8f8e88", lw=0.6); ax.set_xlim(-1.3, 1.0); ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0]); ax.set_xlabel("与教师评分的受试者级 Spearman ρ（n = 12）")
    ax.grid(axis="y", visible=False); ax.set_title("运动学知识库：太极专长规则（p < 0.01）", fontsize=7.6); panel(ax, "(d)", x=-0.47)
    # (e) karate predicted grade vs age
    ax = fig.add_subplot(row2[0, 1])
    s = pd.read_csv(RES / "paper_karate_subject_preds.csv")
    sc = ax.scatter(s.age, s.p, c=s.y, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]),
                    s=13, edgecolor="white", linewidth=0.4, zorder=3, vmin=1, vmax=13)
    cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.03); cb.set_label("真实等级序数（9级=1，4段=13）", fontsize=5.8); cb.ax.tick_params(labelsize=5.8); cb.outline.set_linewidth(0.4)
    ks = STATS["karate"]
    ax.text(0.97, 0.04, f"ρ(预测等级, 年龄) = {ks['rho_pred_age']:.2f}\nρ(真实等级, 年龄) = {ks['rho_grade_age']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color=INK, linespacing=1.35)
    ax.set_xlabel("年龄 / 岁"); ax.set_ylabel("留一受试者预测等级序数"); ax.set_title("空手道：等级预测与年龄共线", fontsize=7.6); panel(ax, "(e)", x=-0.36)
    # (f) subset correlations
    ax = fig.add_subplot(row2[0, 2])
    names = ["全体\nn=37", "偏相关\n控制年龄", "儿童\nn=23", "成人\nn=14"]
    vals = [ks["rho_subject"], ks["partial_rho_ctrl_age"], ks["rho_children"], ks["rho_adults"]]
    ps = [ks["p"], ks["partial_p"], ks["p_children"], ks["p_adults"]]
    cols_ = [BLUE, "#6da7ec", "#b7d3f6", "#b7d3f6"]
    ax.bar(np.arange(4), vals, color=cols_, width=0.62)
    for i, (v, p) in enumerate(zip(vals, ps)):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(i, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}\n{star}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.9, color=INK, linespacing=1.1)
    ax.axhline(0, color="#8f8e88", lw=0.6); ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=5.9); ax.set_ylim(-0.3, 1.0)
    ax.set_ylabel("预测与真实等级的Spearman ρ"); ax.grid(axis="x", visible=False); ax.set_title("空手道：控制年龄后的等级信号", fontsize=7.6); panel(ax, "(f)", x=-0.42)
    save(fig, "fig2_skill_knowledge")


# =====================================================================================
# Figure 3: recognition, complementarity, robustness, viewpoint
# =====================================================================================
HMDB_ZH = {"draw_sword": "拔剑", "fencing": "击剑", "hit": "击打", "kick": "踢击", "punch": "拳击", "sword": "剑术对练", "sword_exercise": "剑术练习"}


def fig3():
    fig = plt.figure(figsize=(6.7, 5.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.66, wspace=0.34, left=0.1, right=0.985, top=0.955, bottom=0.14)
    cm = pd.read_csv(RES / "e1_confusion_HMDB51-MA.csv", index_col=0)
    classes = list(cm.index); zh = [HMDB_ZH[c] for c in classes]
    cmn = cm.values / cm.values.sum(1, keepdims=True) * 100
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]), vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(zh, rotation=40, ha="right"); ax.set_yticks(range(len(classes))); ax.set_yticklabels(zh)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别"); ax.grid(False); ax.spines[:].set_visible(False)
    ax.set_title("HMDB51-MA 融合模型混淆矩阵 (%，划分1)", fontsize=7.8); panel(ax, "(a)", x=-0.3)
    # (b) per-class recall
    ax = fig.add_subplot(gs[0, 1])
    rec = {}
    for nm, f in (("骨架流 ST-GCN", "e1_confusion_HMDB51-MA_skeleton.csv"), ("外观流 VideoMAE", "e1_confusion_HMDB51-MA_videomae.csv"), ("后期融合", "e1_confusion_HMDB51-MA.csv")):
        c = pd.read_csv(RES / f, index_col=0).loc[classes, classes].values
        rec[nm] = np.diag(c) / c.sum(1)
    order = np.argsort(rec["骨架流 ST-GCN"] - rec["外观流 VideoMAE"])[::-1]
    x = np.arange(len(classes)); w = 0.26
    for k, (nm, col) in enumerate(zip(rec, (ORANGE, BLUE, AQUA))):
        ax.bar(x + (k - 1) * w, rec[nm][order], width=w - 0.02, color=col, label=nm)
    ax.set_xticks(x); ax.set_xticklabels([zh[i] for i in order], rotation=40, ha="right"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("类别召回率")
    ax.legend(loc="upper center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handlelength=1.2); ax.grid(axis="x", visible=False)
    ax.set_title("骨架流与外观流的类别互补性 (HMDB51-MA)", fontsize=7.8); panel(ax, "(b)", x=-0.2)
    # (c) perturbations
    ax = fig.add_subplot(gs[1, 0])
    e1 = pd.read_csv(RES / "e1_recognition.csv")
    base = e1[e1.stream == "skeleton-STGCN"].groupby("dataset").top1.mean()
    pe = e1[e1.stream.str.contains("/")].copy(); pe["k"] = pe.stream.str.split("/").str[1]
    pa = pe.groupby(["dataset", "k"]).top1.mean().unstack(0)
    order_k = ["flip", "jitter03", "jitter06", "crop75", "crop50", "second", "occl20", "occl40", "occl60"]
    names = ["原始", "水平翻转", "抖动3%", "抖动6%", "保留75%时长", "保留50%时长", "去除第二人", "遮挡20%", "遮挡40%", "遮挡60%"]
    for k, (ds, col) in enumerate((("HMDB51-MA", BLUE), ("UCF101-MA", ORANGE))):
        vals = [base[ds]] + [pa.loc[o, ds] for o in order_k]
        ax.plot(range(len(vals)), vals, marker="o", ms=3.4, lw=1.4, color=col, label=ds)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_ylim(0, 1.0); ax.set_ylabel("Top-1准确率（骨架流）")
    ax.axvspan(6.5, 9.5, color="#fbeee6", zorder=0); ax.text(8, 0.93, "关节点遮挡", ha="center", fontsize=6.4, color="#b4501f")
    ax.legend(loc="lower left", fontsize=6.5); ax.set_title("测试时扰动下的骨架流鲁棒性（3组划分均值）", fontsize=7.8); panel(ax, "(c)", x=-0.2)
    # (d) viewpoint
    ax = fig.add_subplot(gs[1, 1])
    v = pd.read_csv(RES / "e4_viewpoint.csv"); azs = [0, 45, 90, 135, 180, 225, 270, 315]
    mixed = v[v.protocol.str.startswith("all-views")].iloc[0]
    ax.plot(azs, [mixed[f"acc_az{a}"] for a in azs], marker="s", ms=3.4, lw=1.6, color=INK, label="八视角混合训练")
    for col, a0 in ((BLUE, 0), (AQUA, 45), (ORANGE, 90)):
        rr = v[v.protocol == f"train-single-view-{a0}/test-all-views"].iloc[0]
        ax.plot(azs, [rr[f"acc_az{a}"] for a in azs], marker="o", ms=3.2, lw=1.3, color=col, label=f"仅{a0}°单视角训练")
        ax.plot([a0], [rr[f"acc_az{a0}"]], marker="o", ms=6.5, mfc="none", mec=col, mew=1.0)
    ax.axhline(0.2, color="#b9b8b1", lw=0.7, ls=(0, (3, 2))); ax.text(157, 0.188, "随机水平（0.2）", fontsize=6, color=MUTED, ha="center", va="top")
    ax.set_xticks(azs); ax.set_xticklabels([f"{a}°" for a in azs]); ax.set_ylim(0, 1.0); ax.set_xlabel("测试相机方位角"); ax.set_ylabel("Top-1准确率（5类空手道技术）")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="upper center", ncol=2, fontsize=6.0, bbox_to_anchor=(0.5, -0.2), columnspacing=0.8)
    ax.set_title("动捕多视角投影监督与视角泛化（受试者不重叠）", fontsize=7.8); panel(ax, "(d)", x=-0.2)
    save(fig, "fig3_recognition_robustness")


# =====================================================================================
# Figure 4: video-LLM kinematic audit
# =====================================================================================
K_ZH = {"Gyaku-Zuki": "逆突", "Mae-Geri": "前踢", "Mawashi-Geri-gedan": "下段回旋踢", "Mawashi-Geri-jodan": "上段回旋踢", "Ushiro-Mawashi-Geri": "后旋踢"}
LIMB_ZH = {"right_arm": "右臂", "left_arm": "左臂", "right_leg": "右腿", "left_leg": "左腿"}
FOOT_ZH = {"no_kick": "无踢击", "below_hip": "低于髋", "hip_to_shoulder": "髋至肩", "above_shoulder": "高于肩"}


def frames_of(video, n=4):
    with av.open(str(SYNTH_DIR / video)) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = np.linspace(0, len(fr) - 1, n + 2).round().astype(int)[1:-1]
    return [fr[i][20:220, 50:270] for i in idx]


def fig4():
    d = pd.read_csv(RES / "e3_vlm_synth_rescored.csv", keep_default_na=False, na_values=[""])
    S = pd.read_csv(RES / "e3_summary_rescored.csv")
    fig = plt.figure(figsize=(6.7, 5.55))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.78, 1.0], hspace=0.3, wspace=0.55, left=0.07, right=0.985, top=0.975, bottom=0.085)
    # (a) qualitative
    sub = GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[0, :], wspace=0.04, hspace=0.18, width_ratios=[1, 1, 1, 1, 0.12, 3.4])
    ex = [("synth_0000_karate.mp4", "示例1：逆突（出拳）"), ("synth_0030_karate.mp4", "示例2：前踢")]
    for rix, (vid, title) in enumerate(ex):
        fr = frames_of(vid)
        for j in range(4):
            a = fig.add_subplot(sub[rix, j]); a.imshow(fr[j]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color("#d4d3cc"); sp.set_linewidth(0.5)
            if j == 0:
                a.set_ylabel(title, fontsize=6.8, labelpad=3)
                if rix == 0:
                    panel(a, "(a)", x=-0.55, y=1.06)
        t = fig.add_subplot(sub[rix, 5]); t.axis("off")
        rows = d[d["clip"] == vid].set_index("condition")
        tr = rows.iloc[0]
        lines = [("真值", f"{K_ZH[tr.label]} | 打击肢体 {LIMB_ZH[tr.truth_striking_limb]} | 脚最高点 {FOOT_ZH[tr.truth_peak_foot_height]}", INK)]
        for cond, nm, col in (("zeroshot", "零样本", ORANGE), ("grounded", "证据提示", BLUE)):
            p = rows.loc[cond]
            ok = [p.res_technique == "pass", p.res_striking_limb == "pass", p.res_peak_foot_height == "pass"]
            mk = lambda b: "√" if b else "×"
            lines.append((nm, f"{K_ZH.get(p.pred_technique, p.pred_technique)}{mk(ok[0])} | 打击肢体 {LIMB_ZH.get(p.pred_striking_limb, p.pred_striking_limb)}{mk(ok[1])} | "
                              f"脚最高点 {FOOT_ZH.get(p.pred_peak_foot_height, p.pred_peak_foot_height)}{mk(ok[2])}", col))
        expl = {"zeroshot": "模型解释：执行后旋踢，右腿伸展旋转并高过肩部（与画面及所选类别均矛盾）" if rix == 0 else "模型解释：与示例1逐字相同（120条中118条解释相同）",
                "grounded": "模型解释：右臂向前伸出完成冲拳，判为逆突" if rix == 0 else "模型解释：高位回旋踢，身体明显旋转（高度证据被用于错误的类别推断）"}
        yy = 0.97
        for nm, txt, col in lines:
            t.text(0.0, yy, f"{nm}", fontsize=6.7, fontweight="bold", color=col, va="top", transform=t.transAxes)
            t.text(0.155, yy, txt, fontsize=6.5, color=INK, va="top", transform=t.transAxes)
            yy -= 0.2
            if nm != "真值":
                t.text(0.155, yy + 0.02, expl["zeroshot" if nm == "零样本" else "grounded"], fontsize=5.9, color=MUTED, va="top", transform=t.transAxes)
                yy -= 0.17
    # (b) synthetic field-wise dumbbell
    fields = [("technique", "技术类别"), ("striking_limb", "打击肢体(分左右)"), ("striking_limb_type", "肢体类型(臂/腿)"), ("peak_foot_height", "脚最高点"),
              ("body_turn", "转体≥90°"), ("knee_extended_at_peak", "峰值膝伸展")]
    ax = fig.add_subplot(gs[1, 0])
    yy = np.arange(len(fields))[::-1]
    for k, (src, off, mk) in enumerate((("synthetic-karate", 0.16, "o"), ("synthetic-taichi", -0.16, "D"))):
        z = S[(S.set == src) & (S.condition == "zeroshot")].iloc[0]; g = S[(S.set == src) & (S.condition == "grounded")].iloc[0]
        for yv, (f, _) in zip(yy + off, fields):
            a, b = z[f"{f}_acc"], g[f"{f}_acc"]
            ax.plot([a, b], [yv, yv], color="#c9c8c1", lw=1.1, zorder=1)
            ax.scatter([a], [yv], s=14, marker=mk, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.3)
            ax.scatter([b], [yv], s=14, marker=mk, color=BLUE, zorder=3, edgecolor="white", linewidth=0.3)
    ax.axhspan(-0.5, 1.5, color="#f4f3ee", zorder=0); ax.text(0.02, -0.38, "证据未覆盖字段", fontsize=5.9, color=MUTED, ha="left", va="center")
    ax.set_yticks(yy); ax.set_yticklabels([n for _, n in fields], fontsize=6.6); ax.set_xlim(0, 1.05); ax.set_xlabel("字段准确率")
    ax.scatter([], [], marker="o", color="#8f8e88", s=12, label="空手道合成(120)"); ax.scatter([], [], marker="D", color="#8f8e88", s=12, label="太极合成(80)")
    ax.scatter([], [], marker="s", color=ORANGE, s=12, label="零样本"); ax.scatter([], [], marker="s", color=BLUE, s=12, label="证据提示")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6, columnspacing=0.8, handletextpad=0.2)
    ax.grid(axis="y", visible=False); ax.set_title("合成视频（精确真值）", fontsize=7.8); panel(ax, "(b)", x=-0.62)
    # (c) wild dumbbell
    ax = fig.add_subplot(gs[1, 1])
    wf = [("technique", "技术类别"), ("n_people", "画面人数"), ("striking_limb", "打击肢体(分左右)"), ("striking_limb_type", "肢体类型(臂/腿)"), ("peak_foot_height", "脚最高点")]
    yy = np.arange(len(wf))[::-1]
    for k, (src, off, mk) in enumerate((("HMDB51-MA", 0.16, "o"), ("UCF101-MA", -0.16, "D"))):
        z = S[(S.set == src) & (S.condition == "zeroshot")].iloc[0]; g = S[(S.set == src) & (S.condition == "grounded")].iloc[0]
        for yv, (f, _) in zip(yy + off, wf):
            a, b = z[f"{f}_acc"], g[f"{f}_acc"]
            ax.plot([a, b], [yv, yv], color="#c9c8c1", lw=1.1, zorder=1)
            ax.scatter([a], [yv], s=14, marker=mk, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.3)
            ax.scatter([b], [yv], s=14, marker=mk, color=BLUE, zorder=3, edgecolor="white", linewidth=0.3)
    ax.set_yticks(yy); ax.set_yticklabels([n for _, n in wf], fontsize=6.6); ax.set_xlim(0, 1.05); ax.set_xlabel("字段准确率（二维姿态伪真值）")
    ax.scatter([], [], marker="o", color="#8f8e88", s=12, label="HMDB51-MA(202)"); ax.scatter([], [], marker="D", color="#8f8e88", s=12, label="UCF101-MA(255)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1, fontsize=6, handletextpad=0.2)
    ax.grid(axis="y", visible=False); ax.set_title("网络视频", fontsize=7.8); panel(ax, "(c)", x=-0.62)
    # (d) technique distribution (mode collapse)
    ax = fig.add_subplot(gs[1, 2])
    kd = d[d.source == "synthetic-karate"]
    labs = ["Gyaku-Zuki", "Mae-Geri", "Mawashi-Geri-gedan", "Mawashi-Geri-jodan", "Ushiro-Mawashi-Geri"]
    dist = {"真实分布": kd[kd.condition == "zeroshot"].label.value_counts(normalize=True),
            "零样本": kd[kd.condition == "zeroshot"].pred_technique.value_counts(normalize=True),
            "证据提示": kd[kd.condition == "grounded"].pred_technique.value_counts(normalize=True)}
    yy = np.arange(3)[::-1]
    for yv, (nm, vc) in zip(yy, dist.items()):
        left = 0
        for c, col in zip(labs, (BLUE, ORANGE, AQUA, YELLOW, VIOLET)):
            w_ = float(vc.get(c, 0.0))
            if w_ > 0:
                ax.barh(yv, w_, left=left, color=col, height=0.55, edgecolor="white", linewidth=0.6)
                if w_ >= 0.12:
                    ax.text(left + w_ / 2, yv, f"{w_:.0%}", ha="center", va="center", fontsize=5.8, color="white")
            left += w_
    ax.set_yticks(yy); ax.set_yticklabels(list(dist.keys()), fontsize=6.6); ax.set_xlim(0, 1); ax.set_xlabel("预测类别占比（空手道合成视频）")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, edgecolor="none", label=K_ZH[c]) for c, col in zip(labs, (BLUE, ORANGE, AQUA, YELLOW, VIOLET))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, fontsize=6, columnspacing=0.6, handletextpad=0.3, handlelength=1.0)
    ax.grid(False); ax.set_title("零样本类别坍缩", fontsize=7.8); panel(ax, "(d)", x=-0.38)
    save(fig, "fig4_llm_audit")


def main(which, out_dir, synth_dir):
    global PF, SYNTH_DIR, STATS
    PF = pathlib.Path(out_dir); ensure_dirs(PF)
    SYNTH_DIR = pathlib.Path(synth_dir)
    with open(RES / "paper_stats.json", encoding="utf-8") as fh:
        STATS = json.load(fh)
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[w]()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="*", help="figures to draw, any of 1 2 3 4 (default: all)")
    ap.add_argument("--out-dir", default=str(FIG), help="output folder (default: figures_dir)")
    ap.add_argument("--synth-dir", default=None, help="synthetic videos for Fig. 4(a) "
                    "(default: <cache_dir>/synth_videos if it exists, else data/synthetic_benchmark/videos)")
    a = ap.parse_args()
    if any(f not in ("1", "2", "3", "4") for f in a.figs):
        ap.error("figures must be among 1 2 3 4")
    synth = a.synth_dir or (CACHE / "synth_videos" if (CACHE / "synth_videos").is_dir() else SYNTH_BENCHMARK_DIR / "videos")
    main(a.figs or ["1", "2", "3", "4"], a.out_dir, synth)
