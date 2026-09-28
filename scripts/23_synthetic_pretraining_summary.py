"""Summarise E7 (synthetic-skeleton pre-training -> fine-tuning on in-the-wild sets): mean/sd over splits, paired deltas, sign test.

Paper: Table 3(b) and Sec. 2.4. Input: <results_dir>/e7_synth_pretrain.csv and e7_synth_pretrain_cross.csv (from 22).
Output: <results_dir>/e7_synth_pretrain.json (replaces the shorter summary written at the end of 22). CPU only."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import numpy as np, pandas as pd
from scipy.stats import binomtest
from mkaudit.config import RES
from mkaudit.utils import save_json


def main():
    df = pd.read_csv(RES / "e7_synth_pretrain.csv"); cross = pd.read_csv(RES / "e7_synth_pretrain_cross.csv")
    out = {"mean": [], "paired": []}
    for (ds, frac, stream, init), g in df.groupby(["dataset", "label_frac", "stream", "init"]):
        out["mean"].append({"dataset": ds, "label_frac": frac, "stream": stream, "init": init, "mean": g.top1.mean(), "sd": g.top1.std(ddof=1), "per_split": g.sort_values("split").top1.round(4).tolist()})
    for stream in ("skeleton", "fusion"):
        for frac in (1.0, 0.25):
            for init in ("pretrain-1view", "pretrain-8view"):
                d = []
                for ds in ("HMDB51-MA", "UCF101-MA"):
                    a = df[(df.dataset == ds) & (df.label_frac == frac) & (df.stream == stream) & (df.init == "scratch")].sort_values("split").top1.values
                    b = df[(df.dataset == ds) & (df.label_frac == frac) & (df.stream == stream) & (df.init == init)].sort_values("split").top1.values
                    d += list(b - a)
                d = np.array(d); npos = int((d > 0).sum()); nneg = int((d < 0).sum())
                p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else 1.0
                out["paired"].append({"stream": stream, "label_frac": frac, "init": init, "deltas_pp": (100 * d).round(1).tolist(), "mean_delta_pp": float(100 * d.mean()),
                                      "n_pos": npos, "n_neg": nneg, "sign_p": float(p)})
    out["cross"] = cross.to_dict("records")
    save_json(out, RES / "e7_synth_pretrain.json")
    m = pd.DataFrame(out["mean"]); m["mean"] = (100 * m["mean"]).round(1); m["sd"] = (100 * m["sd"]).round(1)
    pd.set_option("display.width", 200)
    print(m.drop(columns="per_split").to_string(index=False))
    for r in out["paired"]:
        print(r)


if __name__ == "__main__":
    main()
