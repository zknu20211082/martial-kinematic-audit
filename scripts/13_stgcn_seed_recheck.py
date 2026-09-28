"""Re-check the ST-GCN Taichi LOSO regression with different seeds and longer training (stability of the negative result).

Paper: Table 2 (ST-GCN row "3 random seeds": seed 0 from 10, seeds 1 and 2 from this script) and Sec. 2.1.
Output: <results_dir>/e2_stgcn_recheck.csv (rewritten after every run). Needs a CUDA GPU."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import time, warnings
import numpy as np, pandas as pd
from mkaudit.config import DD, RES, ensure_dirs, metadata_path
from mkaudit.skill import loso_stgcn, report_reg

warnings.filterwarnings("ignore")


def main():
    ensure_dirs(RES)
    part = pd.read_csv(metadata_path("UMONS-TAICHI", "participants.csv")).set_index("id")
    out = []
    for sensor in ("qualisys", "kinect"):
        df = pd.read_csv(DD / f"taichi_{sensor}.csv"); df = df[df.pid.isin(part.index)].reset_index(drop=True)
        y = part.loc[df.pid, "skill_mean"].values.astype(float); groups = df.pid.values
        z = np.load(DD / f"taichi_{sensor}_sequences.npz"); key2i = {k: i for i, k in enumerate(z["keys"])}
        X = z["X"][[key2i[f] for f in df.file]]
        for seed, epochs in ((1, 40), (2, 40)):
            t0 = time.time()
            pred = loso_stgcn(X, y, groups, "reg", n_folds=None, epochs=epochs, seed=seed)
            out.append(report_reg(f"Taichi/{sensor}/ST-GCN/LOSO seed{seed} ep{epochs}", y, pred, groups, {"sensor": sensor, "seed": seed, "epochs": epochs, "train_time_s": time.time() - t0}))
            pd.DataFrame(out).to_csv(RES / "e2_stgcn_recheck.csv", index=False)


if __name__ == "__main__":
    main()
