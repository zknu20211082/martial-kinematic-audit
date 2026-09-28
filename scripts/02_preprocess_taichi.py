"""Parse UMONS-TAICHI segmented Qualisys TSV + segmented Kinect -> unified skeletons + descriptors.

Paper: Secs. 1.1-1.2 and Table 1 (2149 Qualisys / 1815 Kinect segments). Outputs in <work_dir>:
taichi_qualisys.csv, taichi_kinect.csv (54 descriptors per segment) and taichi_{qualisys,kinect}_sequences.npz."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import glob, os
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from mkaudit.config import DD, ensure_dirs
from mkaudit.skeleton import interp_nan, to_z_up, resample
from mkaudit.descriptors import descriptors, normalize_sequence
from mkaudit.mocap_taichi import TD, FPS_OUT, Q_JOINTS, K_JOINTS, PAT, parse_name, read_tsv, to_unified


def main():
    ensure_dirs(DD)
    rows, seqs = [], {}
    files = sorted(glob.glob(str(TD / "Segmented_TSV" / "*.tsv")))
    for f in tqdm(files, desc="taichi-qualisys"):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not PAT.match(stem):
            continue
        info = parse_name(stem)
        names, arr, freq = read_tsv(f)
        Jt = to_unified(arr / 1000.0, names, Q_JOINTS)
        Jt = to_z_up(interp_nan(Jt))
        Jr = resample(Jt, freq, FPS_OUT)
        d = descriptors(Jr, FPS_OUT); d.update(info); d["file"] = stem; d["sensor"] = "qualisys"; d["src_fps"] = freq
        rows.append(d); seqs[stem] = normalize_sequence(Jr, 64)
    dfq = pd.DataFrame(rows); dfq.to_csv(DD / "taichi_qualisys.csv", index=False)
    np.savez_compressed(DD / "taichi_qualisys_sequences.npz", keys=np.array(list(seqs.keys())), X=np.stack(list(seqs.values())))
    print("qualisys segments", len(dfq)); print(dfq.groupby("gesture").size().to_dict())

    rows, seqs = [], {}
    kfiles = sorted(glob.glob(str(TD / "Segmented_Kinect" / "*.txt")))
    for f in tqdm(kfiles, desc="taichi-kinect"):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not PAT.match(stem):
            continue
        info = parse_name(stem)
        a = np.loadtxt(f)
        if a.ndim == 1 or len(a) < 5:
            continue
        ts = a[:, 0] / 1000.0; xyz = a[:, 1:76].reshape(len(a), 25, 3) / 1000.0
        # irregular timestamps -> uniform 30 Hz
        fps_k = 30.0
        n = max(2, int(round((ts[-1] - ts[0]) * fps_k)) + 1)
        tu = np.linspace(ts[0], ts[-1], n)
        xyz = interp1d(ts, xyz, axis=0, kind="linear")(tu)
        Jt = to_unified(xyz, list(range(25)), K_JOINTS)
        Jt = to_z_up(interp_nan(Jt))
        Jr = resample(Jt, fps_k, FPS_OUT)
        d = descriptors(Jr, FPS_OUT); d.update(info); d["file"] = stem; d["sensor"] = "kinect"; d["src_fps"] = fps_k
        rows.append(d); seqs[stem] = normalize_sequence(Jr, 64)
    dfk = pd.DataFrame(rows); dfk.to_csv(DD / "taichi_kinect.csv", index=False)
    np.savez_compressed(DD / "taichi_kinect_sequences.npz", keys=np.array(list(seqs.keys())), X=np.stack(list(seqs.values())))
    print("kinect segments", len(dfk)); print(dfk.groupby("gesture").size().to_dict())


if __name__ == "__main__":
    main()
