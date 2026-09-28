"""Parse all Kyokushin karate C3D files -> unified skeletons, single-strike segments, descriptors.

Paper: Secs. 1.1-1.2 and Table 1 (1411 files, 4524 strike segments). Outputs in <work_dir>:
karate_strikes.csv (54 descriptors per strike), karate_files.csv (per-file log), karate_sequences.npz
(normalised 64-frame sequences for ST-GCN and the synthetic projections)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import re, glob, os
import numpy as np, pandas as pd
from tqdm import tqdm
from mkaudit.config import DD, ensure_dirs
from mkaudit.skeleton import JOINTS, interp_nan, to_z_up, resample
from mkaudit.descriptors import descriptors, normalize_sequence
from mkaudit.mocap_karate import ROOT, TECH, COND, RENAME, FPS_OUT, FILE_PATTERN, load_subject_skeleton, segment_strikes


def main():
    ensure_dirs(DD)
    files = sorted(glob.glob(str(ROOT / "*" / "*" / "*.c3d")))
    rows, seq_store, meta = [], {}, []
    n_virtual = 0
    for f in tqdm(files, desc="karate"):
        stem = os.path.splitext(os.path.basename(f))[0]
        canon = RENAME.get(stem, stem)
        m = re.match(FILE_PATTERN, canon)
        if not m:
            continue
        date, code, S, E, Tt = m.groups()
        if S not in TECH or E not in COND:
            meta.append({"file": stem, "error": f"unknown code {S} {E}"}); continue
        try:
            Jt, rate, ev, uv, nl = load_subject_skeleton(f, code)
        except Exception as ex:
            meta.append({"file": stem, "error": str(ex)}); continue
        n_virtual += uv == len(JOINTS)
        Jt = interp_nan(Jt)
        Jt = to_z_up(Jt)
        Jr = resample(Jt, rate, FPS_OUT)
        segs = segment_strikes(Jr, FPS_OUT, S)
        meta.append({"file": stem, "canon": canon, "code": code, "tech": S, "cond": E, "trial": Tt, "rate": rate,
                     "n_frames": len(Jt), "n_events": len(ev), "n_strikes": len(segs), "virtual_joints": uv, "n_labels": nl})
        for k, (s, e, p, v) in enumerate(segs):
            seg = Jr[s:e]
            d = descriptors(seg, FPS_OUT)
            d.update({"file": canon, "code": code, "tech": S, "tech_name": TECH[S], "cond": E, "cond_name": COND[E],
                      "trial": Tt, "strike_idx": k, "peak_frame": p, "peak_speed_strike": v})
            rows.append(d)
            seq_store[f"{canon}#{k}"] = normalize_sequence(seg, 64)
    df = pd.DataFrame(rows)
    df.to_csv(DD / "karate_strikes.csv", index=False)
    pd.DataFrame(meta).to_csv(DD / "karate_files.csv", index=False)
    np.savez_compressed(DD / "karate_sequences.npz", keys=np.array(list(seq_store.keys())), X=np.stack(list(seq_store.values())))
    print("files", len(files), "parsed", len(meta), "all-virtual-joint files", n_virtual, "strikes", len(df))
    print(df.groupby(["tech_name", "cond_name"]).size())


if __name__ == "__main__":
    main()
