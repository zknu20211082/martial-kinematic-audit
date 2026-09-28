"""VideoMAE-base (Kinetics-400 fine-tuned) clip features for every video: 3 temporal clips x 16 frames, mean-pooled tokens + logits.

Paper: Sec. 1.4 (appearance stream and the Kinetics-400 zero-shot mapping). Feeds 20 and 22.
Output: <cache_dir>/videomae/<clip>.npz (feat: 3x768, logits: 3x400). Requires a CUDA GPU."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import time
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from mkaudit.config import CACHE, DD
from mkaudit.videomae import read_all, sample_clips, load_processor, load_videomae

OUT = CACHE / "videomae"


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(DD / "video_index.csv")
    proc = load_processor()
    model = load_videomae().cuda().eval().half()
    todo = [r for r in idx.itertuples() if not (OUT / (r.clip + ".npz")).exists()]
    t0 = time.time()
    for r in tqdm(todo, desc="videomae"):
        try:
            frames = read_all(r.path)
        except Exception as ex:
            print("decode fail", r.clip, ex); continue
        if len(frames) < 2:
            continue
        feats, logits = [], []
        for ii in sample_clips(len(frames)):
            clip = [frames[i] for i in ii]
            inp = proc(clip, return_tensors="pt")["pixel_values"].cuda().half()
            out = model(pixel_values=inp, output_hidden_states=True)
            feats.append(out.hidden_states[-1].mean(1).float().cpu().numpy()[0])
            logits.append(out.logits.float().cpu().numpy()[0])
        np.savez_compressed(OUT / (r.clip + ".npz"), feat=np.stack(feats), logits=np.stack(logits))
    print(f"clips {len(todo)} time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
