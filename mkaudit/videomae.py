"""VideoMAE-base (Kinetics-400 fine-tuned) appearance stream: frame sampling and model loading (paper Sec. 1.4)."""
import av
import numpy as np

from .config import HF_CACHE

MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"  # the paper used Hub revision 488eb9a0565f257b32866000305c8178965eb9f6


def read_all(path):
    frames = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return frames


def sample_clips(n, n_clips=3, L=16):
    idxs = []
    for k in range(n_clips):
        center = (k + 0.5) * n / n_clips
        step = max(1, n // (L * 2))
        base = np.arange(L) - L / 2
        ii = np.clip(np.round(center + base * step).astype(int), 0, n - 1)
        idxs.append(ii)
    return idxs


def load_processor(cache_dir=None):
    from transformers import VideoMAEImageProcessor
    return VideoMAEImageProcessor.from_pretrained(MODEL, cache_dir=str(cache_dir or HF_CACHE))


def load_videomae(cache_dir=None):
    """transformers 5.x names the attention biases query.bias / value.bias, but the checkpoint stores q_bias / v_bias,
    so they are silently zero-initialised. Copy them back in (key has no bias by design)."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEForVideoClassification
    cache_dir = str(cache_dir or HF_CACHE)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL, cache_dir=cache_dir)
    ck = hf_hub_download(MODEL, "model.safetensors", cache_dir=cache_dir)  # the checkpoint file loaded above
    sd = model.state_dict(); n = 0
    with safe_open(ck, "pt") as s:
        for k in s.keys():
            if k.endswith(".q_bias") or k.endswith(".v_bias"):
                tgt = k.replace(".q_bias", ".query.bias").replace(".v_bias", ".value.bias")
                if tgt in sd:
                    sd[tgt] = s.get_tensor(k); n += 1
    model.load_state_dict(sd)
    print(f"patched {n} attention bias tensors")
    return model
