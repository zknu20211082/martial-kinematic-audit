"""E6: parameter counts and per-clip latency of each pipeline component on the RTX 3090 (batch 1).

Paper: Sec. 2.7 (deployment cost). Needs a CUDA GPU, <work_dir>/video_index.csv and (for the Qwen2.5-VL row) the raw
outputs e3_vlm_{wild,synth}.csv in <results_dir>. Output: <results_dir>/e6_efficiency.csv."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

from mkaudit.cuda_dlls import expose_torch_cuda_dlls
expose_torch_cuda_dlls()  # Windows: must run before onnxruntime / rtmlib are imported
import time
import torch
import numpy as np, pandas as pd
from mkaudit.config import DD, RES, HF_CACHE, ensure_dirs
from mkaudit.stgcn import STGCN
from mkaudit.recognition import COCO_BONES


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000


def main():
    ensure_dirs(RES)
    rows = []
    # skeleton ST-GCN (2D, 2 persons, 48 frames)
    m = STGCN(3, 17, COCO_BONES, 7, n_persons=2, base=64).cuda().eval()
    x = torch.randn(1, 3, 48, 17, 2).cuda()
    with torch.no_grad():
        rows.append({"component": "ST-GCN skeleton classifier (ours)", "params_M": sum(p.numel() for p in m.parameters()) / 1e6, "latency_ms": bench(lambda: m(x))})
    # RTMPose pipeline on one real clip (detector + pose, every 2nd frame)
    from rtmlib import Body
    import av
    body = Body(mode="balanced", backend="onnxruntime", device="cuda")
    idx = pd.read_csv(DD / "video_index.csv"); path = idx.path[0]
    frames = []
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % 2 == 0:
                frames.append(fr.to_ndarray(format="bgr24"))
    body(frames[0]); t = time.time()
    for f in frames:
        body(f)
    dt = (time.time() - t) * 1000
    rows.append({"component": "RTMPose-m + YOLOX-m 2D pose (ONNX CUDA)", "params_M": 25.3 + 13.7, "latency_ms": dt, "note": f"{len(frames)} frames of one clip, {dt / len(frames):.1f} ms/frame"})
    # VideoMAE-base 3 clips x 16 frames
    from transformers import VideoMAEForVideoClassification
    vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", cache_dir=str(HF_CACHE)).cuda().eval().half()
    xv = torch.randn(3, 16, 3, 224, 224).cuda().half()
    with torch.no_grad():
        rows.append({"component": "VideoMAE-base K400 (3 x 16 frames)", "params_M": sum(p.numel() for p in vm.parameters()) / 1e6, "latency_ms": bench(lambda: vm(pixel_values=xv), n=10)})
    del vm; torch.cuda.empty_cache()
    # Qwen2.5-VL-7B generation latency (from E3 logs if present)
    for f in ("e3_vlm_wild.csv", "e3_vlm_synth.csv"):
        if (RES / f).exists():
            d = pd.read_csv(RES / f)
            rows.append({"component": f"Qwen2.5-VL-7B-Instruct bf16, 8 frames, JSON output ({f})", "params_M": 8290, "latency_ms": float(d.gen_s.mean() * 1000), "note": f"n={len(d)} generations"})
    df = pd.DataFrame(rows); df.to_csv(RES / "e6_efficiency.csv", index=False); print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
