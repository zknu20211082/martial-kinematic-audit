"""mkaudit: motion-capture kinematic knowledge mining and kinematic-hallucination auditing of video LLMs for
martial-arts videos.

Modules
-------
config        storage locations (configs/paths.yaml + MKA_* environment variables)
skeleton      unified 19-joint skeleton, gap filling, resampling, vertical-axis alignment, smoothing
descriptors   the 54 kinematic descriptors and sequence normalisation
mocap_karate  Kyokushin karate C3D parsing and single-strike segmentation
mocap_taichi  UMONS-TAICHI Qualisys / Kinect parsing
synth         multi-view projection, stick-figure rendering, exact kinematic ground truth
stgcn         compact ST-GCN with training / prediction helpers
recognition   2D-skeleton loading, augmentation and perturbations for in-the-wild video
videomae      VideoMAE appearance stream (with the attention-bias loading fix for transformers 5.x)
cuda_dlls     Windows helper exposing torch's CUDA DLLs to onnxruntime
vlm           Qwen2.5-VL prompt template, frame sampling, JSON parsing
verifier      kinematic claim verifier (KCV), re-scoring, KF and hallucination rate
skill         leave-one-subject-out skill regression, KKB rule mining
stats         bootstrap CIs, exact McNemar, Benjamini-Hochberg, age-partial correlation
utils         small I/O helpers

Modules are imported on demand (``from mkaudit.verifier import rescore``); importing the package itself does not load
torch or any model.
"""
__version__ = "1.0.0"
