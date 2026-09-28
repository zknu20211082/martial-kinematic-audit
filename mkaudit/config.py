"""Storage locations of the pipeline.

Locations are read from ``configs/paths.yaml`` (or from the YAML file named by the environment variable
``MKA_CONFIG``). Each entry can be overridden by an environment variable, which takes precedence:

    MKA_DATA_ROOT     data_root    raw datasets (Kyokushin, UMONS-TAICHI, hmdb51, UCF101)
    MKA_WORK_DIR      work_dir     derived data (skeleton/descriptor tables, sequences, video index, predictions)
    MKA_CACHE_DIR     cache_dir    caches (2D poses, VideoMAE features, rendered videos, pre-trained weights)
    MKA_HF_CACHE      hf_cache     Hugging Face model cache (default: <cache_dir>/hf)
    MKA_RESULTS_DIR   results_dir  result tables (CSV / JSON)
    MKA_FIGURES_DIR   figures_dir  paper figures

Relative paths in the YAML file are resolved against the repository root; relative paths given through environment
variables are resolved against the current working directory.

The short names ``DATA`` (data root), ``DD`` (derived data), ``CACHE``, ``HF_CACHE``, ``RES`` (results) and ``FIG``
(figures) are the names used throughout the experiment code.
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "data" / "metadata"                  # authors' metadata tables (participants, gestures)
SYNTH_BENCHMARK_DIR = REPO_ROOT / "data" / "synthetic_benchmark"  # the 200 rendered videos used in the paper

_KEYS = {  # key -> (environment variable, default)
    "data_root": ("MKA_DATA_ROOT", "./data/raw"),
    "work_dir": ("MKA_WORK_DIR", "./work"),
    "cache_dir": ("MKA_CACHE_DIR", "./cache"),
    "hf_cache": ("MKA_HF_CACHE", ""),
    "results_dir": ("MKA_RESULTS_DIR", "./results"),
    "figures_dir": ("MKA_FIGURES_DIR", "./figures"),
}


def load_paths(config=None):
    """Return {key: absolute Path} for all storage locations (see module docstring)."""
    cfg = Path(config or os.environ.get("MKA_CONFIG") or REPO_ROOT / "configs" / "paths.yaml")
    values = {}
    if cfg.is_file():
        with open(cfg, encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
    out = {}
    for key, (env, default) in _KEYS.items():
        if os.environ.get(env, "").strip():
            out[key] = Path(os.path.abspath(os.path.expanduser(os.environ[env].strip())))
            continue
        v = values.get(key)
        v = default if v is None else str(v).strip()
        if not v:  # only hf_cache may be empty -> derived from cache_dir below
            out[key] = None
            continue
        p = Path(os.path.expanduser(v))
        out[key] = Path(os.path.abspath(p if p.is_absolute() else REPO_ROOT / p))
    for key in ("data_root", "work_dir", "cache_dir", "results_dir", "figures_dir"):
        if out[key] is None:
            out[key] = Path(os.path.abspath(REPO_ROOT / _KEYS[key][1]))
    if out["hf_cache"] is None:
        out["hf_cache"] = out["cache_dir"] / "hf"
    return out


PATHS = load_paths()
DATA = PATHS["data_root"]
DD = PATHS["work_dir"]
CACHE = PATHS["cache_dir"]
HF_CACHE = PATHS["hf_cache"]
RES = PATHS["results_dir"]
FIG = PATHS["figures_dir"]
DATA_ROOT, WORK_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR = DATA, DD, CACHE, RES, FIG


def ensure_dirs(*paths):
    """Create output directories (default: work, cache and results directories)."""
    for p in (paths or (DD, CACHE, RES)):
        Path(p).mkdir(parents=True, exist_ok=True)


def metadata_path(dataset, name):
    """Authors' metadata table such as ``participants.csv`` or ``gesture_classes.csv``.

    The experiments read ``<data_root>/<dataset>/<name>``; when that file is absent the copy shipped in
    ``data/metadata/<dataset>/<name>`` is used."""
    p = DATA / dataset / name
    return p if p.exists() else METADATA_DIR / dataset / name


def describe():
    """Human-readable summary of the resolved locations."""
    return "\n".join(f"{k:12s} {v}" for k, v in PATHS.items())
