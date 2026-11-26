"""Shared configuration of the additional probe-validation experiments.

The paper's add-on experiments are all performed on the last hidden layers of
VGG-11 and Res-18 trained on the four-class CIFAR subset.  They share the same
eight sites and the same cached features.
"""
from pathlib import Path

from ...common.config import (CLEAN_ROOT, FEATURE_ROOT, FIGURE_ROOT, MODELS as MAIN_MODELS,
                             MODEL_ROOT, REPO_ROOT, RESULT_ROOT)

DATA_FILE = CLEAN_ROOT / "classification" / "data.npz"
CACHE = FEATURE_ROOT

MODELS = ["VGG-11", "Res-18"]
N_CLASSES = 4
SEED_BASE = 20261108
SELECT_FRAC = 0.20

# site -> (model, layer name, flattened dimension, description)
SITES = {
    "VGG_L3": ("VGG-11", "re_lu_3", 4096, "conv 64 @ 8x8"),
    "VGG_L5": ("VGG-11", "re_lu_5", 2048, "conv 128 @ 4x4"),
    "VGG_L7": ("VGG-11", "re_lu_7", 1024, "conv 256 @ 2x2"),
    "VGG_L9": ("VGG-11", "re_lu_9", 256, "dense 256 (last hidden)"),
    "RES_L3": ("Res-18", "re_lu_6", 4096, "stage2 block1 64 @ 8x8"),
    "RES_L4": ("Res-18", "re_lu_8", 4096, "stage2 block2 64 @ 8x8"),
    "RES_L6": ("Res-18", "re_lu_12", 2048, "stage3 block2 128 @ 4x4"),
    "RES_L8": ("Res-18", "re_lu_16", 1024, "stage4 block2 256 @ 2x2"),
}
SITE_ORDER = ["VGG_L3", "VGG_L5", "VGG_L7", "VGG_L9",
              "RES_L3", "RES_L4", "RES_L6", "RES_L8"]


def features_path(model_key: str) -> Path:
    """Cached 4-class features: dataset/features/features4_<Model>.npz."""
    return CACHE / f"features4_{model_key}.npz"


def checkpoint_path(model_key: str) -> Path:
    return MODEL_ROOT / "classification" / "clean" / MAIN_MODELS[model_key]["ckpt"]


def result_dir(add_id: str) -> Path:
    d = RESULT_ROOT / add_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure_dir(add_id: str) -> Path:
    d = FIGURE_ROOT / add_id
    d.mkdir(parents=True, exist_ok=True)
    return d
