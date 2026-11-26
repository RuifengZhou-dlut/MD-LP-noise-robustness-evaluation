"""Project-level configuration: paths, tasks and the model registry.

Every path is derived from this file, so the repository can be cloned anywhere
and moved between machines without editing scripts.

Layout (all paths relative to the repository root)::

    dataset/clean/<task>/                 processed clean splits
    dataset/noise/<condition>/<task>/     generated perturbations (by noise type)
    dataset/features/                     cached hidden-layer features (add experiments)
    model/<task>/<clean|noise_trained>/   trained weights
    result/probe/<exp>/<task>/<Model>/    MD-LP weights and probe accuracy
    result/criteria/<exp>/<task>/         robustness / std / se / drop matrices
    result/repeat_raw/<task>/              per-repeat raw matrices (SE audit)
    result/add_<id>_<name>/               data of each additional experiment
    figure/generated/<exp>/<task>/             figures of the main experiment
    figure/add_<id>_<name>/               figures of each additional experiment

Experiment dimension (``exp``):
    clean : networks trained on clean samples (main experiment)
    noise : networks trained on Gaussian / salt-pepper augmented samples
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]        # <repo>/src
REPO_ROOT = Path(os.environ.get("MDLP_ROOT", CODE_ROOT.parent))  # <repo>

DATASET_ROOT = REPO_ROOT / "dataset"
CLEAN_ROOT = DATASET_ROOT / "clean"
NOISE_ROOT = DATASET_ROOT / "noise"
FEATURE_ROOT = DATASET_ROOT / "features"

MODEL_ROOT = REPO_ROOT / "model"
RESULT_ROOT = REPO_ROOT / "result"
PROBE_ROOT = RESULT_ROOT / "probe"
CRITERIA_ROOT = RESULT_ROOT / "criteria"
REPEAT_ROOT = RESULT_ROOT / "repeat_raw"

FIGURE_ROOT = REPO_ROOT / "figure"
MAIN_FIGURE_ROOT = FIGURE_ROOT / "generated"
FIGURES_ROOT = FIGURE_ROOT  # backward-compatible alias

TASKS = ("regression", "classification", "classification_full")

# ------------------------------------------------------------------ task level
TASK_CFG = {
    "regression": {
        "cache_key": "reg",
        "perturbation_key": "regression",
        "n_splits": 9,                 # nine age-quantile thresholds
        "grouping": "quantile",        # positive class: y < threshold
        "num_classes": 1,              # sigmoid regression head
        "activation": "sigmoid",
        "repeats": {"attack": 10, "gauss": 10, "salt": 10, "move": 10, "occ": 10},
        "dataset_file": CLEAN_ROOT / "regression" / "parts",
        "split_keys": ("x_train", "y_train", "x_test", "y_test"),
        "image_encoding": "float16",
    },
    "classification": {
        "cache_key": "cla",
        "perturbation_key": "classification",
        "n_splits": 4,                 # four one-vs-rest facets
        "grouping": "class",
        "num_classes": 4,
        "activation": "softmax",
        "repeats": {"attack": 10, "gauss": 10, "salt": 10, "move": 10, "occ": 10},
        "dataset_file": CLEAN_ROOT / "classification" / "data.npz",
        "split_keys": ("x_train", "y_train", "x_test", "y_test"),
        "image_encoding": "uint8",
    },
    "classification_full": {
        "cache_key": "cla_full",
        "perturbation_key": "classification_full",
        "n_splits": 10,                # ten one-vs-rest facets (full CIFAR-10)
        "grouping": "class",
        "num_classes": 10,
        "activation": "softmax",
        "repeats": {"attack": 10, "gauss": 10, "salt": 10, "move": 10, "occ": 10},
        "dataset_file": CLEAN_ROOT / "classification_full" / "data.npz",
        "split_keys": ("x_train", "y_train", "x_test", "y_test"),
        "image_encoding": "uint8",
    },
}

CONDITIONS = ("base", "attack", "gauss", "salt", "move", "occ")
PERTURBATIONS = ("attack", "gauss", "salt", "move", "occ")

# ------------------------------------------------------------------ experiment dimension
EXPS = ("clean", "noise")
EXP_ALIAS = {}
NOISE_CONDITIONS = ("base", "gauss", "salt")


def normalize_exp(exp: str) -> str:
    """Validate the experiment name."""
    if exp not in EXPS:
        raise ValueError(f"unknown experiment {exp!r}; choose from {EXPS}")
    return exp


def exp_dirname(exp: str) -> str:
    """Folder name of the experiment branch under ``model/<task>/``."""
    return "noise_trained" if normalize_exp(exp) == "noise" else "clean"


# ------------------------------------------------------------------ model registry
# l_label    : indices into model.layers used as substructure probes
# pred_layer : truncation layer used to compute the network prediction
MODELS = {
    "VGG-11": {
        "builder": "vggnet16",
        "nn": [32, 32, 64, 64, 128, 128, 256, 256],
        "l_label": [2, 6, 10, 13, 17, 20, 24, 27, 32, 35],
        "pred_layer": "re_lu_9",
        "ckpt": "VGG_11.h5",
        "pgd": "VGG_pgd.npy",
        "regression": {"opt": "sgd", "lr": 0.1, "wd": 1e-4, "nesterov": False, "mixup": 0.05},
        "classification": {"opt": "sgd", "lr": 0.1, "wd": 1e-4, "nesterov": True, "mixup": 0.2},
    },
    "Res-18": {
        "builder": "resnet18",
        "nn": [32, 32, 32, 64, 64, 128, 128, 256, 256],
        "l_label": [3, 11, 18, 27, 34, 43, 50, 59, 66],
        "pred_layer": "re_lu_16",
        "ckpt": "Res_18.h5",
        "pgd": "Res_pgd.npy",
        "regression": {"opt": "sgd", "lr": 0.1, "wd": 1e-4, "nesterov": True, "mixup": 0.05},
        "classification": {"opt": "sgd", "lr": 0.1, "wd": 1e-4, "nesterov": True, "mixup": 0.2},
    },
    "ViT-8": {
        "builder": "vit",
        "nn": None,
        "l_label": [2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        "pred_layer": "dense_16",
        "ckpt": "ViT_8.h5",
        "pgd": "ViT_pgd.npy",
        "regression": {"opt": "adamw", "lr": 5e-4, "wd": 1e-5, "mixup": 0.05},
        "classification": {"opt": "adamw", "lr": 2e-4, "wd": 1e-5, "mixup": 0.2},
    },
    "Mix-8": {
        "builder": "mlp_mixer",
        "nn": None,
        "l_label": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
        "pred_layer": "reg_dense_0",
        "ckpt": "Mix_8.h5",
        "pgd": "Mix_pgd.npy",
        "regression": {"opt": "adamw", "lr": 5e-4, "wd": 1e-5, "mixup": 0.05},
        "classification": {"opt": "adamw", "lr": 5e-4, "wd": 1e-5, "mixup": 0.2},
    },
}

TRAIN_DEFAULTS = dict(epochs=200, warmup_epochs=5, batch_size=128, image_size=32, pad=4, last_n=20)


# ------------------------------------------------------------------ path helpers
def probe_dir(task: str, model: str, exp: str = "clean") -> Path:
    return PROBE_ROOT / normalize_exp(exp) / task / model


def probe_acc_pkl(task: str, model: str, exp: str = "clean") -> Path:
    return probe_dir(task, model, exp) / "probe_acc.pkl"


def probe_weights_dir(task: str, model: str, exp: str = "clean") -> Path:
    return probe_dir(task, model, exp)


def exp_dir(exp: str = "clean") -> Path:
    return CRITERIA_ROOT / normalize_exp(exp)


def task_dir(task: str, exp: str = "clean") -> Path:
    return exp_dir(exp) / task


def robust_pkl(task: str, model: str, exp: str = "clean") -> Path:
    return task_dir(task, exp) / f"{model}_robustness.pkl"


def robust_std_pkl(task: str, model: str, exp: str = "clean") -> Path:
    return task_dir(task, exp) / f"{model}_std.pkl"


def robust_se_pkl(task: str, model: str, exp: str = "clean") -> Path:
    return task_dir(task, exp) / f"{model}_se.pkl"


def drop_pkl(task: str, model: str, exp: str = "clean") -> Path:
    return task_dir(task, exp) / f"{model}_drop.pkl"


def summary_csv(task: str, exp: str = "clean") -> Path:
    return exp_dir(exp) / f"{task}.csv"


def dataset_file(task: str, exp: str = "clean") -> Path:
    return TASK_CFG[task]["dataset_file"]


def split_file(task: str, exp: str = "clean") -> Path:
    return dataset_file(task, exp)


def perturbation_file(task: str, condition: str, repeat: int) -> Path:
    """dataset/noise/<condition>/<task>/<repeat>.npy."""
    return NOISE_ROOT / condition / TASK_CFG[task]["perturbation_key"] / f"{repeat:02d}.npy"


def attack_dir(task: str, repeat: int, exp: str = "clean") -> Path:
    return NOISE_ROOT / "attack" / TASK_CFG[task]["perturbation_key"] / f"{repeat:02d}"


def attack_npy(task: str, model: str, repeat: int, exp: str = "clean") -> Path:
    return attack_dir(task, repeat, exp=exp) / MODELS[model]["pgd"]


def checkpoint_path(task: str, model: str, exp: str = "clean") -> Path:
    return MODEL_ROOT / task / exp_dirname(exp) / MODELS[model]["ckpt"]


def repeat_raw_dir(task: str) -> Path:
    return REPEAT_ROOT / task


def figure_dir(task: str, exp: str = "clean") -> Path:
    return MAIN_FIGURE_ROOT / normalize_exp(exp) / task


def add_result_dir(add_id: str) -> Path:
    return RESULT_ROOT / add_id


def add_figure_dir(add_id: str) -> Path:
    return FIGURE_ROOT / "generated" / add_id
