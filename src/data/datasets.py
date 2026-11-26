"""Dataset reading (UTKFace age regression, CIFAR-10 four-class / full subsets)."""
import pickle

import numpy as np
import tensorflow as tf

from ..common.config import TASK_CFG, dataset_file


def _decode_images(x):
    """Decode compact on-disk image arrays back to float32 in [0, 1]."""
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    if x.dtype == np.float16:
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _load_part(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_raw(task, exp="clean"):
    """Return (x_train, y_train, x_test, y_test).

    A dataset path can be a single ``.npz`` / ``.pkl`` file or a directory of
    ``part-*.npz`` shards (used to keep every file below GitHub's 100 MB limit).
    """
    path = dataset_file(task, exp)
    keys = TASK_CFG[task]["split_keys"]

    if path.is_dir():
        parts = sorted(path.glob("part-*.npz"))
        if not parts:
            raise FileNotFoundError(f"no part-*.npz shards under {path}")
        arrays = [_load_part(p) for p in parts]
        x_train = np.concatenate([a["x_train"] for a in arrays], axis=0)
        y_train = np.concatenate([a["y_train"] for a in arrays], axis=0)
        ref = next(a for a in arrays if "x_test" in a)
        return x_train, y_train, ref["x_test"], ref["y_test"]

    if path.suffix == ".npz":
        with np.load(path) as data:
            return tuple(data[k] for k in keys)

    with open(path, "rb") as f:
        x_train, y_train, x_test, y_test = pickle.load(f)
    return x_train, y_train, x_test, y_test


def load_train_xy(task, exp="clean"):
    """Training/evaluation arrays with images as float32 in [0, 1].

    Regression labels stay float64 so that quantile thresholds are bit-identical
    to the ones used for the paper.
    """
    x_train, y_train, x_test, y_test = load_raw(task, exp)

    if task == "regression":
        return (_decode_images(x_train), np.asarray(y_train).reshape(-1).astype(np.float64),
                _decode_images(x_test), np.asarray(y_test).reshape(-1).astype(np.float64))

    y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int32)
    return _decode_images(x_train), y_train, _decode_images(x_test), y_test


def load_probe_labels(task, exp="clean"):
    """Label vector used by the probe facets (one-dimensional)."""
    _, y_train, _, _ = load_train_xy(task, exp)
    return np.asarray(y_train).reshape(-1)


def to_onehot(y, num_classes):
    return tf.keras.utils.to_categorical(np.asarray(y).reshape(-1), num_classes=num_classes)
