"""Hidden-layer feature extraction shared by the additional experiments.

Features are cached in ``dataset/features/features4_<Model>.npz`` and reused by
the weight-stability and rotation experiments.
"""
import argparse

import numpy as np

from ...common.config import TASK_CFG
from .sites import MODELS, SITES, features_path


def load_classification_data():
    """Return (x_train, y_train) of the four-class CIFAR subset."""
    path = TASK_CFG["classification"]["dataset_file"]
    if path.suffix == ".npz":
        with np.load(path) as d:
            x, y = d["x_train"], d["y_train"]
        x = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
    else:
        import pickle
        with open(path, "rb") as f:
            x, y, _, _ = pickle.load(f)
        x = np.asarray(x, dtype=np.float32)
    return x, np.asarray(y).reshape(-1).astype(np.int32)


def build_submodel(model_key, layers):
    """Return (full model, sub-model exposing ``layers``)."""
    import tensorflow as tf
    from ...models.loader import load_model

    model = load_model("classification", model_key, exp="clean")
    sub = tf.keras.Model(model.input,
                         [model.get_layer(name).output for name in layers])
    return model, sub



def extract(sub, x, batch=256):
    """Batch-extract flattened features aligned with ``sub.outputs``."""
    outs = [[] for _ in range(len(sub.outputs))]
    for s in range(0, len(x), batch):
        vals = sub.predict(np.asarray(x[s:s + batch], dtype=np.float32), verbose=0)
        if not isinstance(vals, list):
            vals = [vals]
        for i, v in enumerate(vals):
            v = np.asarray(v, dtype=np.float32)
            if v.ndim > 2:
                v = v.reshape(v.shape[0], -1)
            outs[i].append(v)
    return [np.concatenate(o, axis=0) for o in outs]


def extract_all(models=None):
    """Extract and cache features for the requested models."""
    models = models or MODELS
    x, y = load_classification_data()
    print(f"[data] train={x.shape} classes={np.bincount(y)}", flush=True)

    for model_key in models:
        sites = {k: v for k, v in SITES.items() if v[0] == model_key}
        layers = [v[1] for v in sites.values()]
        print(f"[{model_key}] extracting {len(layers)} sites from {len(x)} samples ...", flush=True)
        model, sub = build_submodel(model_key, layers)
        feats = extract(sub, x)
        pred = np.argmax(model.predict(x, batch_size=256, verbose=0), axis=1).astype(np.int32)

        arr = dict(y_train=y, pred_train=pred)
        for (site, (_, _, d, _)), a in zip(sites.items(), feats):
            if a.shape[1] != d:
                raise RuntimeError(f"{site}: expected d={d}, got {a.shape[1]}")
            arr[f"train_{site}"] = a
            print(f"  {site:>7s} d={d:5d} {a.shape}", flush=True)

        path = features_path(model_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arr)
        print(f"[{model_key}] saved {path}  net_train_acc={np.mean(pred == y):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="cache hidden-layer features for add-on experiments")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    extract_all(args.models)


if __name__ == "__main__":
    main()
