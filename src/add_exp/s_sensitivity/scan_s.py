#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection-ratio sensitivity scan (s = 0.4 / 0.6 / 0.8).

The main experiment uses s = 0.2.  This script reproduces the same selection
rule with larger ratios, reusing the main-experiment probe weights and noise
realizations:

    keep only samples the network classifies correctly;
    per OVR facet, sort by the last-layer MD-LP score and keep the
    floor(s * n_pos) / floor(s * n_neg) smallest-margin samples;
    report the whole-network OVR accuracy (mean over facets).

s = 0.2 itself is taken from ``result/criteria/clean/classification``.

Usage:
    python -m src.add_exp.s_sensitivity.scan_s --repeats 10
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from ...common.config import (MODELS, TASK_CFG, attack_npy, perturbation_file,
                             probe_weights_dir)
from ...models.loader import layer_names, load_model
from ..common.sites import result_dir

CONDS = ["attack", "gauss", "salt", "move", "occ"]
ATTACK_FILE = {"VGG-11": "VGG_pgd.npy", "Res-18": "Res_pgd.npy",
               "ViT-8": "ViT_pgd.npy", "Mix-8": "Mix_pgd.npy"}
S_LIST = [0.4, 0.6, 0.8]
BATCH = 256
TASK = "classification"


def load_data():
    path = TASK_CFG[TASK]["dataset_file"]
    if path.suffix == ".npz":
        with np.load(path) as d:
            x, y = d["x_train"], d["y_train"]
    else:
        with open(path, "rb") as f:
            x, y, _, _ = pickle.load(f)
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x, np.asarray(y).reshape(-1).astype(np.int32)


def clean_scores(model, model_key, x):
    """Return (network argmax predictions, last-layer probe projections [N, 4])."""
    last = layer_names(model, model_key)[-1]
    d = probe_weights_dir(TASK, model_key, "clean")
    w = np.load(d / f"{last}_w.npy").astype(np.float32)
    b = np.load(d / f"{last}_b.npy").astype(np.float32)
    view = tf.keras.Model(inputs=model.input,
                          outputs=[model.get_layer(last).output, model.output])
    feats, preds = [], []
    for s in range(0, len(x), BATCH):
        f, out = view(x[s:s + BATCH], training=False)
        f = tf.reshape(f, [f.shape[0], -1]).numpy()
        feats.append(f @ w.T + b)
        preds.append(np.argmax(out.numpy(), axis=1))
    return np.concatenate(preds), np.concatenate(feats)


def select_indices(preds, P, y, s, n_class):
    correct = preds == y
    sel = []
    for k in range(n_class):
        lab = (y == k)
        vals = (2.0 * lab - 1.0) * P[:, k]
        n_pos = int(lab.sum())
        n_neg = len(y) - n_pos
        n1, n0 = int(np.floor(s * n_pos)), int(np.floor(s * n_neg))
        idx1 = np.where(lab & correct)[0]
        idx0 = np.where(~lab & correct)[0]
        idx1 = idx1[np.argsort(vals[idx1], kind="stable")][:n1]
        idx0 = idx0[np.argsort(vals[idx0], kind="stable")][:n0]
        sel.append((idx0, idx1))
    return sel


def ovr_from_sel(pred_all, sel):
    accs = []
    for k, (idx0, idx1) in enumerate(sel):
        correct = int(np.sum(pred_all[idx1] == k)) + int(np.sum(pred_all[idx0] != k))
        accs.append(correct / max(len(idx0) + len(idx1), 1))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--s-list", default="0.4,0.6,0.8")
    ap.add_argument("--out", default=str(result_dir("add_exp/s_sensitivity") /
                                         "noise_s_rank_maincal.json"))
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    s_list = [float(v) for v in args.s_list.split(",")]

    x, y = load_data()
    n_class = int(TASK_CFG[TASK]["num_classes"])
    print(f"n={len(y)} classes={n_class} repeats={args.repeats}", flush=True)

    sel = {}
    for key in models:
        model = load_model(TASK, key)
        preds, P = clean_scores(model, key, x)
        for s in s_list:
            pair = select_indices(preds, P, y, s, n_class)
            sel[(key, s)] = pair
            n = len(np.unique(np.concatenate([np.concatenate(p) for p in pair])))
            print(f"  {key} s={s}: selected {n}", flush=True)
        del model, P, preds
        tf.keras.backend.clear_session()

    acc = {(c, s, k): [] for c in CONDS for s in s_list for k in models}
    for rep in range(args.repeats):
        for cond in CONDS:
            for key in models:
                model = load_model(TASK, key)
                if cond == "attack":
                    xp = np.load(attack_npy(TASK, key, rep))
                else:
                    xp = np.load(perturbation_file(TASK, cond, rep))
                need = np.unique(np.concatenate(
                    [np.concatenate(p) for s in s_list for p in sel[(key, s)]]))
                pred = model.predict(xp[need], batch_size=BATCH, verbose=0).argmax(axis=1)
                pred_all = np.full(len(y), -1, dtype=np.int32)
                pred_all[need] = pred
                for s in s_list:
                    acc[(cond, s, key)].append(ovr_from_sel(pred_all, sel[(key, s)]))
                del xp, pred, pred_all, model
                tf.keras.backend.clear_session()
            print(f"  rep{rep} {cond} done", flush=True)

    out = {
        "metric": "main-experiment caliber: OVR mean on net-correct samples, "
                  "per-facet top floor(s*n_pos)/floor(s*n_neg) by last-layer probe score",
        "s_list": s_list,
        "s0.2_source": "result/criteria/clean/classification/<Model>_robustness.pkl (10 repeats)",
        "repeats": args.repeats,
        "results": {},
    }
    for cond in CONDS:
        out["results"][cond] = {}
        for s in s_list:
            out["results"][cond][str(s)] = {
                key: {"mean": float(np.mean(acc[(cond, s, key)])),
                      "std": float(np.std(acc[(cond, s, key)]))}
                for key in models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[ok]", args.out, flush=True)


if __name__ == "__main__":
    main()
