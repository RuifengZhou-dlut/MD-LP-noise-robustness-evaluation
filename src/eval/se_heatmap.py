#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: standard error (SE) of the mean per-layer accuracy.

Layout
------
Rows    : 5 stacked panels, one per corruption (PGD, Gauss, Salt, Move, Occ)
In panel: 4 networks (VGGNet, ResNet, ViT, MLP-Mixer)
Columns : Classification block (L0..L9, Out) | Regression block (L0..L9, Out),
          22 contiguous columns; the two blocks touch (no spacer), hidden layers
          and the network output are adjacent as well.
Colours : classification = yellow -> red (high -> low lightness);
          regression     = green  -> blue (high -> low lightness).
          The two ramps are chosen on the SAME CIELAB lightness sequence
          (L* = 98 / 84 / 69 / 54 / 39), so brightness means the same thing in
          both blocks; only hue changes. One horizontal colourbar per block at
          the bottom, both spanning 0 .. vmax.
Cell    : SE of the mean = s/sqrt(n), s = std across the 10 repeats, averaged over facets (classification: 4 OVR
          facets; regression: 9 quantile buckets). Units: x 10^-3.

Also writes
    se_by_layer.csv   long format: task, model, layer, condition, se
    se_summary.csv    aggregated over conditions: mean / min / max per cell

Usage:
    python -m src.eval.se_heatmap
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from ..common.config import FIGURE_ROOT, RESULT_ROOT, robust_se_pkl

HERE = Path(__file__).resolve().parent
OUT_DIR = RESULT_ROOT / "standard_error"
BASE = HERE.parents[1]

CONDS = ["attack", "gauss", "salt", "move", "occ"]
COND_TITLE = {"attack": "PGD", "gauss": "Gauss", "salt": "Salt", "move": "Move", "occ": "Occ"}
TASKS = [("classification", "Classification"), ("regression", "Regression")]
MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
MODEL_LABEL = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
N_HIDDEN = 10            # VGGNet / ViT / MLP-Mixer; ResNet has 9
N_BLOCK = N_HIDDEN + 1   # hidden layers + output
N_ROW = len(MODELS)

# Matched CIELAB lightness (L* = 98.4 / 84.5 / 69.7 / 53.8 / 39.5 vs 96.4 / 83.6 /
# 68.4 / 55.1 / 39.0): warm goes yellow->red, cool goes green->blue, same brightness law.
PALETTE = {
    "classification": ["#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"],
    "regression": ["#EDF8E9", "#A8DDB5", "#41B6C4", "#2B8CBE", "#175E9E"],
}

CELL_PT = 7.0
TICK_PT = 7.0
LABEL_PT = 8.0


def load_se(base, task, model):
    """-> (n_conditions, n_layers+1), SE (= s/sqrt(n-1), s is the ddof=0 std) averaged over facets."""
    path = robust_se_pkl(task, model, "clean")
    with open(str(path), "rb") as f:
        d = pickle.load(f)
    return np.stack([np.asarray(d[c], dtype=np.float64).mean(axis=1) for c in CONDS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = {(t, m): load_se(base, t, m) for t, _ in TASKS for m in MODELS}

    # ---- one (4 x 11) grid per task per condition ---------------------------
    grids = {}
    for task, _ in TASKS:
        for ci, cond in enumerate(CONDS):
            g = np.full((N_ROW, N_BLOCK), np.nan)
            for mi, m in enumerate(MODELS):
                v = raw[(task, m)][ci]          # (n_layers+1,)
                n_hidden = v.shape[0] - 1
                g[mi, :n_hidden] = v[:-1]
                g[mi, N_BLOCK - 1] = v[-1]      # output, adjacent to the last hidden layer
            grids[(task, cond)] = g

    # ---- csv ----------------------------------------------------------------
    with open(str(out / "se_by_layer.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "layer", "condition", "se"])
        for task, _ in TASKS:
            for m in MODELS:
                for ci, cond in enumerate(CONDS):
                    v = raw[(task, m)][ci]
                    for j in range(v.shape[0] - 1):
                        w.writerow([task, m, "L%d" % j, cond, "%.6f" % v[j]])
                    w.writerow([task, m, "out", cond, "%.6f" % v[-1]])

    with open(str(out / "se_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "layer", "se_mean_over_conditions",
                    "se_min_over_conditions", "se_max_over_conditions"])
        for task, _ in TASKS:
            for m in MODELS:
                a = raw[(task, m)]
                for j in range(a.shape[1] - 1):
                    w.writerow([task, m, "L%d" % j, "%.6f" % a[:, j].mean(),
                                "%.6f" % a[:, j].min(), "%.6f" % a[:, j].max()])
                w.writerow([task, m, "out", "%.6f" % a[:, -1].mean(),
                            "%.6f" % a[:, -1].min(), "%.6f" % a[:, -1].max()])

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.size": TICK_PT,
        "axes.titlesize": LABEL_PT,
        "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,      # embed TrueType, not Type 3 (required by most journals)
        "ps.fonttype": 42,
    })

    cmaps = {t: LinearSegmentedColormap.from_list("cmap_" + t, PALETTE[t]) for t, _ in TASKS}

    allvals = np.concatenate([g.ravel() for g in grids.values()]) * 1e3
    vmax = float(np.ceil(np.nanmax(allvals) * 100) / 100)
    vmin = 0.0
    span = vmax - vmin

    fig, axes = plt.subplots(len(CONDS), 2, figsize=(6.5, 7.0),
                             gridspec_kw=dict(left=0.112, right=0.992,
                                              top=0.885, bottom=0.145,
                                              wspace=0.0, hspace=0.17))
    ims = {}
    labels_x = ["L%d" % i for i in range(N_HIDDEN)] + ["Out"]

    for ti, (task, _) in enumerate(TASKS):
        for ci, cond in enumerate(CONDS):
            ax = axes[ci, ti]
            ax.set_facecolor("#e6e6e6")     # shows through NaN (ResNet's missing L9)
            g = grids[(task, cond)] * 1e3
            im = ax.imshow(g, cmap=cmaps[task], vmin=vmin, vmax=vmax, aspect="auto")
            ims[(task, cond)] = im

            ax.set_yticks(range(N_ROW))
            if ti == 0:
                ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
            else:
                ax.set_yticklabels([])
            ax.set_xticks(range(N_BLOCK))
            ax.set_xticklabels(labels_x if ci == len(CONDS) - 1 else [])
            ax.tick_params(axis="both", length=0, pad=1.5)

            for i in range(N_ROW):
                for j in range(N_BLOCK):
                    v = g[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "n/a", ha="center", va="center",
                                fontsize=CELL_PT, color="#555555")
                    else:
                        ax.text(j, i, "%.2f" % v, ha="center", va="center",
                                fontsize=CELL_PT,
                                color="white" if v > vmin + 0.60 * span else "#222222")
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)

    # block headers: centred over each block, just above the "PGD" title
    y_head = axes[0, 0].get_position().y1 + 0.026
    for ti, (task, label) in enumerate(TASKS):
        p = axes[0, ti].get_position()
        fig.text((p.x0 + p.x1) / 2.0, y_head, label,
                 ha="center", va="bottom", fontsize=LABEL_PT, fontweight="bold")

    # corruption name above every panel pair (left aligned, just above the cells)
    for ci, cond in enumerate(CONDS):
        axes[ci, 0].set_title(COND_TITLE[cond], loc="left", fontsize=LABEL_PT - 0.5,
                              fontweight="bold", pad=1.5)

    # bold the "Out" tick label: it is the last column of each block
    for ti in range(len(TASKS)):
        for t in axes[-1, ti].get_xticklabels():
            if t.get_text() == "Out":
                t.set_fontweight("bold")

    # two horizontal colourbars, one under each block; explicit axes so that the
    # gaps to the heatmap and between the two bars stay at ~half a character
    p_last = axes[-1, 0].get_position()
    xL0, seam = axes[-1, 0].get_position().x0, axes[-1, 0].get_position().x1
    xR1 = axes[-1, 1].get_position().x1
    h_cb, gap = 0.011, 0.014          # bar height and gap between the two bars (fig fraction)
    y_cb = p_last.y0 - 0.030 - h_cb   # 0.030 leaves ~half a character below the x labels
    caxL = fig.add_axes([xL0, y_cb, (seam - xL0) - gap / 2.0, h_cb])
    caxR = fig.add_axes([seam + gap / 2.0, y_cb, (xR1 - seam) - gap / 2.0, h_cb])
    for ti, (task, _) in enumerate(TASKS):
        cax = caxL if ti == 0 else caxR
        cb = fig.colorbar(ims[(task, CONDS[-1])], cax=cax, orientation="horizontal")
        cb.set_label(r"SE ($\times 10^{-3}$)", fontsize=LABEL_PT - 0.5, labelpad=2)
        cb.ax.tick_params(labelsize=TICK_PT, length=1.5, pad=1.5)
        cb.outline.set_visible(False)

    outp = FIGURE_ROOT / "generated" / "standard_error" / "se_heatmap.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outp), dpi=400, bbox_inches="tight", pad_inches=0.015)
    outp_pdf = FIGURE_ROOT / "generated" / "standard_error" / "se_heatmap.pdf"
    fig.savefig(str(outp_pdf), format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    print("[ok] wrote", outp)
    print("[ok] wrote", outp_pdf)
    print("[ok] wrote", out / "se_by_layer.csv")
    print("[ok] wrote", out / "se_summary.csv")
    print("range %.4f .. %.4f | vmax %.1f | cells %d" % (
        np.nanmin(allvals) / 1e3, np.nanmax(allvals) / 1e3, vmax, np.isfinite(allvals).sum()))


if __name__ == "__main__":
    main()
