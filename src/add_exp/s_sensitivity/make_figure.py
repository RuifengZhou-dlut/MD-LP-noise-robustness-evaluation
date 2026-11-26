#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 5.4.3, main-experiment setting: merge s=0.2 (main-experiment result) with
s=0.4/0.6/0.8 (recomputed by replicating the selection rule) and plot.

s=0.2 is the last row of result/criteria/clean/classification/<Model>_robustness.pkl
(whole-network OVR); the other s values come from
result/add_exp/s_sensitivity/noise_s_rank_maincal.json.

Outputs:
  result/add_exp/s_sensitivity/noise_s_rank_maincal_full.json
  figure/generated/clean/classification/s_scan_classification_maincal.pdf|png
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ...viz.clean_plots import (FONT_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE,
                                      LINE_W, MARKER_SIZE, _grid, _k_for)

from ..common.sites import figure_dir, result_dir
from ...common.config import task_dir

CAL = result_dir("add_exp/s_sensitivity")
SRC = CAL / "noise_s_rank_maincal.json"
CRIT = task_dir("classification", "clean")
FULL = CAL / "noise_s_rank_maincal_full.json"
OUT = figure_dir("add_exp/s_sensitivity") / "s_scan_classification_maincal"

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
COLORS = {"VGG-11": "#1f77b4", "Res-18": "#ff7f0e", "ViT-8": "#2ca02c", "Mix-8": "#d62728"}
MARKERS = {"VGG-11": "o", "Res-18": "s", "ViT-8": "^", "Mix-8": "D"}

NOISES = [("attack", r"$\gamma_p$ (PGD)"), ("gauss", r"$\gamma_g$ (Gaussian)"),
          ("salt", r"$\gamma_s$ (salt-pepper)"), ("move", r"$\gamma_m$ (motion-blur)"),
          ("occ", r"$\gamma_o$ (occlusion)")]
S_LIST = ["0.2", "0.4", "0.6", "0.8"]


def build_full():
    d = json.load(open(str(SRC), encoding="utf-8"))
    res = {c: {s: {} for s in S_LIST} for c, _ in NOISES}
    for cond, _ in NOISES:
        for m in MODELS:
            rb = pickle.load(open(str(CRIT / f"{m}_robustness.pkl"), "rb"))
            res[cond]["0.2"][m] = {"mean": float(np.asarray(rb[cond])[-1].mean()), "std": None}
            for s in ["0.4", "0.6", "0.8"]:
                res[cond][s][m] = d["results"][cond][s][m]
    full = {"s_list": S_LIST, "results": res,
            "note": "s=0.2 from criteria robustness (10 repeats); s>=0.4 replicated main selection"}
    FULL.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[ok]", FULL)
    return full


def plot(full):
    xs = [float(s) for s in S_LIST]
    fig, axes, fig_w = _grid(3, 2, gapx=1.7, gapy=1.5)
    k = _k_for(fig_w)
    flat = [ax for row in axes for ax in row]

    for ax, (noise, label) in zip(flat, NOISES):
        vals = []
        for m in MODELS:
            ys = [full["results"][noise][s][m]["mean"] for s in S_LIST]
            vals.append(ys)
            ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], markersize=MARKER_SIZE,
                    linewidth=LINE_W)
        v = np.concatenate([np.asarray(a) for a in vals])
        pad = 0.4 * (v.max() - v.min() + 1e-9)
        ax.set_ylim(max(0.0, v.min() - pad), min(1.02, v.max() + pad))
        ax.set_xlim(0.12, 0.88)
        ax.set_xticks(xs)
        ax.set_xticklabels(S_LIST, fontsize=FONT_TICK * k)
        ax.tick_params(axis="y", labelsize=FONT_TICK * k)
        span = v.max() - v.min()
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f" if span < 0.01 else "%.2f"))
        ax.grid(True, linestyle="--", linewidth=2.0)
        ax.set_title(label, fontsize=FONT_TITLE * k, pad=14)
        ax.set_ylabel("ACC", fontsize=FONT_LABEL * k)

    shift = 0.5 * (7.0 + 1.7) / fig_w
    for ax in axes[1][:2]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
    axes[1][2].axis("off")

    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], markersize=12,
                      linewidth=LINE_W, label=TITLES[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=FONT_LEGEND * k,
               frameon=False, bbox_to_anchor=(0.5, 0.012), handlelength=2.2, columnspacing=2.0)
    fig.text(0.5, 1.55 / fig.get_size_inches()[1], "selection ratio $s$",
             ha="center", fontsize=FONT_LABEL * k)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    fig.savefig(str(OUT.with_suffix(".png")), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    plot(build_full())
