"""Noise-branch figures (fixed data-area size, canvas derived from the margins).

  - curve subplots use a fixed 7.0 x 3.9375 in data area (16:9); radar data area 7x7 in;
  - canvas = data area + label margins (in), so larger fonts only grow the canvas and
    never squeeze the data area;
  - fonts are scaled with the canvas width so that a down-scale to the A4 text width
    of 6.5 in still meets the target size (FIG_PT, default 7.5 pt);
  - y axis follows the data range; ticks are numbers/letters; legend in one row below.

Layout:
  classification radar: 2x2, 8 axes (4 classes x gauss/salt)
  regression radar    : 2 rows (gauss/salt) x 4 columns, 9 axes each
  substructure metric : 2 rows (gauss/salt) x 4 columns
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib.lines import Line2D

from ..common.config import FIGURES_ROOT, MODELS, figure_dir, normalize_exp, probe_acc_pkl, robust_pkl

COLOR_CLEAN = "red"
COLOR_NOISE = "blue"
LEGEND_CLEAN = "train only with clean samples"
LEGEND_NOISE = "train with noise-imposed samples"

CLASS_NAMES = ("air", "auto", "ship", "truck")
MODEL_TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
MODEL_MARKERS = {"VGG-11": "o", "Res-18": "s", "ViT-8": "^", "Mix-8": "D"}

DATA_W, DATA_H = 7.0, 7.0 * 9 / 16
LINE_W = 5.5
MARKER_SIZE = 12
FONT_TICK, FONT_LABEL, FONT_TITLE, FONT_LEGEND = 30, 32, 32, 30
TARGET_PT = float(os.environ.get("FIG_PT", "7.5"))   # final displayed font size (pt)
FONT_RADAR_LABEL, FONT_RADAR_GRID = 30, 30
TEXTWIDTH = float(os.environ.get("FIG_TEXTWIDTH", "6.3"))


def _k_for(fig_w):
    return (TARGET_PT * fig_w / TEXTWIDTH) / FONT_TICK


def _grid(cols, rows, left=1.9, right=0.4, top=1.15, bottom=2.7,
          gapx=1.35, gapy=1.0, dw=DATA_W, dh=DATA_H):
    fig_w = left + cols * dw + (cols - 1) * gapx + right
    fig_h = top + rows * dh + (rows - 1) * gapy + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for r in range(rows):
        row = []
        for c in range(cols):
            x = (left + c * (dw + gapx)) / fig_w
            top_off = top + r * (dh + gapy)
            y = (fig_h - top_off - dh) / fig_h
            row.append(fig.add_axes([x, y, dw / fig_w, dh / fig_h]))
        axes.append(row)
    return fig, axes, fig_w


def _load(exp, task, model):
    with open(probe_acc_pkl(task, model, exp), "rb") as f:
        base = np.asarray(pickle.load(f)["base"], dtype=np.float64)
    with open(robust_pkl(task, model, exp), "rb") as f:
        rob = {c: np.asarray(v, dtype=np.float64) for c, v in pickle.load(f).items()}
    return base, rob


def _xticks(model, n):
    from .plots import MODEL_XTICK_LABELS
    prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}[model]
    return list(MODEL_XTICK_LABELS[prefix])[:n]


def _tick_plan(labels, font):
    def width(seq):
        return len(seq) * max(len(s) for s in seq) * 0.17 * (font / 36.0)

    if len(labels) <= 8 and width(labels) < DATA_W * 0.65:
        return list(range(len(labels))), labels, 0
    n = len(labels)
    idx = list(range(0, n, 2))
    if idx[-1] != n - 1:
        if n - 1 - idx[-1] == 1:
            idx.pop()
        idx.append(n - 1)
    if width([labels[i] for i in idx]) < DATA_W * 0.90:
        return idx, [labels[i] for i in idx], 0
    if width(labels[::3]) < DATA_W * 0.90:
        return list(range(0, len(labels), 3)), labels[::3], 0
    return list(range(len(labels))), labels, 45


def _legend(fig, k, y=0.012):
    handles = [
        Line2D([0], [0], color=COLOR_CLEAN, linewidth=LINE_W, markersize=12,
               marker="o", label=LEGEND_CLEAN),
        Line2D([0], [0], color=COLOR_NOISE, linewidth=LINE_W, markersize=12,
               marker="s", label=LEGEND_NOISE),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FONT_LEGEND * k,
               frameon=False, bbox_to_anchor=(0.5, y), handlelength=2.4, columnspacing=3.0)


def draw_polygon_radar(ax, labels, values, colors, marker="o",
                       linewidth=4.0, markersize=10,
                       label_fontsize=FONT_RADAR_LABEL, title=None,
                       title_fontsize=FONT_TITLE):
    values = np.asarray(values, dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.pi / 2 - angles
    ax.set_aspect("equal")
    ax.axis("off")

    for g in np.linspace(0, 1, 5):
        xs, ys = g * np.cos(angles), g * np.sin(angles)
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color="gray", linewidth=1.0, alpha=0.5)
    for angle in angles:
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color="gray", linewidth=1.0, alpha=0.5)

    for vals, color in zip(values, colors):
        xs, ys = vals * np.cos(angles), vals * np.sin(angles)
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=color, linewidth=linewidth,
                marker=marker, markersize=markersize, markevery=1,
                markerfacecolor=color, markeredgecolor=color)
        ax.fill(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=color, alpha=0.15)

    for label, angle in zip(labels, angles):
        cx, cy = np.cos(angle), np.sin(angle)
        ha = "left" if cx > 0.15 else ("right" if cx < -0.15 else "center")
        ax.text(1.17 * cx, 1.17 * cy, label, ha=ha, va="center", fontsize=label_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=18)
    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.48, 1.48)


def _out_dir(task, exp="noise"):
    d = figure_dir(task, exp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_radar(task, exp="noise", models=None):
    """classification: 1 row x 4 columns, 8 axes per panel (4 classes x gauss/salt);
    regression: 2 rows (gauss/salt) x 4 columns, 9 axes per panel (9 threshold splits).
    """
    models = models or list(MODELS)

    if task == "classification":
        fig, axes, fig_w = _grid(4, 1, left=0.4, right=0.4, top=1.75, bottom=2.0,
                                 gapx=3.3, dw=7.0, dh=6.3)
        k = _k_for(fig_w)
        labels = ([f"{c},g" for c in CLASS_NAMES] +
                  [f"{c},s" for c in CLASS_NAMES])
        for ax, model in zip(axes[0], models):
            _, rc = _load("clean", task, model)
            _, rn = _load(exp, task, model)
            draw_polygon_radar(ax, labels,
                               [np.r_[rc["gauss"][-1], rc["salt"][-1]],
                                np.r_[rn["gauss"][-1], rn["salt"][-1]]],
                               colors=[COLOR_CLEAN, COLOR_NOISE],
                               marker=MODEL_MARKERS[model],
                               label_fontsize=FONT_RADAR_LABEL * k,
                               title=MODEL_TITLES[model],
                               title_fontsize=FONT_TITLE * k)
    else:
        fig, axes, fig_w = _grid(4, 2, left=0.4, right=0.4, top=1.75, bottom=2.0,
                                 gapx=3.3, gapy=1.2, dw=7.0, dh=6.3)
        k = _k_for(fig_w)
        for r, cond in enumerate(("gauss", "salt")):
            labels = [rf"$ACC_{{{i}}}$" for i in range(1, 10)]
            for c, model in enumerate(models):
                _, rc = _load("clean", task, model)
                _, rn = _load(exp, task, model)
                draw_polygon_radar(axes[r][c], labels,
                                   [rc[cond][-1], rn[cond][-1]],
                                   colors=[COLOR_CLEAN, COLOR_NOISE],
                                   marker=MODEL_MARKERS[model],
                                   label_fontsize=FONT_RADAR_LABEL * k,
                                   title=f"{MODEL_TITLES[model]} — {cond}",
                                   title_fontsize=FONT_TITLE * k)
    _legend(fig, k, y=0.012)
    sub = ("ACC per class (g = gauss, s = salt)" if task == "classification"
           else "ACC per threshold split (rows: gauss / salt)")
    fig.text(0.5, 1.0 - 0.42 / fig.get_size_inches()[1], sub, ha="center",
             fontsize=FONT_TITLE * k * 0.9)

    out = _out_dir(task, exp) / f"radar_{task}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_noise_metric(task, exp="noise", models=None, conds=("gauss", "salt")):
    models = models or list(MODELS)
    fig, axes, fig_w = _grid(4, len(conds), gapy=1.45)
    k = _k_for(fig_w)
    tick, label, title = FONT_TICK * k, FONT_LABEL * k, FONT_TITLE * k

    for r, cond in enumerate(conds):
        for c, model in enumerate(models):
            ax = axes[r][c]
            _, rc = _load("clean", task, model)
            _, rn = _load(exp, task, model)
            y_c = rc[cond].mean(axis=1)
            y_n = rn[cond].mean(axis=1)
            n = min(len(y_c), len(y_n))
            x = np.arange(n)
            labels = _xticks(model, n)
            mk = MODEL_MARKERS[model]

            ax.plot(x, y_c[:n], color=COLOR_CLEAN, linewidth=LINE_W, marker=mk,
                    markersize=MARKER_SIZE)
            ax.plot(x, y_n[:n], color=COLOR_NOISE, linewidth=LINE_W, marker=mk,
                    markersize=MARKER_SIZE)

            vals = np.r_[y_c[:n], y_n[:n]]
            ax.set_ylim(max(0.0, vals.min() - 0.05), min(1.02, vals.max() + 0.05))
            pos, texts, rot = _tick_plan(labels, tick)
            ax.grid(True, linestyle="--", linewidth=2.0)
            ax.set_xticks(x[pos])
            ax.set_xticklabels(texts, fontsize=tick, rotation=rot,
                               ha="right" if rot else "center")
            ax.tick_params(axis="y", labelsize=tick)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_title(f"{MODEL_TITLES[model]} — {cond}", fontsize=title, pad=14)
            if c == 0:
                ax.set_ylabel("ACC", fontsize=label)

    _legend(fig, k)
    fig.text(0.5, 1.55 / fig.get_size_inches()[1], "substructures", ha="center", fontsize=FONT_LABEL * k)

    out = _out_dir(task, exp) / f"noise_metric_{task}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def run(task, exp="noise", models=None):
    outs = [plot_radar(task, exp, models), plot_noise_metric(task, exp, models)]
    for o in outs:
        print("[ok]", o)
    return outs

























