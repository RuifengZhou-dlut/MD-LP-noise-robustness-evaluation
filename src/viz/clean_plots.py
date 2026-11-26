"""Clean-branch figures (fixed 16:9 data area, canvas derived from the margins).

Key points:
  - the **data area** of every subplot is fixed at 7.0 x 3.9375 in (exactly 16:9),
    independent of the font size;
  - canvas size = data area + margins reserved for the labels (in), so larger fonts
    only grow the canvas and never squeeze the data area;
  - fonts are scaled with the canvas width so that a down-scale to the A4 text width
    (6.5 in) still meets the target size (FIG_PT, default 7.5 pt; fonts are never shrunk);
  - x axis = network substructures: **every substructure tick is drawn**, rotated 60°
    and right-aligned to avoid overlap;
  - y axis: all subplots of one figure share a single tick set (same step, same
    bounds) instead of one per panel;
  - the y axis title only appears in the leftmost column; in the 2x2 summary only the
    bottom row shows the x tick labels, the top row stays blank.

Output (figure/generated/clean/<task>/):
  overall_<task>.pdf|png        2x2
  facet_<task>_<k>.pdf|png      1x4
"""
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from ..common.config import FIGURES_ROOT, MODELS, figure_dir, normalize_exp, probe_acc_pkl, robust_pkl

CONDS = ("base", "attack", "gauss", "salt", "move", "occ")
COND_COLORS = {"base": "black", "attack": "orange", "gauss": "red",
               "salt": "blue", "move": "green", "occ": "purple"}
COND_LABELS = {"base": "base", "attack": "$\\gamma_{p}$", "gauss": "$\\gamma_{g}$",
               "salt": "$\\gamma_{s}$", "move": "$\\gamma_{m}$", "occ": "$\\gamma_{o}$"}

MODEL_TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
MODEL_MARKERS = {"VGG-11": "o", "Res-18": "s", "ViT-8": "^", "Mix-8": "D"}

DATA_W, DATA_H = 7.0, 7.0 * 9 / 16      # 16:9 data area
LINE_W = 5.5
MARKER_SIZE = 12
FONT_TICK, FONT_LABEL, FONT_TITLE, FONT_LEGEND = 30, 32, 32, 30
TARGET_PT = float(os.environ.get("FIG_PT", "7.5"))   # final displayed font size (pt)
# A4 text width: 210mm - 2x25mm margins = 160mm = 6.30 in (override with FIG_TEXTWIDTH)
TEXTWIDTH = float(os.environ.get("FIG_TEXTWIDTH", "6.3"))

X_ROT = 60.0            # substructure tick rotation (prevents overlap with all ticks shown)


def _k_for(fig_w):
    """Return the font scale that keeps ticks at TARGET_PT once scaled to TEXTWIDTH."""
    return (TARGET_PT * fig_w / TEXTWIDTH) / FONT_TICK


def _grid(cols, rows, left=2.3, right=0.5, top=1.0, bottom=4.4,
          gapx=2.15, gapy=1.15, dw=DATA_W, dh=DATA_H):
    """Lay out in inches: returns (fig, axes[rows][cols], fig_w). The data area is
    always dw x dh."""
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
    return {"base": base, **rob}


def _xticks(model):
    from .plots import MODEL_XTICK_LABELS
    prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}[model]
    return list(MODEL_XTICK_LABELS[prefix])


def _ylim_shared(matrices):
    """One shared y tick set for all subplots: returns (ylo, yhi, step) with a
    rounded step and bounds snapped to it."""
    vals = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in matrices])
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    step = 0.25
    for s in (0.02, 0.05, 0.1, 0.2, 0.25):
        if (hi - lo) / s <= 5.0:
            step = s
            break
    ylo = math.floor(lo / step) * step
    yhi = math.ceil(hi / step) * step
    if yhi - ylo < 2 * step:              # data is nearly flat: give at least two steps
        yhi = ylo + 2 * step
    return ylo, yhi, step


def _decorate(ax, model, x, labels, ylo, yhi, step, k,
              show_xlabel=True, show_ylabel=True):
    tick, label, title = FONT_TICK * k, FONT_LABEL * k, FONT_TITLE * k
    ax.grid(True, linestyle="--", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick, rotation=X_ROT,
                       ha="right", rotation_mode="anchor")
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=tick, pad=6)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(ylo, yhi)
    ax.set_title(MODEL_TITLES[model], fontsize=title, pad=16)
    if show_ylabel:
        ax.set_ylabel("ACC", fontsize=label, labelpad=10)


def _legend(fig, k, y=0.010):
    handles = [Line2D([0], [0], color=COND_COLORS[c], linewidth=LINE_W,
                      markersize=12, marker="o", label=COND_LABELS[c]) for c in CONDS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=FONT_LEGEND * k,
               frameon=False, bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=2.0)


def _xlabel(fig, bottom_row, k):
    """Place "substructures" below the tick labels of that row (positioned from the
    rendered bounding boxes so no text is ever covered)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y_low = min(t.get_window_extent(renderer).y0
                for ax in bottom_row for t in ax.get_xticklabels())
    h_px = fig.get_size_inches()[1] * fig.dpi
    y = (y_low - 0.12 * fig.dpi) / h_px
    fig.text(0.5, y, "substructures", ha="center", va="top",
             fontsize=FONT_LABEL * k)


def _out_dir(task, exp="clean"):
    d = figure_dir(task, exp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_overall(task, exp="clean", models=None):
    models = models or list(MODELS)
    fig, axes, fig_w = _grid(2, 2)
    k = _k_for(fig_w) * 1.173          # 2x2 enlargement: still >= 7 pt at 0.8 down-scale
    data = {m: _load(exp, task, m) for m in models}

    # one shared y tick set for the whole figure
    ylo, yhi, step = _ylim_shared(
        [data[m][c].mean(axis=1) for m in models for c in CONDS if c in data[m]])

    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            model = models[r * 2 + c]
            d = data[model]
            n = d["base"].shape[0]
            x = np.arange(n)
            for cond in CONDS:
                if cond not in d:
                    continue
                ax.plot(x, d[cond].mean(axis=1), color=COND_COLORS[cond],
                        linewidth=LINE_W, marker=MODEL_MARKERS[model],
                        markersize=MARKER_SIZE)
            _decorate(ax, model, x, _xticks(model)[:n], ylo, yhi, step, k,
                      show_xlabel=(r == len(axes) - 1), show_ylabel=(c == 0))
    _legend(fig, k)
    _xlabel(fig, axes[-1], k)

    out = _out_dir(task, exp) / f"overall_{task}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_facets(task, exp="clean", models=None):
    models = models or list(MODELS)
    data = {m: _load(exp, task, m) for m in models}
    n_facets = data[models[0]]["base"].shape[1]
    outs = []
    for idx in range(n_facets):
        fig, axes, fig_w = _grid(4, 1)
        k = _k_for(fig_w)

        ylo, yhi, step = _ylim_shared(
            [data[m][c][:, idx] for m in models for c in CONDS if c in data[m]])

        for i, (ax, model) in enumerate(zip(axes[0], models)):
            d = data[model]
            n = d["base"].shape[0]
            x = np.arange(n)
            for cond in CONDS:
                if cond not in d:
                    continue
                ax.plot(x, d[cond][:, idx], color=COND_COLORS[cond],
                        linewidth=LINE_W, marker=MODEL_MARKERS[model],
                        markersize=MARKER_SIZE)
            _decorate(ax, model, x, _xticks(model)[:n], ylo, yhi, step, k,
                      show_ylabel=(i == 0))
        _legend(fig, k)
        _xlabel(fig, axes[0], k)

        out = _out_dir(task, exp) / f"facet_{task}_{idx + 1}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        outs.append(out)
    return outs


def run(task, exp="clean", models=None):
    outs = [plot_overall(task, exp, models)] + plot_facets(task, exp, models)
    for o in outs:
        print("[ok]", o)
    return outs






