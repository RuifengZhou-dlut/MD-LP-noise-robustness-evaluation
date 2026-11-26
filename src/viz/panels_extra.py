"""Panels for two additional paper figures: noise_metric (2x4 per task) and
overall_classification_full (1x2).

- noise_metric: rows = gauss / salt, columns = 4 networks; the two lines are
  clean-trained / noise-trained.
- classification_full: left = whole-network accuracy per condition (grouped bars),
  right = std over repeated runs.

Geometry matches the main figures (data area 1.155x0.650, 4 columns = 6.28 in; every
row has a y axis, x only on the last row).
Output: figure/generated/panels/extra/... + manifest_extra.csv
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from ..common.config import MODELS
from .clean_plots import MODEL_MARKERS, _load
from .noise_plots import COLOR_CLEAN, COLOR_NOISE
from .plots import MODEL_XTICK_LABELS

from ..common.config import CRITERIA_ROOT, FIGURE_ROOT

OUT = FIGURE_ROOT / "generated" / "panels" / "extra"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODEL_KEYS = list(MODELS)
TITLE = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mixer"}
COND_T = {"attack": "PGD", "gauss": "Gauss", "salt": "Salt", "move": "Move",
          "occ": "Occ"}
COL = {"VGG-11": "#1f77b4", "Res-18": "#ff7f0e", "ViT-8": "#2ca02c",
       "Mix-8": "#d62728"}

DW, DH = 1.125, 1.125 * 9 / 16
LEFT, RIGHT = 0.26, 0.12
TICK, LABEL = 7.5, 8.0
X_ROT = 60.0


def _ticks(model):
    prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}[model]
    return list(MODEL_XTICK_LABELS[prefix])


def noise_metric(task, rows):
    for r, cond in enumerate(("gauss", "salt")):
        show_x = r == 1                     # ticks in every cell; title only on the last row
        bottom = 0.52 if show_x else 0.34    # the last row carries substructures
        for c, m in enumerate(MODEL_KEYS):
            rc = _load("clean", task, m)
            rn = _load("noise", task, m)
            y_c = rc[cond].mean(axis=1)
            y_n = rn[cond].mean(axis=1)
            n = min(len(y_c), len(y_n))
            x = np.arange(n)
            w, h = LEFT + DW + RIGHT, bottom + DH + 0.06
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_axes((LEFT / w, bottom / h, DW / w, DH / h))
            mk = MODEL_MARKERS[m]
            ax.plot(x, y_c[:n], color=COLOR_CLEAN, lw=1.1, marker=mk, ms=2.4,
                    markeredgewidth=0)
            ax.plot(x, y_n[:n], color=COLOR_NOISE, lw=1.1, marker=mk, ms=2.4,
                    markeredgewidth=0)
            vals = np.r_[y_c[:n], y_n[:n]]
            ax.set_ylim(max(0.0, vals.min() - 0.05), min(1.02, vals.max() + 0.05))
            ax.set_xlim(-0.5, n - 0.5)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(True, ls="--", lw=0.5, alpha=0.9)
            ax.tick_params(labelsize=TICK, length=1.5, pad=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(_ticks(m)[:n], rotation=X_ROT, ha="right",
                               rotation_mode="anchor", fontsize=TICK)
            if show_x:
                ax.set_xlabel("substructures", fontsize=LABEL, labelpad=1.5)
            if c == 0:
                ax.set_ylabel("ACC", fontsize=LABEL, labelpad=1.5)
            p = OUT / f"noise_metric_{task}_{cond}_{m}.pdf"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            rows.append([str(p.relative_to(OUT.parent)), task,
                         f"noise_metric_{task}", r, c,
                         f"{TITLE[m]} ({COND_T[cond]})", f"{w:.3f}", f"{h:.3f}",
                         int(show_x)])


def legend_pair(out):
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(2.6, 0.24))
    h = [Line2D([0], [0], color=COLOR_CLEAN, lw=1.2, marker="o", ms=3.2,
                label="train on clean samples"),
         Line2D([0], [0], color=COLOR_NOISE, lw=1.2, marker="s", ms=3.2,
                label="train on noisy samples")]
    fig.legend(handles=h, loc="center", ncol=2, fontsize=LABEL, frameon=False,
               handlelength=2.0, columnspacing=1.6, handletextpad=0.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def full_cifar(rows):
    import csv as _csv
    path = CRITERIA_ROOT / "clean" / "classification_full.csv"
    data = {}
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            data[(r["model"], r["condition"])] = (
                float(r["final"]),
                None if r["std_final"] == "" else float(r["std_final"]))
    models = [m for m in MODEL_KEYS if (m, "base") in data]
    conds = ["attack", "gauss", "salt", "move", "occ"]
    x = np.arange(len(conds))
    n = len(models)
    width = 0.8 / max(n, 1)

    specs = [("acc", "whole-network OVR accuracy", (0.30, 0.85), 1.0),
             ("std", r"std ($\times 10^{-3}$)", None, 1e3)]
    for col, (tag, ylab, ylim, scale) in enumerate(specs):
        w = LEFT + 2.62 + RIGHT
        h = 0.62 + 1.15 + 0.06
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_axes((LEFT / w, 0.62 / h, 2.62 / w, 1.15 / h))
        for k, m in enumerate(models):
            vals = []
            for c in conds:
                v, sd = data[(m, c)]
                vals.append(v * scale if tag == "acc" else (sd or 0.0) * scale)
            ax.bar(x + (k - (n - 1) / 2) * width, vals, width * 0.92,
                   color=COL[m], edgecolor="white", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_T[c] for c in conds], fontsize=TICK)
        if ylim:
            ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.85)
        ax.tick_params(labelsize=TICK, length=1.5, pad=1.5)
        ax.set_ylabel(ylab, fontsize=LABEL, labelpad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        p = OUT / f"classification_full_{tag}.pdf"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        rows.append([str(p.relative_to(OUT.parent)), "classification_full",
                     "classification_full", 0, col,
                     "robustness by condition" if tag == "acc"
                     else "repeat-to-repeat std", f"{w:.3f}", f"{h:.3f}", 1])

    # legend strip for the four networks
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(3.0, 0.24))
    h = [Line2D([0], [0], color=COL[m], lw=4.0, label=TITLE[m]) for m in models]
    fig.legend(handles=h, loc="center", ncol=len(models), fontsize=LABEL,
               frameon=False, handlelength=1.6, columnspacing=1.6,
               handletextpad=0.5)
    fig.savefig(OUT / "legend_models.pdf", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["file", "task", "group", "row", "col", "title", "width_in",
             "height_in", "x_labels"]]
    for task in ("classification", "regression"):
        noise_metric(task, rows)
    legend_pair(OUT / "legend_pair.pdf")
    full_cifar(rows)
    with open(OUT / "manifest_extra.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[ok] {len(rows) - 1} extra panels -> {OUT}")


if __name__ == "__main__":
    main()
