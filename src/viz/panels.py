"""Paper figure panels (main figures: overall 2x2 / class 4x4 / threshold 5x4+4x4 /
radar / legend).

Uniform standard
------------------------
* Curve line width **1.1 pt**, marker **2.4 pt**, grid 0.5 pt dashed — identical for
  every panel, so line widths and fonts stay exactly consistent once pasted 1:1 into
  the paper.
* **Every panel whose x axis is the hidden-layer substructure draws the x axis**
  (1-E…, 2-T…, different per model; the four cells of the 2x2 main figure have
  different x axes, so each cell needs its own).
* The 2x2 uses 4-5 y grid lines; the 4/5-row facet figures are sparser, only 3-4.
* 16:9 data area: 2.499x1.406 for the 2x2, 1.135x0.639 for facets (4 columns = 6.28 in).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ..common.config import MODELS
from .clean_plots import COND_COLORS, COND_LABELS, MODEL_MARKERS, _load
from .noise_plots import COLOR_CLEAN, COLOR_NOISE, draw_polygon_radar
from .plots import MODEL_XTICK_LABELS

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODEL_KEYS = list(MODELS)
MODEL_TITLE = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT",
               "Mix-8": "Mixer"}
CLASS_TITLE = ("air", "auto", "ship", "truck")
X_ROT = 60.0

# ---- paper-wide line width / marker / grid
LW, MS, GRID_LW = 1.1, 2.4, 0.5

OVERALL = dict(dw=2.450, dh=2.450 * 9 / 16, left=0.55, right=0.12,
               top=0.04, bot=0.62, tick=8.8, label=9.4)
FACET = dict(dw=1.140, dh=1.140 * 9 / 16, left=0.34, right=0.12,
             top=0.04, bot=0.52, tick=7.5, label=8.0)


def _xticks(model):
    prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}[model]
    return list(MODEL_XTICK_LABELS[prefix])


def _ylim_step(mats, max_ticks=5):
    v = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in mats])
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    step = 0.25
    for s in (0.02, 0.05, 0.1, 0.2, 0.25):
        if (hi - lo) / s <= max_ticks - 1:
            step = s
            break
    ylo = np.floor(lo / step) * step
    return ylo, max(np.ceil(hi / step) * step, ylo + 2 * step), step



def _draw(ax, series, model, ylo, yhi, step, g, show_x=True, show_y=True):
    n = len(next(iter(series.values())))
    x = np.arange(n)
    mk = MODEL_MARKERS[model]
    for cond, y in series.items():
        ax.plot(x, y, color=COND_COLORS[cond], linewidth=LW, marker=mk,
                markersize=MS, markeredgewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", linewidth=GRID_LW, alpha=0.9)
    ax.tick_params(labelsize=g["tick"], length=1.8, pad=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_xticks(model)[:n], rotation=X_ROT, ha="right",
                       rotation_mode="anchor", fontsize=g["tick"])
    if show_x:
        ax.set_xlabel("substructures", fontsize=g["label"], labelpad=1.5)
    if show_y:
        ax.set_ylabel("ACC", fontsize=g["label"], labelpad=1.5)



def measure(series, model, g, ylo, yhi, step, show_x=True, show_y=True):
    """Measure the margins (inches) that all text of this cell actually occupies
    (ticks + axis titles)."""
    pad = 1.4
    w, h = g["dw"] + pad, g["dh"] + pad
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((pad / 2 / w, pad / 2 / h, g["dw"] / w, g["dh"] / h))
    _draw(ax, series, model, ylo, yhi, step, g, show_x, show_y)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return dict(left=pad / 2 - bb.x0, bottom=pad / 2 - bb.y0,
                right=bb.x1 - (pad / 2 + g["dw"]),
                top=bb.y1 - (pad / 2 + g["dh"]))


def curve_panel(series, model, g, ylo, yhi, step, out, margins,
                show_x=True, show_y=True):
    left, bottom = margins["left"], margins["bottom"]
    right, top = margins["right"], margins["top"]
    w = left + g["dw"] + right
    h = bottom + g["dh"] + top
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((left / w, bottom / h, g["dw"] / w, g["dh"] / h))
    _draw(ax, series, model, ylo, yhi, step, g, show_x, show_y)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return w, h


def legend_conditions(out, g):
    fig = plt.figure(figsize=(3.10, 0.26))
    handles = [Line2D([0], [0], color=COND_COLORS[c], linewidth=LW,
                      marker="o", markersize=MS, label=COND_LABELS[c])
               for c in COND_LABELS]
    fig.legend(handles=handles, loc="center", ncol=6, fontsize=8.0,
               frameon=False, handlelength=2.0, columnspacing=1.8,
               handletextpad=0.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def legend_training(out):
    fig = plt.figure(figsize=(2.4, 0.26))
    handles = [Line2D([0], [0], color=COLOR_CLEAN, linewidth=LW, marker="o",
                      markersize=MS, label="train on clean samples"),
               Line2D([0], [0], color=COLOR_NOISE, linewidth=LW, marker="s",
                      markersize=MS, label="train on noisy samples")]
    fig.legend(handles=handles, loc="center", ncol=2, fontsize=8.0,
               frameon=False, handlelength=2.0, columnspacing=2.0,
               handletextpad=0.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def radar_panel(task, model, out, cond=None):
    side = 1.15
    fig = plt.figure(figsize=(side, side))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    rc = _load("clean", task, model)
    rn = _load("noise", task, model)
    if task == "classification":
        labels = [f"{c},g" for c in CLASS_TITLE] + [f"{c},s" for c in CLASS_TITLE]
        vals = [np.r_[rc["gauss"][-1], rc["salt"][-1]],
                np.r_[rn["gauss"][-1], rn["salt"][-1]]]
    else:
        labels = [rf"$ACC_{{{i}}}$" for i in range(1, 10)]
        vals = [rc[cond][-1], rn[cond][-1]]
    draw_polygon_radar(ax, labels, vals, colors=[COLOR_CLEAN, COLOR_NOISE],
                       marker=MODEL_MARKERS[model], linewidth=0.6, markersize=1.4,
                       label_fontsize=6.6, title=None)
    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.48, 1.48)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return side, side


def _margins(cells, g, show_x=True, show_y=True):
    """cells: [(series, model, ylo, yhi, step)] -> margins shared within the group
    (maximum over the cells)."""
    m = {"left": 0.0, "bottom": 0.0, "right": 0.0, "top": 0.0}
    for series, model, ylo, yhi, step in cells:
        d = measure(series, model, g, ylo, yhi, step, show_x, show_y)
        for k in m:
            m[k] = max(m[k], d[k])
    return m


def build(task, out_root, models=None):
    models = models or MODEL_KEYS
    out_root = Path(out_root)
    rows = [["file", "task", "group", "row", "col", "n_rows", "n_cols", "model",
             "title", "width_in", "height_in"]]
    data = {m: _load("clean", task, m) for m in models}

    def emit(group, row, col, n_rows, n_cols, model, title, p, w, h):
        rows.append([str(p.relative_to(out_root)), task, group, row, col, n_rows,
                     n_cols, model, title, f"{w:.3f}", f"{h:.3f}"])

    # ---- overall 2x2
    g = OVERALL
    d = out_root / task / "overall"
    d.mkdir(parents=True, exist_ok=True)
    ser = {m: {c: data[m][c].mean(axis=1) for c in data[m]} for m in models}
    ylo, yhi, step = _ylim_step([v for m in models for v in ser[m].values()], 5)
    cells = [(ser[m], m, ylo, yhi, step) for m in models]
    mg = _margins(cells, g)
    for i, m in enumerate(models):
        r, c = divmod(i, 2)
        t = f"{MODEL_TITLE[m]} reg" if task == "regression" else MODEL_TITLE[m]
        p = d / f"{m}.pdf"
        w, h = curve_panel(ser[m], m, g, ylo, yhi, step, p, mg)
        emit("overall", r, c, 2, 2, m, t, p, w, h)

    # ---- facets (one shared margin set per group -> every facet has the same size)
    g = FACET
    n_f = data[models[0]]["base"].shape[1]
    groups = []
    if task == "classification":
        for k in range(n_f):
            ylo, yhi, step = _ylim_step(
                [data[m][c][:, k] for m in models for c in data[m]], 4)
            cells = [({c: data[m][c][:, k] for c in data[m]}, m, ylo, yhi, step)
                     for m in models]
            groups.append((f"class/c{k + 1}", k, 4, n_f,
                           [(f"{MODEL_TITLE[m]} ({CLASS_TITLE[k]})", m, cells[j])
                            for j, m in enumerate(models)]))
    else:
        for gname, idxs, nrows in (("thrA", range(0, 5), 5),
                                   ("thrB", range(5, n_f), 4)):
            for r, k in enumerate(list(idxs)):
                ylo, yhi, step = _ylim_step(
                    [data[m][c][:, k] for m in models for c in data[m]], 4)
                cells = [({c: data[m][c][:, k] for c in data[m]}, m, ylo, yhi, step)
                         for m in models]
                groups.append((f"{gname}/t{k + 1}", r, nrows, 4,
                               [(rf"{MODEL_TITLE[m]} ($\tau_{{{k + 1}}}$)", m,
                                 cells[j]) for j, m in enumerate(models)]))
    all_cells = [c for _, _, _, _, its in groups for _, _, c in its]
    m_in = _margins(all_cells, g, False, False)          # ticks only
    m_y = _margins(all_cells, g, False, True)            # + ACC on the right
    m_x = _margins(all_cells, g, True, False)            # + substructures on the last row
    m_xy = _margins(all_cells, g, True, True)            # + both
    for gname, r, nrows, ncols, its in groups:
        dn = out_root / task / gname
        dn.mkdir(parents=True, exist_ok=True)
        for j, (title, m, cell) in enumerate(its):
            show_x = (r == nrows - 1)
            show_y = (j == 0)
            ref = m_xy if (show_x and show_y) else (
                m_x if show_x else (m_y if show_y else m_in))
            mg = dict(left=(m_y["left"] if show_y else m_in["left"]),
                      top=m_in["top"], right=ref["right"],
                      bottom=ref["bottom"])
            p = dn / f"{m}.pdf"
            w, h = curve_panel(cell[0], cell[1], g, cell[2], cell[3],
                               cell[4], p, mg, show_x, show_y)
            emit(gname, r, j, nrows, 4, m, title, p, w, h)

    # ---- radar
    d = out_root / task / "radar"
    d.mkdir(parents=True, exist_ok=True)
    for m in models:
        if task == "classification":
            p = d / f"{m}.pdf"
            w, h = radar_panel(task, m, p)
            emit("radar", 0, models.index(m), 1, 4, m, MODEL_TITLE[m], p, w, h)
        else:
            for cond in ("gauss", "salt"):
                p = d / f"{m}_{cond}.pdf"
                w, h = radar_panel(task, m, p, cond=cond)
                emit(f"radar/{cond}", 0 if cond == "gauss" else 1,
                     models.index(m), 2, 4, m, f"{MODEL_TITLE[m]} ({cond})", p, w, h)

    d = out_root / "legend"
    d.mkdir(parents=True, exist_ok=True)
    legend_conditions(d / "conditions.pdf", FACET)
    legend_training(d / "training.pdf")

    with open(out_root / f"manifest_{task}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[ok] {len(rows) - 1} panels -> {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    build(args.task, args.out, args.models)


if __name__ == "__main__":
    main()
