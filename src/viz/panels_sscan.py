"""s-scan panels (paper F6 formulation): one cell per noise type, 4 model lines per
cell.

Data: result/add_exp/s_sensitivity/noise_s_rank_maincal_full.json
      (s=0.2 takes the last row of the criteria robustness results; s=0.4/0.6/0.8 come
      from the replicated sample selection)
Layout: 3 on top + 2 below (arranged on the LaTeX side), panel size
0.42 + 1.66 + 0.0 = 2.08 in, 3 columns = 6.24 in.
Output: figure/generated/add_exp/s_sensitivity/panels/*.pdf + manifest_sscan.csv
"""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from ..common.config import CRITERIA_ROOT, add_figure_dir, add_result_dir

CAL = add_result_dir("add_exp/s_sensitivity")
FULL = CAL / "noise_s_rank_maincal_full.json"
SRC = CAL / "noise_s_rank_maincal.json"
CRIT = CRITERIA_ROOT / "clean" / "classification"
OUT = add_figure_dir("add_exp/s_sensitivity") / "panels"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT",
          "Mix-8": "MLP-Mixer"}
COLORS = {"VGG-11": "#1f77b4", "Res-18": "#ff7f0e", "ViT-8": "#2ca02c",
          "Mix-8": "#d62728"}
MARKERS = {"VGG-11": "o", "Res-18": "s", "ViT-8": "^", "Mix-8": "D"}
NOISES = [("attack", r"$\gamma_p$ (PGD)"), ("gauss", r"$\gamma_g$ (Gaussian)"),
          ("salt", r"$\gamma_s$ (salt-pepper)"),
          ("move", r"$\gamma_m$ (motion-blur)"),
          ("occ", r"$\gamma_o$ (occlusion)")]
S_LIST = ["0.2", "0.4", "0.6", "0.8"]

DW, DH = 1.66, 1.66 * 9 / 16
LEFT, RIGHT = 0.42, 0.015
TICK, LABEL = 7.5, 8.0


def build_full():
    if FULL.exists():
        return json.load(open(str(FULL), encoding="utf-8"))
    d = json.load(open(str(SRC), encoding="utf-8"))
    res = {c: {s: {} for s in S_LIST} for c, _ in NOISES}
    for cond, _ in NOISES:
        for m in MODELS:
            rb = pickle.load(open(str(CRIT / f"{m}_robustness.pkl"), "rb"))
            res[cond]["0.2"][m] = {"mean": float(np.asarray(rb[cond])[-1].mean()),
                                   "std": None}
            for s in ["0.4", "0.6", "0.8"]:
                res[cond][s][m] = d["results"][cond][s][m]
    full = {"s_list": S_LIST, "results": res}
    FULL.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    return full


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = build_full()
    xs = [float(s) for s in S_LIST]
    rows = [["file", "task", "group", "row", "col", "title", "width_in",
             "height_in", "x_labels"]]
    for i, (noise, label) in enumerate(NOISES):
        show_x = i >= 3                     # 3 top + 2 bottom: only the lower two get an x label
        bottom = 0.46 if show_x else 0.06
        vals = []
        fig_w, fig_h = LEFT + DW + RIGHT, bottom + DH + 0.06
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes((LEFT / fig_w, bottom / fig_h, DW / fig_w, DH / fig_h))
        for m in MODELS:
            ys = [full["results"][noise][s][m]["mean"] for s in S_LIST]
            vals.append(ys)
            ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], ms=2.6, lw=1.0,
                    markeredgewidth=0)
        v = np.concatenate([np.asarray(a) for a in vals])
        pad = 0.4 * (v.max() - v.min() + 1e-9)
        ax.set_ylim(max(0.0, v.min() - pad), min(1.02, v.max() + pad))
        ax.set_xlim(0.12, 0.88)
        ax.set_xticks(xs)
        ax.set_xticklabels(S_LIST, fontsize=TICK)
        span = v.max() - v.min()
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if span < 0.01
                                                        else "%.2f"))
        ax.grid(True, ls="--", lw=0.5, alpha=0.9)
        ax.tick_params(labelsize=TICK, length=1.5, pad=1.5)
        ax.set_xlabel("selection ratio $s$" if show_x else "", fontsize=LABEL,
                      labelpad=1.5)
        ax.set_ylabel("ACC", fontsize=LABEL, labelpad=1.5)
        p = OUT / f"s_scan_{noise}.pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        rows.append([str(p.relative_to(OUT.parent)), "s_scan", "s_scan",
                     i // 3, i % 3, label, f"{fig_w:.3f}", f"{fig_h:.3f}",
                     int(show_x)])

    # legend strip for the 4 models
    fig = plt.figure(figsize=(3.0, 0.24))
    h = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], ms=3.4, lw=1.1,
                label=TITLES[m]) for m in MODELS]
    fig.legend(handles=h, loc="center", ncol=4, fontsize=LABEL, frameon=False,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.4)
    fig.savefig(OUT / "legend_s_scan.pdf", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)

    with open(OUT / "manifest_sscan.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[ok] {len(rows) - 1} s-scan panels -> {OUT}")


if __name__ == "__main__":
    main()
