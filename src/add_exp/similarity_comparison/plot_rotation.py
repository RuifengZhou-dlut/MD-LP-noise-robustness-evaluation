"""Plot MD-LP accuracy and CKA versus the rotation angle theta.

Usage:
    python -m src.add_exp.similarity_comparison.plot_rotation [--out path.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common.sites import figure_dir, result_dir

SITES = ["VGG_L3", "VGG_L9", "RES_L3", "RES_L8"]
CS = ["4", "8", "12", "16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(result_dir("add_exp/similarity_comparison") / "real_rot_insample.json"))
    ap.add_argument("--out", default=str(figure_dir("add_exp/similarity_comparison") / "real_rot_insample.png"))
    args = ap.parse_args()

    import json
    d = json.load(open(args.json, encoding="utf-8"))
    cmap = plt.get_cmap("viridis")

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 5.5, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.1), sharex=True)
    for j, site in enumerate(SITES):
        s = d["sites"][site]
        ths = sorted(s["res"]["4"], key=lambda t: int(t))
        x = [int(t) for t in ths]
        for i, metric in enumerate(("probe", "cka")):
            ax = axes[i, j]
            for k, c in enumerate(CS):
                y = [s["res"][c][t][metric] for t in ths]
                ax.plot(x, y, marker="o", ms=1.4, lw=0.9,
                        color=cmap(k / (len(CS) - 1)), label=f"c={c}")
            ax.set_title(f"{site} (d={s['d']})")
            ax.set_xticks([0, 30, 60, 90])
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.set_ylim(0.45, 1.02) if i == 0 else ax.set_ylim(-0.02, 1.02)
            if i == 1:
                ax.set_xlabel(r"$\theta$ (deg)")
            if j == 0:
                ax.set_ylabel("MD-LP acc" if i == 0 else "CKA")
    axes[0, 0].legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("saved", args.out)


if __name__ == "__main__":
    main()
