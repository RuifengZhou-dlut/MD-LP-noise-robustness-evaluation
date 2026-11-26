#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""8 sites × 10 repeats: weight-stability cosine (only the 10 weights are compared with
each other, never with the full-sample weights).

For each (site, n/d, mode):
  - take the solutions w_1...w_10 from 10 independent draws;
  - compute the 45 pairs cos(w_i, w_j);
  - average those 45 pairs per OVR facet, then average over the 4 facets;
  - that gives stability_cos for the cell (larger means more stable).

Outputs:
  result/add_exp/probe_stability/weight_vs_samples/small_n_stability_by_ratio.csv
  result/add_exp/probe_stability/weight_vs_samples/small_n_stability_summary.csv
  figure/generated/add_exp/probe_stability/weight_vs_samples/exp3_weight_stability_pairwise.png|pdf
  result/add_exp/probe_stability/weight_vs_samples/small_n_stability_summary.md
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from ...common.sites import figure_dir, result_dir
RES = result_dir("add_exp/probe_stability/weight_vs_samples")
FIG = figure_dir("add_exp/probe_stability/weight_vs_samples")
SITE_ORDER = ["VGG_L3", "VGG_L5", "VGG_L7", "VGG_L9",
              "RES_L3", "RES_L4", "RES_L6", "RES_L8"]
DIMS = {"VGG_L3": 4096, "VGG_L5": 2048, "VGG_L7": 1024, "VGG_L9": 256,
        "RES_L3": 4096, "RES_L4": 4096, "RES_L6": 2048, "RES_L8": 1024}
COLORS = {"exact": "#d62728", "diag": "#1f77b4"}
MARKERS = {"exact": "o", "diag": "s"}


def load_data():
    data = {}
    for p in sorted(RES.glob("small_n_stability_*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for site, e in obj.get("sites", {}).items():
            data.setdefault(site, {"d": e.get("d", DIMS.get(site)),
                                   "ratios": {}})["ratios"].update(e.get("ratios", {}))
    return data


def main():
    data = load_data()
    rows = []
    for site in SITE_ORDER:
        if site not in data:
            print("missing", site)
            continue
        for rs, r in sorted(data[site]["ratios"].items(), key=lambda kv: float(kv[0])):
            rows.append(dict(
                site=site, d=int(data[site]["d"]), ratio=float(rs),
                exact_stability=r["exact"].get("pairwise_mean", float("nan")),
                diag_stability=r["diag"].get("pairwise_mean", float("nan")),
                exact_std=r["exact"].get("pairwise_std", float("nan")),
                diag_std=r["diag"].get("pairwise_std", float("nan")),
                exact_min=r["exact"].get("pairwise_min", float("nan")),
                diag_min=r["diag"].get("pairwise_min", float("nan")),
            ))
    if not rows:
        raise SystemExit(f"no data under {RES}")

    with open(RES / "small_n_stability_by_ratio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for site in SITE_ORDER:
        vals = [r for r in rows if r["site"] == site]
        if not vals:
            continue
        ex = np.array([r["exact_stability"] for r in vals])
        dg = np.array([r["diag_stability"] for r in vals])
        at1 = [r for r in vals if abs(r["ratio"] - 1.0) < 1e-9]
        summary.append(dict(
            site=site, d=int(vals[0]["d"]), n_ratios=len(vals),
            exact_avg=float(ex.mean()), diag_avg=float(dg.mean()),
            exact_at1=float(at1[0]["exact_stability"]) if at1 else float("nan"),
            diag_at1=float(at1[0]["diag_stability"]) if at1 else float("nan"),
            exact_std_avg=float(np.mean([r["exact_std"] for r in vals])),
            diag_std_avg=float(np.mean([r["diag_std"] for r in vals])),
            exact_min=float(ex.min()), diag_min=float(dg.min()),
            exact_max=float(ex.max()), diag_max=float(dg.max()),
        ))

    with open(RES / "small_n_stability_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # figure: 2x4 grid, y = mean pairwise cosine among the 10 weights
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.45), sharex=True, sharey=True)
    for i, site in enumerate(SITE_ORDER):
        ax = axes[i // 4, i % 4]
        vals = [r for r in rows if r["site"] == site]
        if not vals:
            ax.set_visible(False)
            continue
        xs = np.array([r["ratio"] for r in vals])
        for mode in ("exact", "diag"):
            y = np.array([r[f"{mode}_stability"] for r in vals])
            sd = np.array([r[f"{mode}_std"] for r in vals])
            ax.fill_between(xs, np.clip(y - sd, 0.0, 1.05),
                            np.clip(y + sd, 0.0, 1.05),
                            color=COLORS[mode], alpha=0.16, lw=0)
            ax.plot(xs, y, color=COLORS[mode], marker=MARKERS[mode],
                    ms=2.2, lw=0.9, label=mode)
        ax.axvline(1.0, color="0.35", ls=":", lw=0.7)
        ax.set_xlim(0.45, 1.55)
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax.set_xticklabels(["0.50", "0.75", "1.00", "1.25", "1.50"])
        ax.set_title(f"{site} (d={DIMS.get(site, vals[0]['d'])})",
                     fontsize=7.5, pad=1.5)
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        if i // 4 == 1:
            ax.set_xlabel("n / d", fontsize=7.5, labelpad=1.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.008),
               handlelength=1.1, handletextpad=0.35, columnspacing=0.9,
               borderaxespad=0.0)
    fig.supylabel(r"mean pairwise $\cos(w_i,w_j)$", fontsize=8.0, x=0.018)
    fig.text(0.5, 0.105,
             "shaded: mean ± 1 s.d. over 45 pairs × 4 OVR facets",
             ha="center", va="center", fontsize=6.2, color="0.35")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.935, bottom=0.215,
                        wspace=0.10, hspace=0.20)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"exp3_weight_stability_pairwise.{ext}", dpi=300)
    plt.close(fig)

    # text summary
    lines = ["# Stability cosine of the 10 repeated weights (8 sites)\n",
             "> Stability cosine = mean over the 45 pairs `cos(w_i,w_j)` of the weights from "
             "the 10 independent draws at the same (site, n/d, mode); averaged over the "
             "4 OVR facets. **Never compared with the full-sample weights.**\n"]
    lines.append("\n## Per site: average over the 13 ratios / value at n/d=1.0\n")
    lines.append("| Site | d | exact mean ± std | diag mean ± std | exact @1.0 | diag @1.0 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in summary:
        lines.append(f"| {r['site']} | {r['d']} | {r['exact_avg']:.3f} ± {r['exact_std_avg']:.3f} | "
                     f"{r['diag_avg']:.3f} ± {r['diag_std_avg']:.3f} | "
                     f"{r['exact_at1']:.3f} | {r['diag_at1']:.3f} |")
    ex_all = np.array([r["exact_stability"] for r in rows])
    dg_all = np.array([r["diag_stability"] for r in rows])
    ex_ok = np.array([r["exact_stability"] for r in rows if r["site"] != "VGG_L7"])
    dg_ok = np.array([r["diag_stability"] for r in rows if r["site"] != "VGG_L7"])
    lines.append(f"\n- 8 sites × 13 ratios: exact mean **{ex_all.mean():.3f}**, diag mean **{dg_all.mean():.3f}**; "
                 f"cells with diag > exact: **{np.mean(dg_all > ex_all)*100:.1f}%**.")
    lines.append(f"- Excluding VGG_L7: exact **{ex_ok.mean():.3f}**, diag **{dg_ok.mean():.3f}**; "
                 f"cells with diag > exact: **{np.mean(dg_ok > ex_ok)*100:.1f}%**.")
    lines.append("\n![stability](../../../../figure/generated/add_exp/probe_stability/weight_vs_samples/exp3_weight_stability_pairwise.png)\n")
    (RES / "small_n_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RES / "small_n_stability_by_ratio.csv", RES / "small_n_stability_summary.csv",
          RES / "small_n_stability_summary.md")
    print("wrote", FIG / "exp3_weight_stability_pairwise.png")


if __name__ == "__main__":
    main()

