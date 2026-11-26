#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize the 10-draw noise stability: two figures + summary tables.

Figure 1: pairwise cos among the 10 noisy weights (repeatability)
Figure 2: cos between the noisy weights and the clean weights (fidelity)
Both figures are a 2×4 grid over the 8 sites; shaded band = mean ± 1 s.d.
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
RES = result_dir("add_exp/probe_stability/weight_vs_noise")
FIG = figure_dir("add_exp/probe_stability/weight_vs_noise")
SITE_ORDER = ["VGG_L3", "VGG_L5", "VGG_L7", "VGG_L9",
              "RES_L3", "RES_L4", "RES_L6", "RES_L8"]
COLORS = {"exact": "#d62728", "diag": "#1f77b4"}
MARKERS = {"exact": "o", "diag": "s"}


def load_data():
    data = {}
    for p in sorted(RES.glob("noise_pairwise_stability_*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj.get("sites"), dict):
            obj = obj["sites"]
        for site, e in obj.items():
            data.setdefault(site, {"d": e.get("d"), "draws": e.get("draws"),
                                   "facets": e.get("facets"), "rows": []})
            data[site]["rows"] = e["rows"]
    return data


def tidy(data):
    rows = []
    for site in SITE_ORDER:
        if site not in data:
            continue
        for r in data[site]["rows"]:
            row = dict(site=site, d=data[site]["d"], rho=float(r["rho"]),
                       rho_pct=100 * float(r["rho"]))
            for metric in ("cos_clean", "pairwise"):
                for mode in ("exact", "diag"):
                    for k in ("mean", "std", "min", "max"):
                        row[f"{metric}_{mode}_{k}"] = r[metric][mode][k]
            rows.append(row)
    return rows


def plot_metric(data, metric, ylabel, note, name):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.45), sharex=True, sharey=True)
    for i, site in enumerate(SITE_ORDER):
        ax = axes[i // 4, i % 4]
        if site not in data:
            ax.set_visible(False)
            continue
        rows = data[site]["rows"]
        xs = np.asarray([100 * r["rho"] for r in rows])
        for mode in ("exact", "diag"):
            y = np.asarray([r[metric][mode]["mean"] for r in rows])
            sd = np.asarray([r[metric][mode]["std"] for r in rows])
            lo = np.clip(y - sd, 0.0, 1.05)
            hi = np.clip(y + sd, 0.0, 1.05)
            ax.fill_between(xs, lo, hi, color=COLORS[mode], alpha=0.18, lw=0)
            ax.plot(xs, y, color=COLORS[mode], marker=MARKERS[mode],
                    ms=2.2, lw=0.9, label=mode)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 260)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([1, 5, 20, 100])
        ax.set_xticklabels(["1", "5", "20", "100"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{site} (d={data[site]['d']})", fontsize=7.5, pad=1.5)
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        if i // 4 == 1:
            ax.set_xlabel("noise / signal (%)", fontsize=7.0, labelpad=1.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.008),
               handlelength=1.1, handletextpad=0.35, columnspacing=0.9,
               borderaxespad=0.0)
    fig.supylabel(ylabel, fontsize=7.5, x=0.020)
    fig.text(0.5, 0.105, note, ha="center", va="center", fontsize=6.2, color="0.35")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.935, bottom=0.215,
                        wspace=0.10, hspace=0.20)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def main():
    data = load_data()
    rows = tidy(data)
    if not rows:
        raise SystemExit(f"no data under {RES}")
    with open(RES / "noise_pairwise_stability_by_rho.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-site summary: average over the 8 rho values + rho=100%
    summary = []
    for site in SITE_ORDER:
        vals = [r for r in rows if r["site"] == site]
        if not vals:
            continue
        at100 = [r for r in vals if abs(r["rho"] - 1.0) < 1e-9][0]
        s = dict(site=site, d=vals[0]["d"], n_rho=len(vals))
        for metric in ("cos_clean", "pairwise"):
            for mode in ("exact", "diag"):
                s[f"{metric}_{mode}_avg"] = float(np.mean([r[f"{metric}_{mode}_mean"] for r in vals]))
                s[f"{metric}_{mode}_at100"] = float(at100[f"{metric}_{mode}_mean"])
        summary.append(s)
    with open(RES / "noise_pairwise_stability_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    plot_metric(data, "pairwise", r"mean pairwise $\cos(w_i,w_j)$",
                "shaded: mean ± 1 s.d. over 45 pairs × 4 facets",
                "exp3_noise_pairwise_stability")
    plot_metric(data, "cos_clean", r"$\cos(w_{noisy},w_{clean})$",
                "shaded: mean ± 1 s.d. over 10 draws × 4 facets",
                "noise_vs_clean_supplementary")

    lines = ["# 10-draw noise experiment: repeatability vs similarity to the clean weights\n",
             "> 8 sites × 8 noise levels × 10 independent draws × 4 OVR facets. "
             "`pairwise` compares the 10 noisy weights only with each other; `cos_clean` "
             "compares the noisy weights with the clean full-sample weights.\n",
             "\n## Per site: average over the 8 rho values / value at rho=100%\n",
             "| Site | d | pair exact | pair diag | clean exact | clean diag |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in summary:
        lines.append(f"| {s['site']} | {s['d']} | {s['pairwise_exact_avg']:.3f} | "
                     f"{s['pairwise_diag_avg']:.3f} | {s['cos_clean_exact_avg']:.3f} | "
                     f"{s['cos_clean_diag_avg']:.3f} |")
    lines.append("\n![pairwise](../../../../figure/generated/add_exp/probe_stability/weight_vs_noise/exp3_noise_pairwise_stability.png)\n")
    lines.append("![clean](../../../../figure/generated/add_exp/probe_stability/weight_vs_noise/noise_vs_clean_supplementary.png)\n")
    lines.append("\n## Per-site values at rho=100%\n")
    lines.append("| Site | pair exact | pair diag | clean exact | clean diag |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summary:
        lines.append(f"| {s['site']} | {s['pairwise_exact_at100']:.3f} | "
                     f"{s['pairwise_diag_at100']:.3f} | {s['cos_clean_exact_at100']:.3f} | "
                     f"{s['cos_clean_diag_at100']:.3f} |")
    (RES / "noise_pairwise_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RES / "noise_pairwise_stability_by_rho.csv", RES / "noise_pairwise_stability_summary.csv",
          RES / "noise_pairwise_stability_summary.md")


if __name__ == "__main__":
    main()

