#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit the main-experiment rank table in the requested layout (s = 0.2/0.4/0.6/0.8).

Data: result/add_exp/s_sensitivity/noise_s_rank_maincal_full.json
      + result/criteria/clean/classification/<Model>_se.pkl (SE at s=0.2)
      + noise_s_rank_maincal.json (SE = std/3 for s>=0.4)
Output: score/rank_s_own_maincal.tex
"""

import json
import pickle
from pathlib import Path

import numpy as np

from ..common.sites import result_dir
from ...common.config import task_dir

OUT_DIR = result_dir("add_exp/s_sensitivity")
FULL = OUT_DIR / "noise_s_rank_maincal_full.json"
RAW = OUT_DIR / "noise_s_rank_maincal.json"
SE_PKL = task_dir("classification", "clean")
OUT = OUT_DIR / "rank_s_own_maincal.tex"

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
NOISES = [("attack", r"$\gamma_p$"), ("gauss", r"$\gamma_g$"), ("salt", r"$\gamma_s$"),
          ("move", r"$\gamma_m$"), ("occ", r"$\gamma_o$")]
S_LIST = ["0.2", "0.4", "0.6", "0.8"]


def se_table():
    raw = json.load(open(str(RAW), encoding="utf-8"))
    out = {s: {c: {} for c, _ in NOISES} for s in S_LIST}
    for m in MODELS:
        d = pickle.load(open(str(SE_PKL / f"{m}_se.pkl"), "rb"))
        for c, _ in NOISES:
            out["0.2"][c][m] = float(np.asarray(d[c], dtype=float)[-1].mean())
    for s in ["0.4", "0.6", "0.8"]:
        for c, _ in NOISES:
            for m in MODELS:
                out[s][c][m] = raw["results"][c][s][m]["std"] / 3.0
    return out


def ranks_fixed(means, thr):
    order = sorted(MODELS, key=lambda m: -means[m])
    out = {order[0]: 1}
    rank, lead = 1, order[0]
    for i, m in enumerate(order[1:], start=2):
        if abs(means[lead] - means[m]) < thr:
            out[m] = rank
        else:
            rank, lead = i, m
            out[m] = rank
    return out


def main():
    full = json.load(open(str(FULL), encoding="utf-8"))
    se = se_table()
    se_s = {s: max(se[s][c][m] for c, _ in NOISES for m in MODELS) for s in S_LIST}
    thr_s = {s: 2.77 * se_s[s] for s in S_LIST}

    ncol = len(NOISES) * len(S_LIST)
    head1 = [r"\multicolumn{1}{c}{}"]
    head2 = [r"\multicolumn{1}{c}{}"]
    cmid = []
    start = 2
    for _, label in NOISES:
        head1.append(r"\multicolumn{%d}{c}{%s}" % (len(S_LIST), label))
        cmid.append(r"\cmidrule(lr){%d-%d}" % (start, start + len(S_LIST) - 1))
        start += len(S_LIST)
    for _ in NOISES:
        head2 += S_LIST

    lines = [
        r"\begin{table}[htbp]",
        r"		\caption{Rankings of the four networks under each noise type for different"
        r" selection ratios $s$.}",
        r"		\label{tab:rank_s_own}",
        r"		\centering",
        r"		\setlength{\tabcolsep}{4pt}",
        r"		\begin{threeparttable}",
        r"			\begin{tabular}{l*{%d}{c}}" % ncol,
        r"				\toprule",
        r"				" + " & ".join(head1) + r" \\",
        r"				" + " ".join(cmid),
        r"				" + " & ".join(head2) + r" \\",
        r"				\midrule",
    ]
    for m in MODELS:
        cells = []
        for c, _ in NOISES:
            for s in S_LIST:
                means = {mm: full["results"][c][s][mm]["mean"] for mm in MODELS}
                cells.append(str(ranks_fixed(means, thr_s[s])[m]))
        lines.append(r"				%s & %s \\" % (TITLES[m], " & ".join(cells)))

    thr_txt = ", ".join("%.2f" % (thr_s[s] * 1e3) for s in S_LIST)
    lines += [
        r"				\bottomrule",
        r"			\end{tabular}",
        r"			\begin{tablenotes}",
        r"				\footnotesize",
        r"				\item[*] The tie threshold is $2.77\,\mathrm{SE}$ with $\mathrm{SE}$ the"
        r" largest standard error among the four networks and five noise types at that $s$:"
        r" $%s$ ($\times 10^{-3}$) for $s = %s$, respectively." % (thr_txt, ", ".join(S_LIST)),
        r"			\end{tablenotes}",
        r"		\end{threeparttable}",
        r"	\end{table}",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("[ok] wrote", OUT)


if __name__ == "__main__":
    main()
