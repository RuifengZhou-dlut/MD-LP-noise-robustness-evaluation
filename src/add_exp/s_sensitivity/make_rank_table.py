#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 5.4.3, main-experiment setting: s-scan rank table (two-level header of
noise x s, networks down the rows, ranks as numbers).

Setting: main-experiment sample selection (keep only samples the network classifies
      correctly, sort by the last-layer probe values in ascending order, and take
      floor(s*n_pos) / floor(s*n_neg) per facet); s=0.2 is taken directly from the
      main experiment and s=0.4/0.6/0.8 are recomputed values.
Decision rule: for each s take the largest SE across the four networks and five noise
      types (s=0.2 from the main-experiment _se.pkl, s>=0.4 from std/sqrt(9) over the
      10 repeats); threshold = 2.77*SE_s; |Delta| < threshold counts as a tie
      (competition ranking 1,1,3,4).
Output: score/s_rank_table_maincal.tex
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
OUT = OUT_DIR / "s_rank_table_maincal.tex"

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
TITLES = {"VGG-11": "VGGNet", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "MLP-Mixer"}
NOISES = [("attack", r"$\gamma_p$"), ("gauss", r"$\gamma_g$"), ("salt", r"$\gamma_s$"),
          ("move", r"$\gamma_m$"), ("occ", r"$\gamma_o$")]
S_LIST = ["0.2", "0.4", "0.6", "0.8"]


def se_table():
    """{s: {noise: {model: SE}}}"""
    full = json.load(open(str(FULL), encoding="utf-8"))
    raw = json.load(open(str(RAW), encoding="utf-8"))
    out = {s: {c: {} for c, _ in NOISES} for s in S_LIST}
    for m in MODELS:
        d = pickle.load(open(str(SE_PKL / f"{m}_se.pkl"), "rb"))
        for c, _ in NOISES:
            out["0.2"][c][m] = float(np.asarray(d[c], dtype=float)[-1].mean())
    for s in ["0.4", "0.6", "0.8"]:
        for c, _ in NOISES:
            for m in MODELS:
                out[s][c][m] = raw["results"][c][s][m]["std"] / 3.0   # n=10 -> sqrt(n-1)=3
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

    head1 = [r"\multicolumn{1}{c}{}"]
    head2 = [r"\multicolumn{1}{c}{}"]
    cmid = []
    start = 2
    for _, label in NOISES:
        head1.append(r"\multicolumn{4}{c}{%s}" % label)
        cmid.append(r"\cmidrule(lr){%d-%d}" % (start, start + 3))
        start += 4
    for _ in NOISES:
        head2 += S_LIST

    lines = [
        r"\begin{table}[htbp]",
        r"	\caption{Rankings of the four networks under each noise type for different"
        r" selection ratios $s$ (rank $1$ = most robust; $\gamma_p$ PGD, $\gamma_g$ Gaussian,"
        r" $\gamma_s$ salt-pepper, $\gamma_m$ motion-blur, $\gamma_o$ occlusion). The"
        r" evaluation follows the main experiment: only samples the network classifies"
        r" correctly are kept, and each facet takes the $\lfloor s\,n_{\mathrm{pos}}\rfloor$"
        r" / $\lfloor s\,n_{\mathrm{neg}}\rfloor$ smallest-margin ones; the $s=0.2$ column is"
        r" the main-experiment result. Two networks share a rank when their accuracy differs by"
        r" less than the tie threshold $2.77\,\mathrm{SE}_s$ given in the table note.}",
        r"	\label{tab:rank_s_maincal}",
        r"	\centering",
        r"	\setlength{\tabcolsep}{4pt}",
        r"	\begin{threeparttable}",
        r"		\begin{tabular}{l*{%d}{c}}" % (len(NOISES) * len(S_LIST)),
        r"			\toprule",
        r"			" + " & ".join(head1) + r" \\",
        r"			" + " ".join(cmid),
        r"			" + " & ".join(head2) + r" \\",
        r"			\midrule",
    ]
    for m in MODELS:
        cells = []
        for c, _ in NOISES:
            for s in S_LIST:
                means = {mm: full["results"][c][s][mm]["mean"] for mm in MODELS}
                cells.append(str(ranks_fixed(means, thr_s[s])[m]))
        lines.append(r"			%s & %s \\" % (TITLES[m], " & ".join(cells)))

    thr_txt = ", ".join("%.2f" % (thr_s[s] * 1e3) for s in S_LIST)
    lines += [
        r"			\bottomrule",
        r"		\end{tabular}",
        r"		\begin{tablenotes}",
        r"			\footnotesize",
        r"			\item[*] Accuracies are whole-network OVR means over ten repeated noise"
        r" realizations. The tie threshold is $2.77\,\mathrm{SE}_s$, with $\mathrm{SE}_s$ the"
        r" largest standard error among the four networks and five noise types at that $s$:"
        r" $%s$ ($\times 10^{-3}$) for $s = %s$, respectively." % (thr_txt, ", ".join(S_LIST)),
        r"		\end{tablenotes}",
        r"	\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("[ok] wrote", OUT)


if __name__ == "__main__":
    main()
