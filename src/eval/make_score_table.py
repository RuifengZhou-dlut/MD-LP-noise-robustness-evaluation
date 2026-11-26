#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the paper-ready booktabs table `score_table.tex` (English version).

Layout
------
Vertical  : two groups, Classification and Regression;
            each network occupies two rows -- weighted score on top, rank below;
Horizontal: Average, gamma_p-dominant, gamma_g-dominant, gamma_s-dominant.
Scores keep 3 significant figures (all scores lie in 0.1-1, so 3 decimals).

Same source as score_report.md: reads ../result/criteria/clean pkl files, recomputes at
full precision, then rounds once -- no double rounding.

Usage:
    python -m src.eval.make_score_table
"""

import argparse
import math
import sys
from pathlib import Path

from . import scoring as cs
from ..common.config import RESULT_ROOT

OUT_DIR = RESULT_ROOT / "scoring"
BASE = Path(__file__).resolve().parents[1]
TASKS = [("classification", "Classification"), ("regression", "Regression")]
SCHEME_HEAD = [
    ("avg", "Average"),
    ("pgd", r"$\gamma_p$-dominant"),
    ("gauss", r"$\gamma_g$-dominant"),
    ("salt", r"$\gamma_s$-dominant"),
]
N_COL = 3 + len(SCHEME_HEAD)


def s3(x):
    """3 significant figures: scores lie in 0.1-1, so 3 decimals == 3 significant figures."""
    return "%.3f" % x


def tie_notes(models, scores):
    """Adjacent pairs whose displayed scores are equal but ranks differ; explained in the note."""
    out = []
    for task, en in TASKS:
        for k, head_name in SCHEME_HEAD:
            order = scores[task][k]["order"]
            for i in range(len(order) - 1):
                a, b = order[i], order[i + 1]
                va = scores[task][k]["score"][a]
                vb = scores[task][k]["score"][b]
                if s3(va) != s3(vb):
                    continue
                thr = 2.0 * math.hypot(scores[task][k]["se"][a], scores[task][k]["se"][b])
                gap = va - vb
                tag = "below $2\\sigma$" if gap < thr else "above $2\\sigma$"
                out.append(
                    "%s, %s: %s and %s both %s (gap %s, %s; ranks %d, %d tied)"
                    % (en, head_name, a, b, s3(va), "%.4f" % gap, tag,
                       scores[task][k]["rank"][a], scores[task][k]["rank"][b])
                )
    return out


def build_table(base=None):
    base = Path(base) if base else BASE
    data = {t: cs.collect(base, "clean", t) for t, _ in TASKS}
    models = data["classification"]["models"]
    scores = {t: cs.build_scores(data[t]) for t, _ in TASKS}

    lines = [
        r"% Packages: booktabs, multirow, amsmath. Switch table -> table* for two-column layouts.",
        r"% Scores are recomputed from the pkl files at full precision and rounded once to 3 significant",
        r"% figures; the 4-decimal values in score_report.md may differ by up to 0.0005 (0.6345 -> 0.635).",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Weighted robustness scores and rankings under four weighting schemes}",
        r"  \label{tab:score}",
        r"  \begin{tabular}{@{}lllcccc@{}}",
        r"    \toprule",
        r"    \multirow{2}{*}{Task} & \multirow{2}{*}{Network} & \multirow{2}{*}{Metric}"
        r" & \multicolumn{4}{c}{Weighting} \\",
        r"    \cmidrule(l){4-%d}" % N_COL,
        r"    & & & " + " & ".join(h for _, h in SCHEME_HEAD) + r" \\",
    ]

    group_rows = 2 * len(models)
    for gi, (task, en) in enumerate(TASKS):
        lines.append(r"    \midrule")
        for mi, m in enumerate(models):
            task_cell = r"\multirow{%d}{*}{%s}" % (group_rows, en) if mi == 0 else ""
            score_cells = [s3(scores[task][k]["score"][m]) for k, _ in SCHEME_HEAD]
            rank_cells = [str(scores[task][k]["rank"][m]) for k, _ in SCHEME_HEAD]
            lines.append(
                r"    %s & \multirow{2}{*}{%s} & Score & %s \\"
                % (task_cell, m, " & ".join(score_cells))
            )
            lines.append(r"    & & Rank & %s \\" % " & ".join(rank_cells))
            if mi < len(models) - 1:
                lines.append(r"    \addlinespace[2pt]")
    notes = tie_notes(models, scores)
    note = (r"    \footnotesize Note: higher score = more robust; rank 1 = most robust. "
            r"Scores use 3 significant figures, ranks are from full-precision scores.")
    if notes:
        note += " " + "; ".join(notes) + "."
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\smallskip",
        r"  \begin{minipage}{\linewidth}",
        note,
        r"  \end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    tex = build_table(args.base)
    out = Path(args.out) / "score_table.tex"
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print("[ok] wrote %s" % out)
