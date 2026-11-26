#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weighted scores and rankings of whole-network robustness (main clean experiment).

Purpose
-------
Weight the **whole-network robustness metric** stored in
`result/criteria/clean/<task>/<Model>_robustness.pkl` over the 5 perturbation
conditions, yielding weighted scores and model rankings under 4 weighting schemes, and
write `score_report.md` (scores, rankings, and how the four rankings differ) together
with a few csv files and a figure.

Metric definition
-----------------
Whole network = last row of the robustness matrix (the output layer), averaged across
facets: 4 one-vs-rest facets for classification, 9 quantile-threshold buckets for
regression. This is the "final" column of `result/criteria/clean/<task>.csv`.

Usage
-----
    python -m src.eval.scoring

Outputs (all written next to this script)
-----------------------------------------
    score_report.md      main document
    scores_wide.csv      per model x per scheme: score / rank / standard error
    scores_long.csv      long format
    ranking_diff.csv     rank comparison and shift relative to mean weighting
    kendall_tau.csv      pairwise rank agreement
    figure/generated/scoring/rank_bump.png
"""

import argparse
import csv
import math
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..common.config import FIGURE_ROOT, RESULT_ROOT, summary_csv, task_dir

HERE = Path(__file__).resolve().parent
OUT_DIR = RESULT_ROOT / "scoring"
DEFAULT_BASE = HERE.parents[1]

COND_ORDER = ["attack", "gauss", "salt", "move", "occ"]
COND_LABEL = {
    "attack": "PGD adversarial",
    "gauss": "Gaussian noise",
    "salt": "Salt-pepper noise",
    "move": "Translation",
    "occ": "Occlusion",
}
MODEL_ORDER = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
MODEL_COLOR = {
    "VGG-11": "#1f77b4",
    "Res-18": "#ff7f0e",
    "ViT-8": "#2ca02c",
    "Mix-8": "#d62728",
}
TASK_LABEL = {"classification": "Classification", "regression": "Regression"}
SCOPE_LABEL = {"classification": "Classification task", "regression": "Regression task",
            "combined": "Both tasks combined"}
SCOPE_EN = {"classification": "Classification", "regression": "Regression", "combined": "Mean of both tasks"}

SCHEMES = [
    ("avg", "Mean weighting", {c: 0.2 for c in COND_ORDER}),
    ("pgd", "PGD-dominant", {"attack": 0.6, "gauss": 0.1, "salt": 0.1, "move": 0.1, "occ": 0.1}),
    ("gauss", "Gauss-dominant", {"attack": 0.1, "gauss": 0.6, "salt": 0.1, "move": 0.1, "occ": 0.1}),
    ("salt", "Salt-dominant", {"attack": 0.1, "gauss": 0.1, "salt": 0.6, "move": 0.1, "occ": 0.1}),
]
SCHEME_KEYS = [s[0] for s in SCHEMES]
SCHEME_LABEL = {s[0]: s[1] for s in SCHEMES}
SCHEME_W = {s[0]: s[2] for s in SCHEMES}
SCHEME_EN = {"avg": "Mean", "pgd": "PGD-dom", "gauss": "Gauss-dom", "salt": "Salt-dom"}

CST = timezone(timedelta(hours=8))


# ---------------------------------------------------------------- reading and computing

def load_pkl(path):
    with open(str(path), "rb") as f:
        return pickle.load(f)


def final_mean(mat):
    """Whole network: last row of the matrix, averaged across facets."""
    arr = np.asarray(mat, dtype=np.float64)
    return float(arr[-1].mean())


def hidden_layer_mean(mat):
    """Layer mean: hidden-layer rows (excluding the whole-network last row) averaged
    across facets, consistent with the layer_mean column of criteria/*.csv.
    Used only in the metric-sensitivity appendix."""
    arr = np.asarray(mat, dtype=np.float64)
    return float(arr[:-1].mean())


def discover_models(task_dir):
    found = sorted(p.name[: -len("_robustness.pkl")] for p in Path(task_dir).glob("*_robustness.pkl"))
    ordered = [m for m in MODEL_ORDER if m in found]
    ordered += [m for m in found if m not in MODEL_ORDER]
    return ordered


def read_criteria_csv(path):
    rows = {}
    with open(str(path), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["model"], row["condition"])] = row
    return rows


def collect(base, exp, task):
    cdir = task_dir(task, exp)
    models = discover_models(cdir)
    if not models:
        raise SystemExit("[error] no *_robustness.pkl found under %s" % cdir)

    table = read_criteria_csv(summary_csv(task, exp))
    data = {"models": models, "acc": {}, "layer_mean": {}, "se": {}, "n_facets": {}, "base_acc": {}}

    for m in models:
        rob = load_pkl(cdir / ("%s_robustness.pkl" % m))
        # The uncertainty of the metric is the standard error of the mean (SE), not the
        # per-measurement standard deviation s.
        se_path = cdir / ("%s_se.pkl" % m)
        if se_path.exists():
            se = load_pkl(se_path)
        else:
            n = _n_repeats(base, exp, task)
            std = load_pkl(cdir / ("%s_std.pkl" % m))
            se = {c: np.asarray(std[c], dtype=np.float64) / math.sqrt(max(n - 1, 1)) for c in std}
        missing = [c for c in COND_ORDER if c not in rob]
        if missing:
            raise SystemExit(
                "[error] %s/%s is missing condition(s) %s; this scoring requires all 5 "
                "conditions (attack/gauss/salt/move/occ), whereas the noise branch only "
                "provides gauss/salt, so the same weights cannot be applied." % (task, m, missing)
            )
        data["acc"][m] = {c: final_mean(rob[c]) for c in COND_ORDER}
        data["layer_mean"][m] = {c: hidden_layer_mean(rob[c]) for c in COND_ORDER}
        data["se"][m] = {c: final_mean(se[c]) for c in COND_ORDER}
        data["n_facets"][m] = int(np.asarray(rob[COND_ORDER[0]]).shape[1])
        # Cross-check against the summary csv (its final column is the whole-network
        # facet mean).
        for c in COND_ORDER:
            ref = float(table[(m, c)]["final"])
            if abs(ref - data["acc"][m][c]) > 5e-4:
                print("[warn] %s/%s/%s pkl=%.6f csv=%.6f mismatch"
                      % (task, m, c, data["acc"][m][c], ref))
        if (m, "base") in table:
            data["base_acc"][m] = float(table[(m, "base")]["final"])
    return data


def _n_repeats(base, exp, task):
    """Fallback used only when `_se.pkl` is missing: the repeat count (10) of the main
    clean experiment."""
    return 10


def weighted_score(acc, weights):
    return sum(weights[c] * acc[c] for c in COND_ORDER)


def weighted_se(std, weights):
    """Propagate each condition's standard error of the mean (SE) by the sum of squared
    weights; the SE already includes the 1/sqrt(n) reduction."""
    return math.sqrt(sum((weights[c] ** 2) * (std[c] ** 2) for c in COND_ORDER))


def rank_models(scores, models):
    order = sorted(models, key=lambda m: (-scores[m], MODEL_ORDER.index(m) if m in MODEL_ORDER else 99))
    return {m: i + 1 for i, m in enumerate(order)}, order


def kendall(rank_a, rank_b, models):
    conc = disc = 0
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[models[i]] - rank_a[models[j]]
            db = rank_b[models[i]] - rank_b[models[j]]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    denom = n * (n - 1) / 2.0
    return (conc - disc) / denom, conc, disc


def flipped_pairs(ranks_ref, ranks_alt, models):
    """Model pairs whose relative order is reversed with respect to the reference
    ranking. Smaller rank = better."""
    out = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            if (ranks_ref[a] - ranks_ref[b]) * (ranks_alt[a] - ranks_alt[b]) < 0:
                out.append((a, b))
    return out


def build_scores(data, metric="final"):
    """Return {scheme: {'score': {m: ..}, 'se': {m: ..}, 'rank': {m: ..}, 'order': [..]}}"""
    out = {}
    for key, _label, w in SCHEMES:
        accs = data["acc"] if metric == "final" else data["layer_mean"]
        scores = {m: weighted_score(accs[m], w) for m in data["models"]}
        ses = {m: weighted_se(data["se"][m], w) for m in data["models"]}
        ranks, order = rank_models(scores, data["models"])
        out[key] = {"score": scores, "se": ses, "rank": ranks, "order": order}
    return out


# ---------------------------------------------------------------- md helpers

def f4(x):
    return "%.4f" % x


def md_table(headers, rows, aligns=None):
    n = len(headers)
    aligns = aligns or ["---"] * n
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)





# ---------------------------------------------------------------- figure

def make_figure(panels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[warn] matplotlib unavailable, skipping the figure: %s" % exc)
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.3))
    if len(panels) == 1:
        axes = [axes]
    xs = list(range(len(SCHEME_KEYS)))
    labels = [SCHEME_EN[k] for k in SCHEME_KEYS]

    for ax, (title, ranks_by_scheme, models) in zip(axes, panels):
        for m in models:
            ys = [ranks_by_scheme[k][m] for k in SCHEME_KEYS]
            ax.plot(xs, ys, "-o", color=MODEL_COLOR.get(m, "#555555"), lw=2.0, ms=7, label=m)
            for x, y in zip(xs, ys):
                ax.annotate(str(int(y)), (x, y), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color=MODEL_COLOR.get(m, "#555555"))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["1st", "2nd", "3rd", "4th"], fontsize=9)
        ax.set_ylim(4.45, 0.55)
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="both", ls=":", lw=0.6, alpha=0.6)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)

    axes[0].set_ylabel("Rank (1 = most robust)", fontsize=10)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main flow

def main():
    ap = argparse.ArgumentParser(description="Weighted robustness scores and rankings of the main experiment")
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="experiment root directory (defaults to the parent of score/)")
    ap.add_argument("--exp", default="clean",
                    help="criteria branch, defaults to clean (the main experiment)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    exp = args.exp
    tasks = ["classification", "regression"]
    data = {t: collect(base, exp, t) for t in tasks}

    # Both tasks share the same set of models
    models = data["classification"]["models"]
    if data["regression"]["models"] != models:
        raise SystemExit("[error] the two tasks have different model sets: %s vs %s"
                         % (models, data["regression"]["models"]))

    scores = {t: build_scores(data[t]) for t in tasks}
    scores_lm = {t: build_scores(data[t], metric="layer_mean") for t in tasks}

    # Combined: equal 0.5 weight per task
    combined = {}
    for key in SCHEME_KEYS:
        sc = {m: 0.5 * (scores["classification"][key]["score"][m]
                        + scores["regression"][key]["score"][m]) for m in models}
        se = {m: 0.5 * math.hypot(scores["classification"][key]["se"][m],
                                  scores["regression"][key]["se"][m]) for m in models}
        ranks, order = rank_models(sc, models)
        combined[key] = {"score": sc, "se": se, "rank": ranks, "order": order}
    allscores = dict(scores)
    allscores["combined"] = combined
    scopes = ["classification", "regression", "combined"]

    # Score gap and 2-sigma threshold between adjacent ranks within each scheme
    ADJ = []
    for s in scopes:
        for k in SCHEME_KEYS:
            order = allscores[s][k]["order"]
            for i in range(len(order) - 1):
                a, b = order[i], order[i + 1]
                gap = allscores[s][k]["score"][a] - allscores[s][k]["score"][b]
                thr = 2.0 * math.hypot(allscores[s][k]["se"][a], allscores[s][k]["se"][b])
                ADJ.append((s, k, a, b, gap, thr))
    THR_MIN = min(x[5] for x in ADJ)
    THR_MAX = max(x[5] for x in ADJ)

    # ---------- csv ----------
    with open(str(out / "scores_wide.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["scope", "model", "base"] + COND_ORDER
        for key in SCHEME_KEYS:
            head += ["%s_score" % key, "%s_se" % key, "%s_rank" % key]
        w.writerow(head)
        for scope in scopes:
            for m in models:
                if scope == "combined":
                    base_acc = ""
                    accs = [""] * len(COND_ORDER)
                else:
                    base_acc = f4(data[scope]["base_acc"].get(m, float("nan")))
                    accs = [f4(data[scope]["acc"][m][c]) for c in COND_ORDER]
                row = [scope, m, base_acc] + accs
                for key in SCHEME_KEYS:
                    row += [f4(allscores[scope][key]["score"][m]),
                            f4(allscores[scope][key]["se"][m]),
                            allscores[scope][key]["rank"][m]]
                w.writerow(row)

    with open(str(out / "scores_long.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "scheme", "model", "score", "se", "rank", "order"])
        for scope in scopes:
            for key in SCHEME_KEYS:
                s = allscores[scope][key]
                for m in models:
                    w.writerow([scope, key, m, f4(s["score"][m]), f4(s["se"][m]),
                                s["rank"][m], s["order"].index(m) + 1])

    with open(str(out / "ranking_diff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "model"] + ["rank_%s" % k for k in SCHEME_KEYS]
                   + ["delta_%s" % k for k in SCHEME_KEYS[1:]] + ["rank_min", "rank_max", "rank_span"])
        for scope in scopes:
            for m in models:
                rs = [allscores[scope][k]["rank"][m] for k in SCHEME_KEYS]
                w.writerow([scope, m] + rs
                           + [allscores[scope][k]["rank"][m] - rs[0] for k in SCHEME_KEYS[1:]]
                           + [min(rs), max(rs), max(rs) - min(rs)])

    with open(str(out / "kendall_tau.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "scheme_a", "scheme_b", "kendall_tau", "concordant", "discordant"])
        for scope in scopes:
            for i in range(len(SCHEME_KEYS)):
                for j in range(i + 1, len(SCHEME_KEYS)):
                    a, b = SCHEME_KEYS[i], SCHEME_KEYS[j]
                    tau, conc, disc = kendall(allscores[scope][a]["rank"], allscores[scope][b]["rank"], models)
                    w.writerow([scope, a, b, "%.4f" % tau, conc, disc])

    # ---------- figure ----------
    panels = []
    for scope in scopes:
        ranks = {k: allscores[scope][k]["rank"] for k in SCHEME_KEYS}
        panels.append((SCOPE_EN[scope], ranks, models))
    make_figure(panels, FIGURE_ROOT / "generated" / "scoring" / "rank_bump.png")

    # ---------- md ----------
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M")
    L = []
    A = L.append

    A("# Weighted robustness scores and rankings of the main-experiment networks")
    A("")
    A("> **Directory** `result/scoring`  ")
    A("> **Data source** `result/criteria/%s` (main experiment: clean-training branch)  " % exp)
    A("> **Generated** %s (CST)  " % now)
    A("> **Script** `src/eval/scoring.py` (reproduces everything in one run)")
    A("")

    # 0
    comb_tops = {k: allscores["combined"][k]["order"][0] for k in SCHEME_KEYS}
    A("## 0. Summary of conclusions")
    A("")
    if len(set(comb_tops.values())) == 1:
        A("1. Under all four weightings the **top model on the combined scope is always %s** "
          "(mean %s, PGD-dominant %s, Gauss-dominant %s, salt-dominant %s)."
          % (comb_tops["avg"],
             f4(allscores["combined"]["avg"]["score"][comb_tops["avg"]]),
             f4(allscores["combined"]["pgd"]["score"][comb_tops["avg"]]),
             f4(allscores["combined"]["gauss"]["score"][comb_tops["avg"]]),
             f4(allscores["combined"]["salt"]["score"][comb_tops["avg"]])))
    else:
        A("1. **The top model on the combined scope changes with the weights**: %s."
          % ", ".join("%s -> %s" % (SCHEME_LABEL[k], comb_tops[k]) for k in SCHEME_KEYS))
    pgd_reg = [x for x in ADJ if x[0] == "regression" and x[1] == "pgd"][0]
    hedge = ""
    if pgd_reg[4] < pgd_reg[5]:
        hedge = (", where the regression %s leads %s by only %s, inside the 2xSE threshold, "
                 "so the two are in fact tied (see 4.4)"
                 % (pgd_reg[2], pgd_reg[3], f4(pgd_reg[4])))
    A("2. Per task, under **mean weighting** the classification leader is %s (%s) and the "
      "regression leader is %s (%s); as soon as the weights favor PGD, the classification "
      "leader becomes %s (%s) and the regression leader becomes %s (%s)%s."
      % (scores["classification"]["avg"]["order"][0], f4(scores["classification"]["avg"]["score"][scores["classification"]["avg"]["order"][0]]),
         scores["regression"]["avg"]["order"][0], f4(scores["regression"]["avg"]["score"][scores["regression"]["avg"]["order"][0]]),
         scores["classification"]["pgd"]["order"][0], f4(scores["classification"]["pgd"]["score"][scores["classification"]["pgd"]["order"][0]]),
         scores["regression"]["pgd"]["order"][0], f4(scores["regression"]["pgd"]["score"][scores["regression"]["pgd"]["order"][0]]),
         hedge))
    n_distinct_cls = len({tuple(scores["classification"][k]["order"]) for k in SCHEME_KEYS})
    n_distinct_reg = len({tuple(scores["regression"][k]["order"]) for k in SCHEME_KEYS})
    A("3. Across the four weightings the four models produce only **%d distinct rankings "
      "(classification) / %d (regression)**: the differences concentrate on a few pairs "
      "that are close in total score or whose strength lies in a single condition, rather "
      "than on a full reshuffle."
      % (n_distinct_cls, n_distinct_reg))
    A("4. **Adjacent ranks whose score gap is below the combined 2xSE threshold should be "
      "treated as tied** (threshold %s to %s on this data; per pair, see section 4.4); "
      "only gaps above the threshold are worth stating as a conclusion." % (f4(THR_MIN), f4(THR_MAX)))
    A("")

    # 1
    A("## 1. Metric definition and weight design")
    A("")
    A("### 1.1 Metric")
    A("")
    A("- We use the **whole-network OVR accuracy**: the last row (output layer) of the "
      "robustness matrix, averaged across facets. There are 4 one-vs-rest facets for "
      "classification and 9 quantile-threshold buckets for regression; this is the "
      "\"final\" column of `result/criteria/clean/<task>.csv`.")
    A("- Evaluation is in-sample: 20000 training samples, and only samples where the "
      "last-layer projection agrees with the whole-network prediction are kept. It is "
      "therefore a **representation-level robustness**, not test-set accuracy.")
    A("- Five perturbations: `attack` (PGD adversarial), `gauss` (Gaussian), `salt` "
      "(salt-pepper), `move` (translation), `occ` (occlusion), each the mean over 10 repeats.")
    A("- **Higher is more robust.** `base` keeps samples where the probe prediction equals "
      "the network prediction, so the whole-network value is always 1.0000; the weighted "
      "score can therefore also be read as a weighted retention rate.")
    A("- We do not use `drop = base - cond`: with base ~ 1 it is only a constant shift of "
      "accuracy and ranks identically to accuracy, while accuracy itself is more "
      "intuitive and directly comparable across models.")
    A("")
    A("### 1.2 The four weighting schemes")
    A("")
    rows = []
    for key, label, w in SCHEMES:
        rows.append([label] + ["%.1f" % w[c] for c in COND_ORDER] + ["%.1f" % sum(w.values())])
    A(md_table(["Scheme"] + ["%s<br>`%s`" % (COND_LABEL[c], c) for c in COND_ORDER] + ["Total"], rows,
               ["---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]))
    A("")
    A("Each of the three \"dominant\" schemes raises its leading condition to **0.6** and "
      "gives the remaining four **0.1** each: the leading condition alone carries 60% of "
      "the weight, just above the sum of the other four (0.4), so it can change a rank on "
      "its own, whereas mean weighting is the only prior-free setting.")
    A("")

    # 2
    A("## 2. Raw panel: whole-network robustness per condition")
    A("")
    A("Start from the raw scores before weighting. The table lists whole-network OVR "
      "accuracy; the final \"mean over 5 conditions\" column is exactly the mean-weighting "
      "score (0.2 per condition).")
    A("")
    for task in tasks:
        A("### %s (%s)" % (TASK_LABEL[task], task))
        A("")
        rows = []
        for m in models:
            accs = [data[task]["acc"][m][c] for c in COND_ORDER]
            rows.append([m, f4(data[task]["base_acc"].get(m, float("nan")))]
                        + [f4(x) for x in accs] + [f4(float(np.mean(accs)))])
        A(md_table(["Model", "base"] + ["%s" % COND_LABEL[c] for c in COND_ORDER]
                   + ["Mean over 5 conditions"], rows,
                   ["---", ":---:"] + [":---:"] * 6))
        A("")
        # Four-network ranking per condition, calling out the strongest / weakest
        for c in COND_ORDER:
            ordered = sorted(models, key=lambda m: -data[task]["acc"][m][c])
            A("- %s: %s is highest (%s), %s is lowest (%s)."
              % (COND_LABEL[c], ordered[0], f4(data[task]["acc"][ordered[0]][c]),
                 ordered[-1], f4(data[task]["acc"][ordered[-1]][c])))
        A("")

    # 3
    A("## 3. Weighted scores and rankings")
    A("")
    A("Cells read `weighted score (rank)`, with rank 1 = most robust. Scores are "
      "accuracies on a [0,1] scale and are not renormalized (all conditions share the same "
      "scale and the same base). **Scores are comparable only within one scope**: "
      "classification, regression and combined measure different facets, so their "
      "magnitudes cannot be compared across scopes and only the orderings within each "
      "scope are meaningful. The paper table is `score_table.tex` (scores kept to 3 "
      "significant figures, recomputed from the pkl at full precision and then rounded).")
    A("")
    for scope in scopes:
        A("### 3.%d %s" % (scopes.index(scope) + 1, SCOPE_LABEL[scope]))
        A("")
        rows = []
        for m in models:
            cells = []
            for key in SCHEME_KEYS:
                cells.append("%s (%d)" % (f4(allscores[scope][key]["score"][m]),
                                          allscores[scope][key]["rank"][m]))
            rows.append([m] + cells)
        A(md_table(["Model"] + [SCHEME_LABEL[k] for k in SCHEME_KEYS], rows,
                   ["---"] + [":---:"] * len(SCHEME_KEYS)))
        A("")
        # One line naming the leader of each scheme
        firsts = ["%s -> %s" % (SCHEME_LABEL[k], allscores[scope][k]["order"][0]) for k in SCHEME_KEYS]
        A("Leader: " + "; ".join(firsts) + ".")
        A("")
        if scope == "combined":
            A("> Combined scope = 0.5 weight on each task's score (equivalent to equal "
              "weight per task); standard errors are propagated under an independence "
              "approximation.")
            A("")

    A("![rank bump](../../figure/generated/scoring/rank_bump.png)")
    A("")
    A("*Figure: rank trajectories under the four weightings (left to right: mean / "
      "PGD-dominant / Gauss-dominant / salt-dominant); the three panels are "
      "classification, regression and the combined scope. Flips between close scores "
      "(e.g. classification under PGD-dominant) show up as crossings in the plot.*")
    A("")

    # 4 differences
    A("## 4. How the four rankings differ")
    A("")
    A("### 4.1 Rank comparison and shift")
    A("")
    A("`Δ` is the rank change relative to mean weighting (negative = moved up). The "
      "narrower the `rank range`, the less sensitive that model's rank is to the weights.")
    A("")
    for scope in scopes:
        A("**%s**" % SCOPE_LABEL[scope])
        A("")
        rows = []
        for m in models:
            rs = [allscores[scope][k]["rank"][m] for k in SCHEME_KEYS]
            rows.append([m] + [str(r) for r in rs]
                        + ["%+d" % (allscores[scope][k]["rank"][m] - rs[0]) for k in SCHEME_KEYS[1:]]
                        + ["%d–%d" % (min(rs), max(rs))])
        A(md_table(["Model"] + [SCHEME_LABEL[k] for k in SCHEME_KEYS]
                   + ["ΔPGD", "ΔGauss", "ΔSalt", "Rank range"],
                   rows, ["---"] + [":---:"] * 8))
        A("")

    A("### 4.2 Rank agreement (Kendall τ)")
    A("")
    A("τ = 1 means the two schemes rank identically, τ = 0 means unrelated, and τ < 0 means "
      "reversed. Four models give only 6 model pairs, so τ takes the discrete values 1, "
      "0.667, 0.333, 0.")
    A("")
    for scope in scopes:
        A("**%s**" % SCOPE_LABEL[scope])
        A("")
        rows = []
        for i in range(len(SCHEME_KEYS)):
            for j in range(i + 1, len(SCHEME_KEYS)):
                a, b = SCHEME_KEYS[i], SCHEME_KEYS[j]
                tau, conc, disc = kendall(allscores[scope][a]["rank"], allscores[scope][b]["rank"], models)
                rows.append([SCHEME_LABEL[a], SCHEME_LABEL[b], "%.3f" % tau, "%d : %d" % (conc, disc)])
        A(md_table(["Scheme A", "Scheme B", "Kendall τ", "Concordant : discordant"], rows,
                   ["---", "---", ":---:", ":---:"]))
        A("")

    A("### 4.3 Which pairs actually flip")
    A("")
    A("Taking mean weighting as the reference, we check pair by pair whether the relative "
      "order reverses.")
    A("")
    any_flip = False
    for scope in scopes:
        A("**%s**" % SCOPE_LABEL[scope])
        A("")
        lines = []
        for key in SCHEME_KEYS[1:]:
            pairs = flipped_pairs(allscores[scope]["avg"]["rank"], allscores[scope][key]["rank"], models)
            if pairs:
                any_flip = True
                bits = []
                for a, b in pairs:
                    first0 = a if allscores[scope]["avg"]["rank"][a] < allscores[scope]["avg"]["rank"][b] else b
                    first1 = a if allscores[scope][key]["rank"][a] < allscores[scope][key]["rank"][b] else b
                    bits.append("%s ↔ %s (mean: %s first -> %s: %s first)"
                                % (a, b, first0, SCHEME_LABEL[key], first1))
                lines.append("- %s: %s" % (SCHEME_LABEL[key], "; ".join(bits)))
            else:
                lines.append("- %s: **identical** to the mean-weighting ranking" % SCHEME_LABEL[key])
        A("\n".join(lines))
        A("")
    if not any_flip:
        A("(On this data all four rankings are identical.)")
        A("")

    A("### 4.4 Score resolution: which rank gaps are meaningful")
    A("")
    A("Propagating the standard error of the 10-repeat mean (SE = s/sqrt(n)) through the "
      "sum of squared weights gives the standard error of each score; if the gap between "
      "two adjacent ranks is below 2x the combined standard error of the pair, the rank "
      "difference cannot be resolved within repeat noise.")
    A("")
    A("| Scope | Scheme | Adjacent ranks | Score gap | 2×SE threshold | Verdict |")
    A("| --- | --- | --- | ---: | ---: | --- |")
    tie_rows = []
    for s0, k, a, b, gap, thresh in ADJ:
        verdict = "**tied**" if gap < thresh else "resolvable"
        if gap < thresh:
            tie_rows.append((s0, k, a, b, gap, thresh))
        A("| %s | %s | %s > %s | %s | %s | %s |"
          % (SCOPE_LABEL[s0], SCHEME_LABEL[k], a, b, f4(gap), f4(thresh), verdict))
    A("")
    if tie_rows:
        A("%d adjacent rank gaps fall within 2xSE:" % len(tie_rows))
        A("")
        for s0, k, a, b, g, t in tie_rows:
            A("- %s · %s: %s and %s differ by %s (threshold %s)"
              % (SCOPE_LABEL[s0], SCHEME_LABEL[k], a, b, f4(g), f4(t)))
        A("")
        A("At these positions the rank is only indicative; describing them as "
          "\"essentially tied\" is safer than claiming \"A beats B\".")
        A("")

    A("### 4.5 Where the differences come from: how to read them")
    A("")
    cls = scores["classification"]
    reg = scores["regression"]

    def cond_order(task, cond):
        return sorted(models, key=lambda m: -data[task]["acc"][m][cond])

    def cond_rank(task, cond, m):
        return cond_order(task, cond).index(m) + 1

    # 1) Classification: who is strong / weak on PGD
    mix_non_pgd_first = all(cond_order("classification", c)[0] == "Mix-8"
                            for c in ["gauss", "salt", "move", "occ"])
    best_atk_cls = cond_order("classification", "attack")[0]
    A("1. **The classification flip is driven by PGD alone.** Mix-8 ranks first on "
      "gauss/salt/move/occ (%s), but only %d on PGD (%s); the strongest model on "
      "classification PGD is %s (%s). Once the PGD weight rises to 0.6, Mix-8 moves from "
      "rank %d to rank %d and %s moves from rank %d to rank %d."
      % ("all four" if mix_non_pgd_first else "not all four",
         cond_rank("classification", "attack", "Mix-8"),
         f4(data["classification"]["acc"]["Mix-8"]["attack"]),
         best_atk_cls, f4(data["classification"]["acc"][best_atk_cls]["attack"]),
         cls["avg"]["rank"]["Mix-8"], cls["pgd"]["rank"]["Mix-8"],
         best_atk_cls, cls["avg"]["rank"][best_atk_cls], cls["pgd"]["rank"][best_atk_cls]))
    # 2) Regression
    best_atk_reg = cond_order("regression", "attack")[0]
    worst_atk_reg = cond_order("regression", "attack")[-1]
    A("2. **PGD-dominant reshuffles regression the most.** The strongest model on PGD is %s "
      "(%s), lifted to rank %d by the 0.6 weight; the weakest is %s (%s), pushed down to "
      "rank %d (it is rank %d under mean weighting)."
      % (best_atk_reg, f4(data["regression"]["acc"][best_atk_reg]["attack"]),
         reg["pgd"]["rank"][best_atk_reg],
         worst_atk_reg, f4(data["regression"]["acc"][worst_atk_reg]["attack"]),
         reg["pgd"]["rank"][worst_atk_reg], reg["avg"]["rank"][worst_atk_reg]))
    # 3) Gauss / salt
    taus = [kendall(scores[t]["avg"]["rank"], scores[t][k]["rank"], models)[0]
            for t in tasks for k in ("gauss", "salt")]
    mild = []
    for t in tasks:
        for k in ("gauss", "salt"):
            for a, b in flipped_pairs(scores[t]["avg"]["rank"], scores[t][k]["rank"], models):
                mild.append("%s · %s: %s ↔ %s (gap %s)"
                            % (TASK_LABEL[t], SCHEME_LABEL[k], a, b,
                               f4(abs(scores[t][k]["score"][a] - scores[t][k]["score"][b]))))
    mild_txt = "; ".join(mild) if mild else "no pair flips at all"
    A("3. **Gauss- and salt-dominant are comparatively mild.** Under these two conditions "
      "the four networks' strength profile follows the overall one, and their τ against "
      "mean weighting lies in %.2f-%.2f; only the closest pairs flip: %s."
      % (min(taus), max(taus), mild_txt))
    A("4. **Every network has one weak condition**, and weighting only decides whose "
      "weakness gets amplified: in classification VGG-11 is weakest on translation "
      "(rank %d of four) and Mix-8 on PGD (rank %d); in regression ViT-8 is weakest on "
      "Gaussian (rank %d) and Mix-8 on PGD (rank %d)."
      % (cond_rank("classification", "move", "VGG-11"),
         cond_rank("classification", "attack", "Mix-8"),
         cond_rank("regression", "gauss", "ViT-8"),
         cond_rank("regression", "attack", "Mix-8")))
    A("")

    # 5
    A("## 5. Conclusions and recommendations")
    A("")
    A("- **Use mean weighting as the default**: it assumes no threat model and suits "
      "conclusions about overall robustness. On this data the classification order is %s "
      "and the regression order is %s."
      % (" > ".join(cls["avg"]["order"]), " > ".join(reg["avg"]["order"])))
    reg_hedge = ""
    if reg["avg"]["order"][0] != reg["pgd"]["order"][0]:
        g = [x for x in ADJ if x[0] == "regression" and x[1] == "pgd"][0]
        if g[4] < g[5]:
            reg_hedge = ("; this regression swap differs by only %s (within 2xSE), so the "
                         "more accurate statement is that %s and %s are tied for first"
                         % (f4(g[4]), g[2], g[3]))
    A("- **Switch to PGD-dominant for white-box adversarial settings**: the conclusions "
      "change substantially, with the classification winner moving from %s to %s and the "
      "regression winner from %s to %s%s. The two sets of conclusions do not contradict "
      "each other; they answer different questions (average robustness vs robustness to "
      "the strongest attack)."
      % (cls["avg"]["order"][0], cls["pgd"]["order"][0], reg["avg"]["order"][0], reg["pgd"]["order"][0],
         reg_hedge))
    A("- **Gauss- and salt-dominant are better treated as ablations**: their τ against "
      "mean weighting is %.2f-%.2f, so they add little information; if only one scheme is "
      "reported, do not substitute them for mean weighting."
      % (min(taus), max(taus)))
    A("- **Report rank gaps below the 2xSE threshold (%s to %s on this data; see section "
      "4.4) as tied**; otherwise a different random seed will flip them."
      % (f4(THR_MIN), f4(THR_MAX)))
    if len(set(comb_tops.values())) == 1:
        A("- **The combined scope gives the most stable conclusion**: %s ranks first under "
          "all four weightings, so its lead does not depend on the weight choice."
          % comb_tops["avg"])
    else:
        A("- **The combined-scope leader changes with the weights**: %s. Any conclusion "
          "should state the weights it uses."
          % ", ".join("%s -> %s" % (SCHEME_LABEL[k], comb_tops[k]) for k in SCHEME_KEYS))
    A("")
    A("## Appendix A: metric sensitivity - what changes under the layer mean")
    A("")
    A("The main metric uses the whole network (output layer). Switching to the layer mean "
      "(average over the hidden-layer rows, excluding the whole-network last row, "
      "consistent with the `layer_mean` column of `criteria/clean/<task>.csv`) changes the "
      "rankings markedly:")
    A("")
    for task in tasks:
        rows = []
        for m in models:
            rows.append([m] + ["%s (%d)" % (f4(scores_lm[task][k]["score"][m]), scores_lm[task][k]["rank"][m])
                               for k in SCHEME_KEYS])
        A("**%s (layer-mean metric)**" % TASK_LABEL[task])
        A("")
        A(md_table(["Model"] + [SCHEME_LABEL[k] for k in SCHEME_KEYS], rows,
                   ["---"] + [":---:"] * len(SCHEME_KEYS)))
        A("")
    lm_cls_top = scores_lm["classification"]["avg"]["order"][0]
    lm_reg_top = scores_lm["regression"]["avg"]["order"][0]
    changes = []
    if lm_cls_top != cls["avg"]["order"][0]:
        changes.append("classification rank 1 moves from %s to %s" % (cls["avg"]["order"][0], lm_cls_top))
    if lm_reg_top != reg["avg"]["order"][0]:
        changes.append("regression rank 1 moves from %s to %s" % (reg["avg"]["order"][0], lm_reg_top))
    if scores_lm["classification"]["avg"]["rank"]["VGG-11"] != cls["avg"]["rank"]["VGG-11"]:
        changes.append("in classification VGG-11 moves from rank %d to rank %d"
                       % (cls["avg"]["rank"]["VGG-11"],
                          scores_lm["classification"]["avg"]["rank"]["VGG-11"]))
    if changes:
        A("Against the main text (mean weighting, whole-network metric): %s." % "; ".join(changes))
    else:
        A("Against the main text (mean weighting, whole-network metric): the two "
          "mean-weighting rankings happen to coincide.")
    A("The reason is that the layer mean also averages in the shallow layers, whose "
      "accuracy is generally low, which favors networks with better cross-layer consistency.")
    A("The two metrics answer different questions. This directory uses the **whole "
      "network** throughout, because network robustness means how the output layer behaves "
      "under perturbation; the layer mean is better suited to discussing how decodable the "
      "representation is at each layer. **Do not mix conclusions across metrics.**")
    A("")
    A("## Appendix B: files and reproduction")
    A("")
    A("| File | Content |")
    A("| --- | --- |")
    A("| `score_report.md` | This document: scores, rankings, and how the four rankings differ |")
    A("| `src/eval/scoring.py` | Computation script (read pkl -> compute scores -> emit tables and figures) |")
    A("| `score_table.tex` | Booktabs paper table (booktabs + multirow, scores to 3 significant figures) |")
    A("| `src/eval/make_score_table.py` | Generates the .tex above |")
    A("| `scores_wide.csv` | Wide table: whole-network accuracy per model x condition plus score / SE / rank for the 4 schemes |")
    A("| `scores_long.csv` | Long table: `scope, scheme, model, score, se, rank` |")
    A("| `ranking_diff.csv` | Rank comparison, shift relative to mean weighting, rank range |")
    A("| `kendall_tau.csv` | Pairwise rank agreement |")
    A("| `figure/generated/scoring/rank_bump.png` | Rank-trajectory figure |")
    A("")
    A("Reproduction:")
    A("")
    A("```bash")
    A("cd <repository root>")
    A("python -m src.eval.scoring    # report + csv + figures")
    A("python -m src.eval.make_score_table     # paper booktabs table .tex")
    A("```")
    A("")
    A("Dependencies: `result/criteria/clean/<task>/<Model>_robustness.pkl`, `_std.pkl`, and "
      "`criteria/clean/<task>.csv` (the csv serves only for cross-checking; the script "
      "verifies that the pkl and the csv `final` column agree to within 5e-4).")
    A("")
    A("> Note: the noise-training branch `result/criteria/noise` provides only the three "
      "conditions `base/gauss/salt` and lacks `attack/move/occ`, so the same 5-condition "
      "weights cannot be applied and it takes no part in this scoring; to compare it, fill "
      "in the missing conditions first or define a separate two-condition weighting.")
    A("")

    (out / "score_report.md").write_text("\n".join(L), encoding="utf-8")

    print("[ok] wrote %s" % (out / "score_report.md"))
    for scope in scopes:
        print("  %-15s" % scope, " | ".join("%s: %s" % (k, " > ".join(allscores[scope][k]["order"]))
                                            for k in SCHEME_KEYS))


if __name__ == "__main__":
    main()
