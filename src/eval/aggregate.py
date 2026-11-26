"""Aggregate per-repeat criteria into the standard error of the class-averaged criterion.

``run_probe_eval`` stores every repeat in ``result/repeat_raw/<task>/``.  This
module turns those raw matrices into ``result/standard_error/se_agg_by_layer.csv`` with the
covariance-aware SE (average over facets first, then take the standard error),
which is the quantity discussed in the paper's uncertainty remark.

Usage::

    python -m src.eval.aggregate
"""
import csv

import numpy as np

from ..common.config import TASKS, RESULT_ROOT, repeat_raw_dir


def run(out_path=None):
    out_path = out_path or (RESULT_ROOT / "standard_error" / "se_agg_by_layer.csv")
    rows = [["task", "model", "layer", "condition", "se_agg", "se_classmax",
             "n_facets", "n_repeats"]]

    for task in TASKS:
        d = repeat_raw_dir(task)
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.npz")):
            model, cond = path.stem.rsplit("_", 1)
            with np.load(path) as z:
                stack, final = z["stack"], z["final"]      # (R,L,K), (R,K)
            R = stack.shape[0]
            denom = np.sqrt(max(R - 1, 1))
            n_facets = final.shape[1]

            a = final.mean(axis=1)
            se = a.std(ddof=0) / denom
            se_cls = (final.std(axis=0, ddof=0) / denom).max()
            rows.append([task, model, "out", cond, "%.8f" % se, "%.8f" % se_cls, n_facets, R])

            for layer in range(stack.shape[1]):
                b = stack[:, layer, :].mean(axis=1)
                se_l = b.std(ddof=0) / denom
                se_lc = (stack[:, layer, :].std(axis=0, ddof=0) / denom).max()
                rows.append([task, model, "L%d" % layer, cond,
                             "%.8f" % se_l, "%.8f" % se_lc, n_facets, R])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("wrote", out_path, len(rows) - 1, "rows")
    return out_path


if __name__ == "__main__":
    run()
