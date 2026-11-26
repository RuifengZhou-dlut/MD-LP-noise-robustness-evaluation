"""Single-run benchmark for Section 5.2.1 (4-class OVR setup, single facet)."""
import argparse
import gc
import json
import os
import time

import numpy as np

from .cfg import SITES, features_path
from ..common.mdlp import (PeakRSSMonitor, accuracy, compute_mdlp, fit_least_squares,
                         fit_logistic_bgd, fit_logistic_sgd, save_json)


def run_method(method, X, y, positive):
    if method == "exact":
        return compute_mdlp(X, y, mode="exact", positive=positive)
    if method == "diag":
        return compute_mdlp(X, y, mode="diag", positive=positive)
    if method == "identity":
        return compute_mdlp(X, y, mode="identity", positive=positive)
    if method == "bgd":
        return fit_logistic_bgd(X, y, lam=1e-4, max_iter=3000, tol=1e-5,
                                positive=positive)
    if method == "sgd":
        return fit_logistic_sgd(X, y, lam=1e-4, epochs=30, batch_size=128, lr0=0.05,
                                positive=positive)
    if method == "ls":
        return fit_least_squares(X, y, positive=positive)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--method", required=True,
                    choices=("exact", "diag", "identity", "bgd", "sgd", "ls"))
    ap.add_argument("--class-idx", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_key = SITES[args.site][0]
    arr = np.load(features_path(model_key))
    X32 = arr[f"train_{args.site}"]
    y = arr["y_train"].astype(np.int32)
    X = np.asarray(X32, dtype=np.float64)
    del X32, arr
    gc.collect()

    _ = run_method(args.method, X, y, args.class_idx)      # warm-up
    del _
    gc.collect()

    times = []
    out = None
    with PeakRSSMonitor(interval=0.005) as mon:
        for _r in range(args.repeats):
            gc.collect()
            t0 = time.perf_counter()
            out = run_method(args.method, X, y, args.class_idx)
            times.append(time.perf_counter() - t0)

    y_c = (y == args.class_idx).astype(np.int32)
    result = dict(
        site=args.site, model=model_key, method=args.method, class_idx=args.class_idx,
        d=int(X.shape[1]), n=int(X.shape[0]), repeats=int(args.repeats),
        times=times,
        time_median=float(np.median(times)),
        time_min=float(np.min(times)),
        time_max=float(np.max(times)),
        peak_rss_delta_bytes=int(mon.delta),
        baseline_rss_bytes=int(mon.baseline),
        timings=out.get("timings", {}),
        info=out.get("info", {}),
        train_accuracy=accuracy(X, y_c, out["w"], out["b"]),
        probe_norm=float(np.linalg.norm(out["w"])),
        env=dict(cpu_count=os.cpu_count(),
                 omp=os.environ.get("OMP_NUM_THREADS"),
                 mkl=os.environ.get("MKL_NUM_THREADS"),
                 openblas=os.environ.get("OPENBLAS_NUM_THREADS")),
    )
    save_json(args.out, result)
    print(json.dumps({"site": args.site, "method": args.method,
                      "median_s": result["time_median"],
                      "peak_mb": result["peak_rss_delta_bytes"] / 2 ** 20}))


if __name__ == "__main__":
    main()
