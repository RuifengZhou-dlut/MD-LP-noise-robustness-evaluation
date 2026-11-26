"""Section 5.2.1 (4-class OVR version): time/memory of single-facet probe construction,
plus the total OVR cost across the four facets.

Each (site, method) runs one representative facet (class 0) in an isolated subprocess;
facet costs are approximately equal, so total OVR time = 4 × single facet
(facets are built sequentially, so peak memory does not accumulate).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from .cfg import ADD_ROOT, EXP2, N_CLASSES, SITES
from ..common.mdlp import save_json

BENCH_DIR = EXP2 / "bench_ovr"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

REPEATS = {"VGG_L3": 2, "VGG_L5": 3, "VGG_L7": 5, "VGG_L9": 5,
           "RES_L3": 2, "RES_L4": 2, "RES_L6": 3, "RES_L8": 5}
METHODS = ("exact", "diag", "identity", "bgd", "sgd", "ls")


def run_one(site, method, repeats, threads=8):
    out = BENCH_DIR / f"{site}__{method}.json"
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads), "NUMEXPR_NUM_THREADS": str(threads)})
    cmd = [sys.executable, "-m", "src.add_exp.probe_efficiency.bench_one_ovr",
           "--site", site, "--method", method, "--class-idx", "0",
           "--repeats", str(repeats), "--out", str(out)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ADD_ROOT), env=env,
                          capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"bench failed: {site}/{method}")
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    res["wall_seconds_total"] = time.perf_counter() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    raw = {}
    for site in args.sites:
        d = SITES[site][2]
        reps = REPEATS.get(site, 3)
        raw[site] = {}
        for method in args.methods:
            print(f"[bench] {site} d={d} {method} repeats={reps} ...", flush=True)
            res = run_one(site, method, reps, threads=args.threads)
            raw[site][method] = res
            print(f"        median={res['time_median']:.4f}s  "
                  f"peak+={res['peak_rss_delta_bytes'] / 2 ** 20:.1f}MB  "
                  f"acc={res['train_accuracy']:.4f}", flush=True)

    summary = {}
    for site in args.sites:
        d = SITES[site][2]
        summary[site] = dict(d=d, model=SITES[site][0], layer=SITES[site][1],
                             n=raw[site][args.methods[0]]["n"], methods={})
        for method in args.methods:
            r = raw[site][method]
            summary[site]["methods"][method] = dict(
                time_median=r["time_median"], time_min=r["time_min"],
                time_max=r["time_max"],
                time_ovr_4facets=r["time_median"] * N_CLASSES,
                peak_mb=r["peak_rss_delta_bytes"] / 2 ** 20,
                baseline_mb=r["baseline_rss_bytes"] / 2 ** 20,
                train_accuracy_facet=r["train_accuracy"],
                timings=r.get("timings", {}), info=r.get("info", {}))
        base_t = summary[site]["methods"].get("diag", {}).get("time_median", np.nan)
        base_m = summary[site]["methods"].get("diag", {}).get("peak_mb", np.nan)
        for method in args.methods:
            m = summary[site]["methods"][method]
            m["time_vs_diag"] = float(m["time_median"] / base_t) if base_t else None
            m["peak_vs_diag"] = float(m["peak_mb"] / base_m) if base_m else None

    save_json(EXP2 / "benchmark_ovr.json",
              dict(raw=raw, summary=summary, repeats=REPEATS, threads=args.threads,
                   n_classes=N_CLASSES))
    print("[done] saved", EXP2 / "benchmark_ovr.json")


if __name__ == "__main__":
    main()
