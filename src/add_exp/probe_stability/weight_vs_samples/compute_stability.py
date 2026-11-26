#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Section 5.2.2 (subsampling weight stability): stability cosine of 10 repeated weights under random
subsampling (exact vs diag).

Only the weights of the 10 independent draws at the same (site, n/d, mode) are
compared with each other; they are never compared against the full-sample weights.
For each facet the mean of the 45 pairs cos(w_i,w_j) is taken, then averaged over the
4 OVR facets. Output is written to result/add_exp/probe_stability/weight_vs_samples/.

Metrics:
  pairwise_mean           mean of the 45 cos values among the 10 weights (stability, primary metric)
  pairwise_min            worst pair
  rank / log10 cond       numerical state of G (exact) and D (diag)
"""
import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from ...common.mdlp import compute_mdlp_analysis
from ...common.sites import features_path as _features_path
from ...common.sites import result_dir

N_CLASSES = 4
MODES = ("exact", "diag")
DEFAULT_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
                  1.05, 1.1, 1.2, 1.3, 1.5]
SITES = {
    "VGG_L3": ("VGG-11", "re_lu_3", 4096),
    "VGG_L5": ("VGG-11", "re_lu_5", 2048),
    "VGG_L7": ("VGG-11", "re_lu_7", 1024),
    "VGG_L9": ("VGG-11", "re_lu_9", 256),
    "RES_L3": ("Res-18", "re_lu_6", 4096),
    "RES_L4": ("Res-18", "re_lu_8", 4096),
    "RES_L6": ("Res-18", "re_lu_12", 2048),
    "RES_L8": ("Res-18", "re_lu_16", 1024),
}


def _subsample(y, n_per, rng):
    idx = []
    for c in range(N_CLASSES):
        pool = np.where(y == c)[0]
        take = min(n_per, pool.size)
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(idx))


def _seed(seed_base, site, ratio, repeat):
    key = f"{site}|{ratio:.6g}|{repeat}|{seed_base}".encode()
    return (seed_base + zlib.crc32(key)) % (2 ** 32)


def _log10cond(c):
    try:
        c = float(c)
    except Exception:
        return 999.0
    if not np.isfinite(c) or c <= 0:
        return 999.0
    return float(np.log10(c))


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)),
                min=float(a.min()), max=float(a.max()))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_site(site, ratios, repeats, seed_base, data, out_path, overwrite=False):
    model, layer, d = SITES[site]
    print(f"[{site}] loading {model} features (d={d})", flush=True)
    with np.load(_features_path(model)) as z:
        X = np.asarray(z[f"train_{site}"], dtype=np.float32)
        y = np.asarray(z["y_train"], dtype=np.int32)
    print(f"[{site}] X={X.shape} y={np.bincount(y).tolist()}", flush=True)

    entry = data["sites"].setdefault(site, {
        "model": model, "layer": layer, "d": d, "ratios": {}, "full": {},
    })
    entry["model"] = model
    entry["layer"] = layer
    entry["d"] = d

    # ---- full-sample reference weights ----
    full_w = {m: [] for m in MODES}
    full_info = {m: {"rank": [], "log10cond": []} for m in MODES}
    t0 = time.time()
    for c in range(N_CLASSES):
        out = compute_mdlp_analysis(X, y, positive=c)
        for m in MODES:
            full_w[m].append(np.asarray(out["probes"][m]["w"], dtype=np.float64))
            if m == "exact":
                full_info[m]["rank"].append(int(out["metrics"]["rank_G"]))
                full_info[m]["log10cond"].append(_log10cond(out["metrics"]["cond_G"]))
            else:
                full_info[m]["rank"].append(int(out["metrics"]["rank_D"]))
                full_info[m]["log10cond"].append(_log10cond(out["metrics"]["cond_D"]))
        del out
    entry["full"] = {
        m: {
            "rank_mean": float(np.mean(full_info[m]["rank"])),
            "rank": full_info[m]["rank"],
            "log10cond_mean": float(np.mean(full_info[m]["log10cond"])),
            "log10cond": full_info[m]["log10cond"],
        } for m in MODES
    }
    print(f"[{site}] full probes ready in {time.time()-t0:.1f}s", flush=True)

    for ri, ratio in enumerate(ratios):
        key = f"{ratio:g}"
        if key in entry["ratios"] and not overwrite:
            print(f"[{site}] ratio={key} already exists, skip", flush=True)
            continue
        n_per = max(2, int(round(ratio * d)) // N_CLASSES)
        n_tot = n_per * N_CLASSES
        if n_tot > y.size:
            print(f"[{site}] ratio={key} n={n_tot} > {y.size}, skip", flush=True)
            continue

        t_ratio = time.time()
        # per-facet weight lists: W[mode][facet] -> [w_repeat, ...]
        W = {m: [[] for _ in range(N_CLASSES)] for m in MODES}
        cos_facet = {m: [] for m in MODES}     # per repeat x facet
        rel_facet = {m: [] for m in MODES}
        rank_facet = {m: [] for m in MODES}
        cond_facet = {m: [] for m in MODES}
        cos_rep = {m: [] for m in MODES}       # per repeat, facet-averaged

        for r in range(repeats):
            rng = np.random.default_rng(_seed(seed_base, site, ratio, r))
            idx = _subsample(y, n_per, rng)
            Xs = X[idx]
            ys = y[idx]
            rep_cos = {m: [] for m in MODES}
            for c in range(N_CLASSES):
                out = compute_mdlp_analysis(Xs, ys, positive=c)
                for m in MODES:
                    w = np.asarray(out["probes"][m]["w"], dtype=np.float64)
                    W[m][c].append(w)
                    cos = float(np.dot(w, full_w[m][c]))
                    rel = float(np.linalg.norm(w - full_w[m][c]))
                    cos_facet[m].append(cos)
                    rel_facet[m].append(rel)
                    rep_cos[m].append(cos)
                    if m == "exact":
                        rank_facet[m].append(int(out["metrics"]["rank_G"]))
                        cond_facet[m].append(_log10cond(out["metrics"]["cond_G"]))
                    else:
                        rank_facet[m].append(int(out["metrics"]["rank_D"]))
                        cond_facet[m].append(_log10cond(out["metrics"]["cond_D"]))
                del out
            for m in MODES:
                cos_rep[m].append(float(np.mean(rep_cos[m])))
            del Xs, ys

        row = dict(ratio=float(ratio), n=int(n_tot), n_per_class=int(n_per),
                   n_over_d=float(ratio), n_over_full=float(n_tot / y.size),
                   repeats=int(repeats))
        for m in MODES:
            pair_means = []
            pair_all = []
            for c in range(N_CLASSES):
                Wm = np.stack(W[m][c], axis=0)  # R x d, unit rows
                C = Wm @ Wm.T
                iu = np.triu_indices(len(Wm), 1)
                if len(iu[0]):
                    vals = C[iu]
                    pair_means.append(float(vals.mean()))
                    pair_all.extend(float(x) for x in vals)
            cs = np.asarray(cos_facet[m], dtype=np.float64)
            cr = np.asarray(cos_rep[m], dtype=np.float64)
            rs = np.asarray(rel_facet[m], dtype=np.float64)
            rk = np.asarray(rank_facet[m], dtype=np.float64)
            cd = np.asarray(cond_facet[m], dtype=np.float64)
            row[m] = {
                "cos_mean": float(cr.mean()),
                "cos_std": float(cr.std(ddof=0)),
                "cos_all_mean": float(cs.mean()),
                "cos_all_std": float(cs.std(ddof=0)),
                "cos_min": float(cs.min()),
                "cos_max": float(cs.max()),
                "cos_lt_0.5_frac": float((cs < 0.5).mean()),
                "cos_repeat_means": [float(x) for x in cr],
                "rel_mean": float(rs.mean()),
                "rel_std": float(rs.std(ddof=0)),
                "rel_min": float(rs.min()),
                "rel_max": float(rs.max()),
                "pairwise_mean": float(np.mean(pair_all)) if pair_all else float("nan"),
                "pairwise_std": float(np.std(pair_all, ddof=0)) if pair_all else float("nan"),
                "pairwise_min": float(np.min(pair_all)) if pair_all else float("nan"),
                "pairwise_max": float(np.max(pair_all)) if pair_all else float("nan"),
                "pairwise_n": int(len(pair_all)),
                "pairwise_facet_means": [float(x) for x in pair_means],
                "pairwise_values": [float(x) for x in pair_all],
                "rank_mean": float(rk.mean()),
                "rank_min": float(rk.min()),
                "rank_max": float(rk.max()),
                "log10cond_mean": float(cd.mean()),
                "log10cond_max": float(cd.max()),
            }
        entry["ratios"][key] = row
        save_json(out_path, data)
        dt = time.time() - t_ratio
        print(f"[{site}] ratio={key} n={n_tot} "
              f"exact cos={row['exact']['cos_mean']:.3f} "
              f"diag cos={row['diag']['cos_mean']:.3f} "
              f"exact pair={row['exact']['pairwise_mean']:.3f} "
              f"diag pair={row['diag']['pairwise_mean']:.3f} "
              f"({dt:.0f}s)", flush=True)


def parse_ratios(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["VGG_L3", "VGG_L5", "VGG_L7",
                                                   "VGG_L9", "RES_L3", "RES_L4",
                                                   "RES_L6", "RES_L8"])
    ap.add_argument("--ratios", default=",".join(f"{r:g}" for r in DEFAULT_RATIOS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--tag", default="8sites_10rep")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out-dir", default=str(result_dir("add_exp/probe_stability/weight_vs_samples")))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ratios = parse_ratios(args.ratios)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"small_n_stability_{args.tag}.json"
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        data = {"meta": {}, "sites": {}}
    data["meta"] = {
        "tag": args.tag,
        "sites": args.sites,
        "ratios": ratios,
        "repeats": args.repeats,
        "seed_base": args.seed,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "validation experiment, independent of add_exp",
    }

    t0 = time.time()
    for site in args.sites:
        if site not in SITES:
            raise SystemExit(f"unknown site {site}")
        run_site(site, ratios, args.repeats, args.seed, data, out_path,
                 overwrite=args.overwrite)
    print(f"[done] {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
