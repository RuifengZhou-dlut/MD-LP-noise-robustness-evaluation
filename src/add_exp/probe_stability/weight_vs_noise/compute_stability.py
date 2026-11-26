#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validation experiment: MD-LP weight stability after 10 independent noise draws
(exact vs diag).

Matches the 10-repeat configuration of the Section 5.2.2 subsampling experiment:
  - 8 sites, 4 OVR facets;
  - 10 independent draws per noise level rho;
  - for every (site, rho, mode), two quantities are recorded:
      1) pairwise: the 45 pairs cos(w_i,w_j) among the 10 noisy weights (repeatability)
      2) vs clean: cos(w_noisy,w_clean) between the noisy weight and the clean
         full-sample weight (fidelity)
  - then averaged over the 4 facets.

Section 5.2.2 (feature-noise weight stability): outputs go to result/add_exp/probe_stability/weight_vs_noise/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from ...common.mdlp import RECOND, _moments, save_json
from ...common.sites import SITES, features_path, result_dir

RHOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
SITE_ORDER = ["VGG_L3", "VGG_L5", "VGG_L7", "VGG_L9",
              "RES_L3", "RES_L4", "RES_L6", "RES_L8"]


def _parts(m):
    I, J, delta, s = m["I"], m["J"], m["delta"], m["s"]
    G = J * m["Cp"] + I * m["Cn"] + (I * J) * np.outer(delta, delta)
    D = np.maximum(J * np.diag(m["Cp"]) + I * np.diag(m["Cn"])
                   + (I * J) * delta ** 2, 0.0)
    return G, D, s


def _exact_w(G, s):
    evals, evecs = np.linalg.eigh(G)
    lmax = float(evals[-1])
    thr = max(RECOND * lmax, np.finfo(np.float64).eps * lmax)
    keep = evals > thr
    w = evecs[:, keep] @ ((evecs[:, keep].T @ s) / evals[keep])
    if float(w @ s) < 0:
        w = -w
    return w / np.linalg.norm(w)


def _diag_w(D, s, mask=None):
    thr = max(RECOND * float(D.max()), 1e-300)
    m = (D > thr) if mask is None else mask
    w = np.divide(s, D, out=np.zeros_like(s), where=m)
    if float(w @ s) < 0:
        w = -w
    return w / np.linalg.norm(w)


def _solve_w(G, s):
    try:
        w = np.linalg.solve(G, s)
    except np.linalg.LinAlgError:
        return _exact_w(G, s)
    if float(w @ s) < 0:
        w = -w
    return w / np.linalg.norm(w)


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)),
                min=float(a.min()), max=float(a.max()), n=int(a.size))


def run(sites, draws, tag, out_dir):
    out_path = Path(out_dir) / f"noise_pairwise_stability_{tag}.json"
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for site in sites:
        if site in data:
            print(f"[{site}] exists, skip", flush=True)
            continue
        model, layer, d = SITES[site][:3]
        with np.load(features_path(model)) as z:
            X = np.asarray(z[f"train_{site}"], dtype=np.float64)
            y = np.asarray(z["y_train"], dtype=np.int32)
        n = X.shape[0]
        sig = X.std(axis=0, ddof=0)
        print(f"[{site}] d={d} n={n} draws={draws} facets=4", flush=True)

        clean = {}
        for c in range(4):
            m = _moments(X, y, positive=c)
            G, D, s = _parts(m)
            del m
            clean[c] = dict(
                we=_exact_w(G, s),
                wd=_diag_w(D, s),
                mask=D > max(RECOND * float(D.max()), 1e-300),
            )
            del G, D
        print(f"[{site}] clean ready", flush=True)

        rng = np.random.default_rng(20261116 + SITE_ORDER.index(site))
        rows = []
        for rho in RHOS:
            cos_clean = {"exact": [], "diag": []}
            W = {"exact": [[] for _ in range(4)], "diag": [[] for _ in range(4)]}
            for c in range(4):
                for _ in range(draws):
                    Xi = rng.standard_normal((n, d))
                    Xn = X + rho * sig * Xi
                    del Xi
                    m = _moments(Xn, y, positive=c)
                    del Xn
                    G, D, s = _parts(m)
                    del m
                    we = _exact_w(G, s) if d <= 1024 else _solve_w(G, s)
                    wd = _diag_w(D, s, mask=clean[c]["mask"])
                    del G, D
                    cos_clean["exact"].append(float(np.dot(we, clean[c]["we"])))
                    cos_clean["diag"].append(float(np.dot(wd, clean[c]["wd"])))
                    W["exact"][c].append(we)
                    W["diag"][c].append(wd)
                print(f"  [rho {rho:g}] facet {c} done", flush=True)

            pair = {"exact": [], "diag": []}
            for mode in ("exact", "diag"):
                for c in range(4):
                    Wm = np.stack(W[mode][c], axis=0)
                    C = Wm @ Wm.T
                    iu = np.triu_indices(Wm.shape[0], 1)
                    pair[mode].extend(C[iu].tolist())
            rows.append(dict(
                rho=float(rho),
                cos_clean={m: _stats(cos_clean[m]) for m in ("exact", "diag")},
                pairwise={m: _stats(pair[m]) for m in ("exact", "diag")},
            ))
            print(f"  rho={rho*100:g}% clean exact={rows[-1]['cos_clean']['exact']['mean']:.3f} "
                  f"pair={rows[-1]['pairwise']['exact']['mean']:.3f} | "
                  f"diag clean={rows[-1]['cos_clean']['diag']['mean']:.3f} "
                  f"pair={rows[-1]['pairwise']['diag']['mean']:.3f}", flush=True)

        data[site] = dict(d=int(d), draws=int(draws), facets=4, rows=rows)
        save_json(out_path, data)
        print(f"[{site}] saved", flush=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--tag", default="8sites_10draws")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--out-dir", default=str(result_dir("add_exp/probe_stability/weight_vs_noise")))
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run(args.sites or SITE_ORDER, args.draws, args.tag, args.out_dir)


if __name__ == "__main__":
    main()

