"""Shared similarity metrics: linear CKA, SVCCA, Spearman ρ/p, and normalization.

Shared by synthetic_separability and other add-on experiments.
"""
import math

import numpy as np


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def linear_cka(X, Y):
    """Linear CKA (Kornblith 2019 definition): ||YᵀX||_F² / (||XᵀX||_F ||YᵀY||_F)."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    ytx = Yc.T @ Xc
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(np.sum(ytx ** 2) / (den + 1e-12))


def svcca_score(X, Y, var=0.99):
    """SVCCA (Raghu 2017): SVD truncated at 99% variance -> CCA -> mean correlation."""
    def proj(A):
        A = A - A.mean(0)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(np.searchsorted(e, var) + 1)
        k = max(1, min(k, len(S)))
        return U[:, :k]
    Qx, _ = np.linalg.qr(proj(X))
    Qy, _ = np.linalg.qr(proj(Y))
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(np.mean(s))


def spearman_p(a, b):
    """Spearman ρ + two-sided p-value (t approximation)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n < 3:
        return 0.0, 1.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    rho = float(ra @ rb / den) if den > 0 else 0.0
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), n - 2))
    except Exception:
        p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
    return rho, p
