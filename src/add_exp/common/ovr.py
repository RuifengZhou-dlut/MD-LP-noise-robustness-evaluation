"""Shared helpers for the 4-class OVR setup (probes, OVR prediction,
susceptible-sample selection)."""
import numpy as np

from .sites import N_CLASSES


def ovr_probes(X, y, mode, **kw):
    """Build the c-vs-rest MD-LP probe for every class c; return a list of length
    N_CLASSES."""
    from .mdlp import compute_mdlp
    return [compute_mdlp(X, y, mode=mode, positive=c, **kw)
            for c in range(N_CLASSES)]


def ovr_logits(X, probes, dtype=np.float64):
    """(n, N_CLASSES) OVR discriminant scores."""
    X = np.asarray(X, dtype=dtype)
    cols = [X @ np.asarray(p["w"], dtype=dtype) + dtype(p["b"]) for p in probes]
    return np.stack(cols, axis=1)


def ovr_predict(X, probes):
    return np.argmax(ovr_logits(X, probes), axis=1)


def ovr_accuracy(X, y, probes, dtype=np.float64):
    """4-class argmax accuracy."""
    y = np.asarray(y).reshape(-1)
    return float(np.mean(ovr_predict(X, probes) == y))


def ovr_facet_accuracy(X, y, probes):
    """c-vs-rest accuracy of every facet (length N_CLASSES)."""
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    out = []
    for c, p in enumerate(probes):
        logit = X @ np.asarray(p["w"], dtype=np.float64) + float(p["b"])
        out.append(float(np.mean((logit >= 0).astype(np.int32) == (y == c).astype(np.int32))))
    return out


def ovr_gap(X, probes, dtype=np.float32):
    """top1 - top2 discriminant margin (smaller means more susceptible)."""
    M = ovr_logits(X, probes, dtype=dtype)
    s = np.sort(M, axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float64)


def select_fragile_ovr(X, y, probes, eligible, frac, dtype=np.float32):
    """By OVR top1-top2 margin, take the smallest floor(frac*n_c) samples within each
    true class (global indices)."""
    eligible = np.asarray(eligible, dtype=np.int64)
    y = np.asarray(y).reshape(-1)
    gap = ovr_gap(np.asarray(X, dtype=dtype)[eligible], probes, dtype=dtype)
    y_e = y[eligible]
    out = {c: np.empty(0, dtype=np.int64) for c in range(N_CLASSES)}
    for c in range(N_CLASSES):
        pos = np.where(y_e == c)[0]
        k = int(frac * len(pos))
        if k <= 0:
            continue
        order = pos[np.argsort(gap[pos], kind="mergesort")]
        out[c] = eligible[order[:k]]
    return out
