"""Shared add_exp module (pure NumPy/CPU): data, MD-LP probes, logistic-regression probes,
selection and noise.

These experiments only construct and evaluate a single binary probe; the cost of
backbone training, feature extraction and noise generation is not included.
"""
import gc
import json
import math
import pickle
import threading
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ paths
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = REPO_ROOT / "dataset" / "features"

SEED_BASE = 20261108
RECOND = 1e-12


# ------------------------------------------------------------------ data
# ------------------------------------------------------------------ basic utilities
def to_py(obj):
    """Recursively convert numpy scalars/arrays into JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(obj), f, ensure_ascii=False, indent=2)


def norm_probe(w, b):
    """Normalize (w, b) by ||w||; a zero direction is returned unchanged and flagged."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(w))
    if not np.isfinite(n) or n <= 0:
        return w, float(b), 0.0
    return w / n, float(b) / n, n


def predict(X, w, b):
    """Deterministic decision rule: logit >= 0 is assigned to class 1."""
    return (np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
            + float(b) >= 0.0).astype(np.int32)


def accuracy(X, y, w, b, idx=None):
    y = np.asarray(y).reshape(-1)
    if idx is not None:
        idx = np.asarray(idx, dtype=np.int64)
        X = np.asarray(X)[idx]
        y = y[idx]
    p = predict(X, w, b)
    return float(np.mean(p == y)) if len(y) else float("nan")


def signed_margin(X, y, w, b):
    """xi = (2y-1)(w^T x + b); larger values are easier (correct side)."""
    y = np.asarray(y).reshape(-1)
    logit = np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + float(b)
    return (2.0 * y - 1.0) * logit


# ------------------------------------------------------------------ MD-LP weights
def _moments(X, y, positive=1):
    """Per-class first/second central moments (no sample-pair enumeration, no retained
    copy of the data)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    Xp = X[y == positive]          # fancy index -> copy, so it can be centered in place
    Xn = X[y != positive]
    if len(Xp) == 0 or len(Xn) == 0:
        raise ValueError("both classes must be non-empty")
    I = float(len(Xp))
    J = float(len(Xn))
    mu_p = Xp.mean(axis=0)
    mu_n = Xn.mean(axis=0)
    Xp -= mu_p                     # center in place to avoid another allocation
    Xn -= mu_n
    Cp = Xp.T @ Xp
    Cn = Xn.T @ Xn
    del Xp, Xn
    delta = mu_p - mu_n
    s = I * J * delta
    return dict(mu_p=mu_p, mu_n=mu_n, Cp=Cp, Cn=Cn,
                I=I, J=J, delta=delta, s=s)


def _exact_bias_from_projection(proj, y, positive=1):
    """MD-LP exact bias: midpoint of the I-th and (I+1)-th order statistics of q=-w^T x."""
    q = -np.asarray(proj, dtype=np.float64).reshape(-1)
    qs = np.sort(q)
    I = int(np.sum(np.asarray(y).reshape(-1) == positive))
    lo = float(qs[I - 1])
    hi = float(qs[I]) if I < len(qs) else lo
    return 0.5 * (lo + hi), (lo, hi)


def _pseudo_inverse_solve(G, s, rcond=RECOND):
    """Truncated pseudo-inverse solve of a symmetric PSD matrix.
    Returns (omega, evals, evecs, rank, lmax, lmin_kept)."""
    evals, evecs = np.linalg.eigh(G)
    lmax = float(np.max(evals))
    if lmax <= 0:
        return np.zeros_like(s), evals, evecs, 0, lmax, 0.0
    thr = max(rcond * lmax, np.finfo(np.float64).eps * lmax)
    keep = evals > thr
    rank = int(np.sum(keep))
    if rank == 0:
        return np.zeros_like(s), evals, evecs, 0, lmax, 0.0
    V = evecs[:, keep]
    lam = evals[keep]
    omega = V @ ((V.T @ s) / lam)
    return omega, evals, evecs, rank, lmax, float(np.min(lam))


def compute_mdlp(X, y, mode="exact", positive=1, rcond=RECOND, return_parts=False):
    """Construct a single MD-LP probe (weights + exact bias).

    mode: exact / diag / identity
    Returns dict(w, b, w_raw, b_raw, raw_norm, info, timings, parts)
    """
    t0 = time.perf_counter()
    m = _moments(X, y, positive=positive)
    t_mom = time.perf_counter() - t0

    I, J, delta, s = m["I"], m["J"], m["delta"], m["s"]
    d = len(delta)

    t1 = time.perf_counter()
    info = {"mode": mode, "n": int(I + J), "d": int(d), "I": int(I), "J": int(J)}
    G = None
    D = None
    if mode == "exact":
        G = J * m["Cp"] + I * m["Cn"] + (I * J) * np.outer(delta, delta)
        del m["Cp"], m["Cn"]
        t_asm = time.perf_counter() - t1
        t2 = time.perf_counter()
        omega, evals, evecs, rank, lmax, lmin = _pseudo_inverse_solve(G, s, rcond=rcond)
        t_solve = time.perf_counter() - t2
        info.update(dict(rank=rank, lmax=lmax, lmin_kept=lmin,
                         cond=(lmax / lmin if lmin > 0 else float("inf"))))
        if return_parts:
            info["_G"] = G
            info["_evals"] = evals
    elif mode == "diag":
        D = J * np.diag(m["Cp"]) + I * np.diag(m["Cn"]) + (I * J) * delta ** 2
        D = np.maximum(D, 0.0)
        t_asm = time.perf_counter() - t1
        t2 = time.perf_counter()
        thr = max(rcond * float(np.max(D)) if np.max(D) > 0 else 0.0, 1e-300)
        omega = np.divide(s, D, out=np.zeros_like(s), where=D > thr)
        t_solve = time.perf_counter() - t2
        nz = D[D > thr]
        info.update(dict(rank=int(len(nz)),
                         cond=(float(np.max(nz) / np.min(nz)) if len(nz) else float("inf"))))
        if return_parts:
            info["_D"] = D
    elif mode == "identity":
        omega = s.copy()
        t_asm = 0.0
        t_solve = 0.0
        info.update(dict(rank=int(np.sum(np.abs(s) > 0)), cond=1.0))
    else:
        raise ValueError(mode)

    # direction consistency (towards the positive class), then the exact bias
    if float(np.dot(omega, s)) < 0:
        omega = -omega
    t3 = time.perf_counter()
    proj = np.asarray(X, dtype=np.float64) @ omega
    b, bracket = _exact_bias_from_projection(proj, y, positive=positive)
    t_bias = time.perf_counter() - t3

    w_n, b_n, raw_norm = norm_probe(omega, b)
    info["b_bracket"] = bracket

    timings = dict(moments=t_mom, assemble=t_asm, solve=t_solve, bias=t_bias,
                   total=time.perf_counter() - t0)
    return dict(w=w_n, b=b_n, w_raw=omega, b_raw=b, raw_norm=raw_norm,
                info=info, timings=timings, parts=m if return_parts else None)


def compute_mdlp_analysis(X, y, positive=1, rcond=RECOND):
    """Compute exact/diag/identity in a single pass and return diagonal-approximation
    accuracy metrics (for analysis)."""
    t0 = time.perf_counter()
    m = _moments(X, y, positive=positive)
    I, J, delta, s = m["I"], m["J"], m["delta"], m["s"]
    d = len(delta)

    G = J * m["Cp"] + I * m["Cn"] + (I * J) * np.outer(delta, delta)
    D = np.maximum(J * np.diag(m["Cp"]) + I * np.diag(m["Cn"]) + (I * J) * delta ** 2, 0.0)
    del m["Cp"], m["Cn"]

    # diagonal-approximation accuracy metrics (computed directly on G/D)
    gnorm2 = float(np.sum(G * G))
    dvec = np.diag(G).copy()
    dnorm2 = float(np.sum(dvec * dvec))
    diag_share = dnorm2 / gnorm2 if gnorm2 > 0 else float("nan")
    offdiag_share = max(0.0, 1.0 - diag_share)
    rel_frob_DG = math.sqrt(offdiag_share)
    trace = float(np.sum(dvec))

    # off-diagonal energy of the correlation matrix (scale-free)
    sd = np.sqrt(np.maximum(dvec, 0.0))
    sd[sd == 0] = 1.0
    R = G / np.outer(sd, sd)
    np.fill_diagonal(R, 0.0)
    mean_abs_offdiag_corr = float(np.sum(np.abs(R)) / max(d * (d - 1), 1))
    del R

    # exact solve
    t1 = time.perf_counter()
    omega_e, evals, evecs, rank, lmax, lmin = _pseudo_inverse_solve(G, s, rcond=rcond)
    del evecs
    t_exact = time.perf_counter() - t1
    evals_sorted = np.sort(evals)
    diag_sorted = np.sort(dvec)
    spec_err = float(np.linalg.norm(diag_sorted - evals_sorted) /
                     (np.linalg.norm(evals_sorted) + 1e-300))
    # eigenvalue effective rank / stable rank
    pos = evals[evals > max(rcond * lmax, 0)]
    stable_rank = float((np.sum(pos) ** 2) / (np.sum(pos ** 2) + 1e-300)) if len(pos) else 0.0

    # diag solve
    thr = max(rcond * float(np.max(D)) if np.max(D) > 0 else 0.0, 1e-300)
    omega_d = np.divide(s, D, out=np.zeros_like(s), where=D > thr)
    nz = D[D > thr]
    cond_D = float(np.max(nz) / np.min(nz)) if len(nz) else float("inf")

    # identity
    omega_i = s.copy()

    # direction consistency
    for name, om in (("exact", omega_e), ("diag", omega_d), ("identity", omega_i)):
        if float(np.dot(om, s)) < 0:
            if name == "exact":
                omega_e = -om
            elif name == "diag":
                omega_d = -om
            else:
                omega_i = -om

    probes = {}
    for name, om in (("exact", omega_e), ("diag", omega_d), ("identity", omega_i)):
        b, bracket = _exact_bias_from_projection(
            np.asarray(X, dtype=np.float64) @ om, y, positive=positive)
        w_n, b_n, raw_norm = norm_probe(om, b)
        probes[name] = dict(w=w_n, b=b_n, w_raw=om, b_raw=b, raw_norm=raw_norm,
                            bracket=bracket)

    # residuals (relative-residual definition from the doc)
    def _residual(om):
        r = G @ om - s
        den = (np.linalg.norm(G, "fro") * np.linalg.norm(om) + np.linalg.norm(s))
        return float(np.linalg.norm(r) / (den + 1e-300))

    res = {
        "exact_vs_G": _residual(omega_e),
        "diag_vs_G": _residual(omega_d),
    }
    rD = D * omega_d - s
    res["diag_vs_D"] = float(np.linalg.norm(rD) /
                             (np.linalg.norm(D) * np.linalg.norm(omega_d) +
                              np.linalg.norm(s) + 1e-300))

    # weight/bias similarity
    we, wd, wi = probes["exact"]["w"], probes["diag"]["w"], probes["identity"]["w"]
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    cos_ed = cosine(we, wd)
    cos_ei = cosine(we, wi)
    cos_di = cosine(wd, wi)
    w_err_d = float(np.linalg.norm(we - wd))
    w_err_i = float(np.linalg.norm(we - wi))
    b_err_d = abs(probes["exact"]["b"] - probes["diag"]["b"])
    b_err_i = abs(probes["exact"]["b"] - probes["identity"]["b"])

    metrics = dict(
        n=int(I + J), d=int(d), I=int(I), J=int(J),
        trace=trace,
        diag_share=diag_share, offdiag_share=offdiag_share, rel_frob_DG=rel_frob_DG,
        mean_abs_offdiag_corr=mean_abs_offdiag_corr,
        eig_spectrum_rel_err=spec_err,
        stable_rank=stable_rank,
        cond_G=(lmax / lmin if lmin > 0 else float("inf")),
        cond_D=cond_D, rank_G=rank, rank_D=int(len(nz)),
        residual=res,
        cos_exact_diag=cos_ed, angle_exact_diag_deg=float(
            math.degrees(math.acos(max(-1.0, min(1.0, cos_ed))))),
        cos_exact_identity=cos_ei, cos_diag_identity=cos_di,
        weight_rel_err_diag=w_err_d, weight_rel_err_identity=w_err_i,
        bias_abs_err_diag=b_err_d, bias_abs_err_identity=b_err_i,
        bias_rel_err_diag=float(b_err_d / (abs(probes["exact"]["b"]) + 1e-300)),
        bias_rel_err_identity=float(b_err_i / (abs(probes["exact"]["b"]) + 1e-300)),
    )
    timings = dict(total=time.perf_counter() - t0, exact_solve=t_exact)

    return dict(probes=probes, metrics=metrics, timings=timings,
                G_norm=math.sqrt(gnorm2), D=D, evals=evals)


# ------------------------------------------------------------------ least squares
def standardize_fit(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def standardize_transform(X, mu, sd):
    return (np.asarray(X, dtype=np.float64) - mu) / sd


def _sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_loss_grad(Z, y_signed, w, b, lam):
    n = Z.shape[0]
    logit = Z @ w + b
    z = y_signed * logit
    loss = float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * lam * np.dot(w, w))
    p = _sigmoid(-z)                      # sigmoid(-y*logit)
    gw = -(Z.T @ (y_signed * p)) / n + lam * w
    gb = -float(np.mean(y_signed * p))
    return loss, gw, gb


def _max_curvature(Z, lam, iters=20, seed=SEED_BASE):
    """Power-iteration estimate of 0.25*lambda_max(Z^T Z/n)+lam, used for a safe step size."""
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(Z.shape[1])
    v /= np.linalg.norm(v) + 1e-300
    lam_max = 0.0
    for _ in range(iters):
        u = Z @ v
        v = Z.T @ u
        lam_max = float(np.linalg.norm(v))
        if lam_max <= 0:
            break
        v /= lam_max
    return 0.25 * lam_max / max(Z.shape[0], 1) + lam


def fit_logistic_bgd(X, y, lam=1e-4, max_iter=3000, tol=1e-5,
                     positive=1, standardize=True):
    """Full-batch gradient descent (BGD) logistic probe.
    Returns (w, b) in the original feature space."""
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    y_signed = np.where(y == positive, 1.0, -1.0)

    t_pre = time.perf_counter()
    if standardize:
        mu, sd = standardize_fit(X)
        Z = standardize_transform(X, mu, sd)
    else:
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = X.copy()
    t_pre = time.perf_counter() - t_pre

    t_lip = time.perf_counter()
    curv = _max_curvature(Z, lam)
    lr = 1.0 / max(curv, 1e-12)
    t_lip = time.perf_counter() - t_lip

    t_train = time.perf_counter()
    w = np.zeros(Z.shape[1], dtype=np.float64)
    b = 0.0
    grad_norm = np.inf
    loss = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        loss, gw, gb = _logistic_loss_grad(Z, y_signed, w, b, lam)
        grad_norm = float(np.sqrt(np.dot(gw, gw) + gb * gb))
        w -= lr * gw
        b -= lr * gb
        if grad_norm < tol:
            break
    t_train = time.perf_counter() - t_train

    t_post = time.perf_counter()
    w_raw = w / sd
    b_raw = b - float(np.sum(w * mu / sd))
    w_n, b_n, raw_norm = norm_probe(w_raw, b_raw)
    t_post = time.perf_counter() - t_post

    info = dict(method="logistic_bgd", iterations=it, grad_norm=grad_norm,
                loss=float(loss), lr=float(lr), lam=float(lam),
                curvature=float(curv), standardize=bool(standardize), n_iter=int(X.shape[0]))
    timings = dict(preprocess=t_pre, curvature=t_lip, train=t_train,
                   post=t_post, total=time.perf_counter() - t0)
    return dict(w=w_n, b=b_n, w_raw=w_raw, b_raw=b_raw, raw_norm=raw_norm,
                info=info, timings=timings)


def fit_logistic_sgd(X, y, lam=1e-4, epochs=30, batch_size=128, lr0=0.05,
                     positive=1, standardize=True, seed=SEED_BASE):
    """Mini-batch SGD logistic probe (fixed number of epochs, cosine learning rate)."""
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    y_signed = np.where(y == positive, 1.0, -1.0)

    t_pre = time.perf_counter()
    if standardize:
        mu, sd = standardize_fit(X)
        Z = standardize_transform(X, mu, sd)
    else:
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = X.copy()
    t_pre = time.perf_counter() - t_pre

    n = Z.shape[0]
    w = np.zeros(Z.shape[1], dtype=np.float64)
    b = 0.0
    rng = np.random.RandomState(seed)
    t_train = time.perf_counter()
    updates = 0
    loss = np.inf
    for ep in range(epochs):
        order = rng.permutation(n)
        frac = ep / max(epochs - 1, 1)
        lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Zb = Z[idx]
            yb = y_signed[idx]
            m = len(idx)
            logit = Zb @ w + b
            z = yb * logit
            loss = float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * lam * np.dot(w, w))
            p = _sigmoid(-z)
            gw = -(Zb.T @ (yb * p)) / m + lam * w
            gb = -float(np.mean(yb * p))
            w -= lr * gw
            b -= lr * gb
            updates += 1
    t_train = time.perf_counter() - t_train

    t_post = time.perf_counter()
    w_raw = w / sd
    b_raw = b - float(np.sum(w * mu / sd))
    w_n, b_n, raw_norm = norm_probe(w_raw, b_raw)
    t_post = time.perf_counter() - t_post

    info = dict(method="logistic_sgd", epochs=int(epochs), updates=int(updates),
                loss=float(loss), batch_size=int(batch_size), lr0=float(lr0),
                lam=float(lam), standardize=bool(standardize), n_iter=int(n))
    timings = dict(preprocess=t_pre, train=t_train, post=t_post,
                   total=time.perf_counter() - t0)
    return dict(w=w_n, b=b_n, w_raw=w_raw, b_raw=b_raw, raw_norm=raw_norm,
                info=info, timings=timings)


def fit_least_squares(X, y, positive=1, standardize=True, rcond=None):
    """Least-squares linear probe with intercept: lstsq([Z,1], y_signed)."""
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    y_signed = np.where(y == positive, 1.0, -1.0)

    t_pre = time.perf_counter()
    if standardize:
        mu, sd = standardize_fit(X)
        Z = standardize_transform(X, mu, sd)
    else:
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = X.copy()
    A = np.empty((Z.shape[0], Z.shape[1] + 1), dtype=np.float64)
    A[:, :-1] = Z
    A[:, -1] = 1.0
    t_pre = time.perf_counter() - t_pre

    t_solve = time.perf_counter()
    coef, residuals, rank, singular = np.linalg.lstsq(A, y_signed, rcond=rcond)
    t_solve = time.perf_counter() - t_solve

    w_z = coef[:-1]
    b_z = float(coef[-1])
    w_raw = w_z / sd
    b_raw = b_z - float(np.sum(w_z * mu / sd))
    w_n, b_n, raw_norm = norm_probe(w_raw, b_raw)
    info = dict(method="least_squares", rank=int(rank),
                n_singular=int(len(singular)),
                cond=float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else float("inf"),
                standardize=bool(standardize), n_iter=int(X.shape[0]))
    timings = dict(preprocess=t_pre, solve=t_solve, total=time.perf_counter() - t0)
    return dict(w=w_n, b=b_n, w_raw=w_raw, b_raw=b_raw, raw_norm=raw_norm,
                info=info, timings=timings)


# ------------------------------------------------------------------ susceptible-sample selection
def select_fragile(X, y, w, b, eligible, frac, positive=1):
    """Among eligible (global indices), take the first floor(frac*n_c) samples of each
    class in ascending within-class signed-margin order."""
    eligible = np.asarray(eligible, dtype=np.int64)
    y = np.asarray(y).reshape(-1)
    mg = signed_margin(np.asarray(X)[eligible], y[eligible], w, b)
    out = {0: np.empty(0, dtype=np.int64), 1: np.empty(0, dtype=np.int64)}
    for c in (0, 1):
        pos = np.where(y[eligible] == c)[0]
        k = int(math.floor(frac * len(pos)))
        if k <= 0:
            out[c] = np.empty(0, dtype=np.int64)
            continue
        order = pos[np.argsort(mg[pos], kind="mergesort")]
        out[c] = eligible[order[:k]]
    return out


def selection_overlap(a, b):
    a = set(int(v) for v in np.asarray(a).reshape(-1))
    b = set(int(v) for v in np.asarray(b).reshape(-1))
    inter = len(a & b)
    union = len(a | b)
    return dict(overlap=inter, n_a=len(a), n_b=len(b),
                jaccard=(inter / union if union else float("nan")),
                frac_a_kept=(inter / len(a) if a else float("nan")))


def selection_stats(sel_ref, sel_other):
    """Report overlap per class and overall."""
    per_class = {}
    all_a, all_b = [], []
    for c in (0, 1):
        per_class[str(c)] = selection_overlap(sel_ref[c], sel_other[c])
        all_a.append(np.asarray(sel_ref[c]).reshape(-1))
        all_b.append(np.asarray(sel_other[c]).reshape(-1))
    per_class["all"] = selection_overlap(np.concatenate(all_a), np.concatenate(all_b))
    return per_class


def topk_overlap_curve(X, y, w_ref, b_ref, w_other, b_other, eligible, fracs):
    """top-k overlap curve: k is the fraction of eligible samples per class."""
    out = []
    for f in fracs:
        a = select_fragile(X, y, w_ref, b_ref, eligible, f)
        b = select_fragile(X, y, w_other, b_other, eligible, f)
        st = selection_stats(a, b)
        out.append(dict(frac=float(f), **st["all"]))
    return out


# ------------------------------------------------------------------ noise
def gaussian_noise(X, a, rng):
    X = np.asarray(X, dtype=np.float32)
    return np.clip(X + np.float32(a) * rng.standard_normal(X.shape).astype(np.float32),
                   0.0, 1.0)


def salt_noise(X, n, rng):
    """Set each pixel of each image to 0/1 with probability n/(H*W)."""
    X = np.asarray(X, dtype=np.float32)
    N, H, W, C = X.shape
    ratio = float(n) / float(H * W)
    mask = rng.random_sample((N, H, W, 1)) < ratio
    vals = rng.randint(0, 2, size=(N, H, W, 1)).astype(np.float32)
    return np.where(mask, vals, X)


NOISE_SPECS = (
    [("gauss", float(a)) for a in (0.1, 0.2, 0.3)] +
    [("salt", int(n)) for n in (32, 64, 128)]
)
NOISE_SEEDS = tuple(range(5))


def make_noise_conditions():
    """Return [(name, level, seed, key)]."""
    out = []
    for name, level in NOISE_SPECS:
        for s in NOISE_SEEDS:
            out.append((name, level, int(s), f"{name}_{level}_s{s}"))
    return out


def apply_noise(X, name, level, seed):
    rng = np.random.RandomState(seed)
    if name == "gauss":
        return gaussian_noise(X, level, rng)
    if name == "salt":
        return salt_noise(X, level, rng)
    raise ValueError(name)


# ------------------------------------------------------------------ peak memory
class PeakRSSMonitor:
    """Sample process RSS in the background; report the peak increase over the baseline."""

    def __init__(self, interval=0.005):
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = None
        self.peak = 0
        self.baseline = 0

    def __enter__(self):
        import psutil
        self._proc = psutil.Process()
        gc.collect()
        self.baseline = int(self._proc.memory_info().rss)
        self.peak = self.baseline

        def _run():
            while not self._stop.is_set():
                try:
                    rss = int(self._proc.memory_info().rss)
                    if rss > self.peak:
                        self.peak = rss
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    @property
    def delta(self):
        return int(self.peak - self.baseline)


# ------------------------------------------------------------------ misc
def bootstrap_indices(eligible, frac=0.8, n_boot=50, seed=SEED_BASE):
    """Subsample eligible without replacement; return a list of index arrays."""
    rng = np.random.RandomState(seed)
    n = len(eligible)
    k = max(2, int(round(frac * n)))
    return [rng.choice(eligible, size=k, replace=False) for _ in range(n_boot)]


def percentile_summary(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(mean=float("nan"), std=float("nan"), median=float("nan"),
                    q25=float("nan"), q75=float("nan"), min=float("nan"), max=float("nan"))
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=0)),
                median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), min=float(np.min(v)), max=float(np.max(v)))
