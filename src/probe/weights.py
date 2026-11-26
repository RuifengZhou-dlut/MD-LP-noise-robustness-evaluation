"""Streaming solver for the linear probe (w, b).

The two variants differ only in how the positive and negative classes are split:
  - regression     : 9 thresholds cut at label quantiles, mask1 = Y < nY
  - classification : 4 facets for the four-class subset (10 for full CIFAR-10),
    cut class-wise one-vs-rest, mask1 = Y == i
Both are therefore expressed through one `grouping` argument; everything else
(the MD-LP direction and the convex piecewise-linear solution for b) is shared.
"""
import os

import numpy as np
import tensorflow as tf

from .cache import load_layer_outputs_and_labels


def _sum_abs_pairwise(p1, p2):
    """sum_{i,j} |p1_i - p2_j|, computed in O((n1+n2) log n2) via sorting + prefix sums."""
    a = np.sort(np.asarray(p1, dtype=np.float64))
    b = np.sort(np.asarray(p2, dtype=np.float64))
    cb = np.concatenate(([0.0], np.cumsum(b)))
    k = np.searchsorted(b, a, side='left')
    left = a * k - cb[k]
    right = (cb[-1] - cb[k]) - a * (len(b) - k)
    return float(np.sum(left + right))


def make_groups(Y, L, grouping):
    """Return a length-L list of criteria; each item is a function(Y_batch) -> bool mask."""
    Y = np.asarray(Y).reshape(-1)

    if grouping == "quantile":
        sortY = np.sort(Y)
        N = len(Y)
        NN = int(N / (L + 1))
        thresholds = [float(sortY[(i + 1) * NN]) for i in range(L)]
        return thresholds, [lambda Yb, t=t: Yb < t for t in thresholds]

    if grouping == "class":
        thresholds = list(range(L))
        return thresholds, [lambda Yb, c=c: Yb == c for c in thresholds]

    raise ValueError(f"Unsupported grouping: {grouping}")


def W_streaming_with_inline_mdlp_stream(layer_name, Y, save_dir, L=9,
                                      grouping="quantile", return_splits=False):
    """MD-LP direction: J = |sum_{i,j}<m,x1_i-x2_j>| / sum_{i,j}|<m,x1_i-x2_j>|.

    m is the diagonal-inverse approximation m_d ~ dmu_d / dbar_d (dbar_d is the
    mean per-dimension squared difference between sample pairs of the two classes).
    """
    weight_list, splits = [], []
    J = 0

    _, masks = make_groups(Y, L, grouping)

    for i in range(L):
        mask_fn = masks[i]
        print(f"\n==== split {i} ====")

        # Pass 1: first and second moments of the two classes
        sum1 = 0.0
        sum2 = 0.0
        sum1_sq = 0.0
        sum2_sq = 0.0
        n1 = 0
        n2 = 0

        for X_b, Y_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):
            if len(X_b.shape) > 2:
                X_b = X_b.reshape(X_b.shape[0], -1)

            mask1 = mask_fn(Y_b)
            mask2 = ~mask1

            if tf.reduce_any(mask1):
                X1_b = tf.boolean_mask(X_b, mask1)
                sum1 += tf.reduce_sum(X1_b, axis=0)
                sum1_sq += tf.reduce_sum(tf.square(X1_b), axis=0)
                n1 += int(tf.reduce_sum(tf.cast(mask1, tf.int32)))

            if tf.reduce_any(mask2):
                X2_b = tf.boolean_mask(X_b, mask2)
                sum2 += tf.reduce_sum(X2_b, axis=0)
                sum2_sq += tf.reduce_sum(tf.square(X2_b), axis=0)
                n2 += int(tf.reduce_sum(tf.cast(mask2, tf.int32)))

        # Diagonal inverse approximation: w_d ~ dmu_d / dbar_d, where
        #   dbar_d = sum_ij (x_id - y_jd)^2 / (n1*n2) = dmu_d^2 + var_a,d + var_b,d
        # follows from n2*sum(x^2) + n1*sum(y^2) - 2*(sum(x))*(sum(y)) in one pass.
        n_pair = float(n1) * float(n2) + 1e-8
        s1 = np.asarray(sum1, dtype=np.float64)
        s2 = np.asarray(sum2, dtype=np.float64)
        q1 = np.asarray(sum1_sq, dtype=np.float64)
        q2 = np.asarray(sum2_sq, dtype=np.float64)

        num = (n2 * s1 - n1 * s2) / n_pair
        dbar = (n2 * q1 + n1 * q2 - 2.0 * s1 * s2) / n_pair
        # Only clamp negatives caused by floating-point error to zero; no extra regularization.
        dbar = np.maximum(dbar, 0.0)
        m_i = np.divide(num, dbar, out=np.zeros_like(num), where=dbar > 0.0)
        m_i = m_i / (np.linalg.norm(m_i) + 1e-8)
        m_i = m_i.astype(np.float32)

        LL = m_i.shape[0]
        m_np = m_i.reshape(-1)

        # Pass 2: one-dimensional projections
        proj1, proj2 = [], []
        proj_all, sgn_all = [], []

        for X_b, Y_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):
            if len(X_b.shape) > 2:
                X_b = X_b.reshape(X_b.shape[0], -1)

            proj = np.asarray(X_b, dtype=np.float32) @ m_np
            mask1 = mask_fn(Y_b)
            proj1.append(proj[mask1])
            proj2.append(proj[~mask1])
            proj_all.append(proj)
            sgn_all.append(np.where(mask1, -1.0, 1.0).astype(np.float32))

        p1 = np.concatenate(proj1).astype(np.float64)
        p2 = np.concatenate(proj2).astype(np.float64)

        L1_total = float(p1.sum()) * len(p2) - float(p2.sum()) * len(p1)
        L1_abs_total = _sum_abs_pairwise(p1, p2)

        J_i = tf.abs(L1_total) / (L1_abs_total + 1e-8)

        m_weighted = tf.reshape(m_i, [1, LL])
        weight_list.append(tf.reshape(tf.sign(L1_total) * m_weighted, [LL]))

        J += J_i

        if return_splits:
            splits.append((p1, p2, np.concatenate(proj_all), np.concatenate(sgn_all)))

    if return_splits:
        return J.numpy() / L, tf.stack(weight_list).numpy(), splits
    return J.numpy() / L, tf.stack(weight_list).numpy()


def F_streaming(layer_name, Y, save_dir, L=9, grouping="quantile"):
    """Exact solution for the bias b: F(b) is convex and piecewise linear, so the optimum is the midpoint between two neighbouring order statistics."""
    J, w_list, splits = W_streaming_with_inline_mdlp_stream(
        layer_name, Y, save_dir=save_dir, L=L, grouping=grouping, return_splits=True)

    b_list, xi_list = [], []

    for i, (p1, p2, p_all, sgn_all) in enumerate(splits):
        n1 = len(p1)
        brk = np.sort(-np.concatenate([p1, p2]))
        lo = float(brk[n1 - 1])
        hi = float(brk[n1]) if n1 < len(brk) else lo

        b_l = 0.5 * (lo + hi)
        b_list.append(b_l)

        xi = (sgn_all * (p_all + b_l)).astype(np.float32)
        print(f"split {i}: b={b_l:.6f}, xi>=0 count:", int(np.sum(xi >= 0)))
        xi_list.append(xi)

    return w_list, b_list, xi_list


# ------------------------------------------------------------------ cache
def save_wb_per_layer(w, b, lname, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"{lname}_w.npy"), w, allow_pickle=True)
    np.save(os.path.join(cache_dir, f"{lname}_b.npy"), b, allow_pickle=True)


def load_wb_if_exists(Y, layer_list, cache_dir, save_dir, L=9, grouping="quantile"):
    """Load w/b layer by layer; recompute and persist any layer that is missing."""
    eva_w, eva_b = [], []

    for lname in layer_list:
        w_path = os.path.join(cache_dir, f"{lname}_w.npy")
        b_path = os.path.join(cache_dir, f"{lname}_b.npy")

        if not os.path.exists(w_path) or not os.path.exists(b_path):
            w, b, xi = F_streaming(lname, Y, save_dir=save_dir, L=L, grouping=grouping)
            save_wb_per_layer(w, b, lname, cache_dir)
        else:
            w = np.load(w_path, allow_pickle=True)
            b = np.load(b_path, allow_pickle=True)

        eva_w.append(w)
        eva_b.append(b)

    return eva_w, eva_b
