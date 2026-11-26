"""Streaming probe evaluation without persisting intermediate-layer features.

Only the indispensable intermediate quantities are kept; everything else is
computed in a single pass over the data:
  1) base set, pass 1: accumulate the first and second moments of both classes per
     facet -> diagonal inverse direction m, then obtain the bias b from the order
     statistics (same construction as the bias routine in weights.py);
  2) base set, pass 2: project each layer's features onto the facet directions to
     get P (N x n_splits) and the full-network prediction; sample selection and
     base accuracy need only P, no features;
  3) each perturbation condition, 1 pass: project layer by layer and accumulate
     the correctness counts of the selected samples.

On disk only w/b (n_splits x D float32 per layer, MB scale) and the result pkl
remain; the extra memory is one N x n_splits projection matrix per layer plus N
predictions. The accuracy matrix averages within chunks and then over chunks
(chunk size 5000).
"""
import numpy as np
import tensorflow as tf

from .evaluate import select_indices_by_pred_match, select_indices_by_pred_threshold

_CHUNK = 5000          # same chunk size as the cached-output reader, so the accuracy convention is unchanged
_BATCH = 256           # forward batch size under GPU/CPU memory limits


# ------------------------------------------------------------------ model view
def make_view(model, layer_list):
    """A multi-output view emitting every intermediate layer (flattened) and the full-network prediction at once."""
    flat = [tf.keras.layers.Flatten(name=f"flat_{i}_{name}")(model.get_layer(name).output)
            for i, name in enumerate(layer_list)]
    return tf.keras.Model(inputs=model.input, outputs=flat + [model.output])


def _batches(n, batch_size):
    for s in range(0, n, batch_size):
        yield s, min(s + batch_size, n)


def _group_masks(Y, n_splits, grouping, thresholds):
    """Global boolean masks (True = positive class), matching the split of weights.make_groups."""
    Y = np.asarray(Y).reshape(-1)
    if grouping == "quantile":
        return [Y < float(t) for t in thresholds]
    return [Y == k for k in range(n_splits)]


# ------------------------------------------------------------------ direction and bias
def _moments(view, X, Y, n_layers, n_splits, grouping, thresholds, batch_size):
    """Pass 1: stream the first and second moments of the positive/negative class of every facet."""
    N = len(Y)
    acc = None
    Y = np.asarray(Y)

    for s, e in _batches(N, batch_size):
        outs = [o.numpy() for o in view(X[s:e], training=False)][:n_layers]
        if acc is None:
            acc = [dict(s1=np.zeros((n_splits, o.shape[1])),
                        q1=np.zeros((n_splits, o.shape[1])),
                        ts=np.zeros(o.shape[1]),
                        tq=np.zeros(o.shape[1]),
                        n1=np.zeros(n_splits, dtype=np.int64)) for o in outs]

        Yb = Y[s:e]
        if grouping == "quantile":
            order = np.argsort(Yb, kind="mergesort")
            pos = np.searchsorted(Yb[order], np.asarray(thresholds), side="left")
            for i, out in enumerate(outs):
                a = acc[i]
                os_ = out[order]
                cs = np.cumsum(os_, axis=0, dtype=np.float64)
                cq = np.cumsum(np.square(os_, dtype=np.float64), axis=0)
                for k in range(n_splits):
                    if pos[k] > 0:
                        a["s1"][k] += cs[pos[k] - 1]
                        a["q1"][k] += cq[pos[k] - 1]
                    a["n1"][k] += pos[k]
                a["ts"] += cs[-1]
                a["tq"] += cq[-1]
        else:
            for i, out in enumerate(outs):
                a = acc[i]
                a["ts"] += out.sum(axis=0, dtype=np.float64)
                a["tq"] += np.square(out, dtype=np.float64).sum(axis=0)
                for k in range(n_splits):
                    m1 = Yb == k
                    if m1.any():
                        sub = out[m1]
                        a["s1"][k] += sub.sum(axis=0, dtype=np.float64)
                        a["q1"][k] += np.square(sub, dtype=np.float64).sum(axis=0)
                        a["n1"][k] += int(m1.sum())

    # Diagonal inverse direction m_d ~ dmu_d / dbar_d (same formula as weights.py, float64 accumulation only)
    dirs = []
    for a in acc:
        n1 = a["n1"].astype(np.float64)
        n2 = float(N) - n1
        n_pair = n1 * n2 + 1e-8
        s2 = a["ts"][None, :] - a["s1"]
        q2 = a["tq"][None, :] - a["q1"]
        num = (n2[:, None] * a["s1"] - n1[:, None] * s2) / n_pair[:, None]
        dbar = (n2[:, None] * a["q1"] + n1[:, None] * q2 - 2.0 * a["s1"] * s2) / n_pair[:, None]
        dbar = np.maximum(dbar, 0.0)
        m = np.divide(num, dbar, out=np.zeros_like(num), where=dbar > 0.0)
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
        dirs.append(m.astype(np.float32))
    return dirs


def _project(view, X, W_list, batch_size):
    """Project each layer's features onto the given directions: returns P_list (N x K float32 per layer) and the full-network predictions."""
    N = len(X)
    P = [np.zeros((N, w.shape[0]), dtype=np.float32) for w in W_list]
    task_is_reg = len(view.output[-1].shape) == 2 and int(view.output[-1].shape[-1]) == 1
    preds = np.zeros(N, dtype=np.float32) if task_is_reg else np.zeros(N, dtype=np.int32)

    for s, e in _batches(N, batch_size):
        outs = [o.numpy() for o in view(X[s:e], training=False)]
        for i, w in enumerate(W_list):
            P[i][s:e] = outs[i] @ w.T
        out = outs[-1]
        preds[s:e] = np.squeeze(out, axis=-1) if task_is_reg else np.argmax(out, axis=1)
    return P, preds


def _bias_and_sign(P_m, masks, n_splits):
    """Derive the bias b and the direction sign from the projection order statistics (mirrors F_streaming)."""
    b_list, sign = [], []
    for k in range(n_splits):
        p = P_m[:, k].astype(np.float64)
        p1, p2 = p[masks[k]], p[~masks[k]]
        n1 = len(p1)
        L1_total = float(p1.sum()) * len(p2) - float(p2.sum()) * n1
        sign.append(np.sign(L1_total) if L1_total != 0 else 1.0)
        brk = np.sort(-np.concatenate([p1, p2]))
        lo = float(brk[n1 - 1])
        hi = float(brk[n1]) if n1 < len(brk) else lo
        b_list.append(0.5 * (lo + hi))
    return np.asarray(b_list, dtype=np.float64), np.asarray(sign, dtype=np.float32)


def _load_wb(wb_cache, layer_list):
    """Load an existing (w, b) cache; return None when any file is missing."""
    import os
    if wb_cache is None:
        return None
    paths = []
    for name in layer_list:
        paths.append(os.path.join(wb_cache, f"{name}_w.npy"))
        paths.append(os.path.join(wb_cache, f"{name}_b.npy"))
    if not all(os.path.exists(p) for p in paths):
        return None
    w_list = [np.asarray(np.load(os.path.join(wb_cache, f"{n}_w.npy"), allow_pickle=True),
                         dtype=np.float32) for n in layer_list]
    b_list = [np.asarray(np.load(os.path.join(wb_cache, f"{n}_b.npy"), allow_pickle=True),
                         dtype=np.float64).reshape(-1) for n in layer_list]
    return w_list, b_list


def solve_base(view, X, Y, layer_list, n_splits, grouping, thresholds,
               wb_cache=None, batch_size=_BATCH, verbose=True):
    """Solve (w, b) for every layer and return the base-set projections P and the full-network predictions preds.

    On a wb_cache hit, pass 1 (moment statistics) is skipped; pass 2 (projection) is still required.
    Returns (w_list, b_list, P_list, preds).
    """
    n_layers = len(layer_list)

    cached = _load_wb(wb_cache, layer_list)
    if cached is not None:
        w_list, b_list = cached
        P, preds = _project(view, X, w_list, batch_size)
        if verbose:
            print(f"[stream] w/b loaded from {wb_cache}")
        return w_list, b_list, P, preds

    masks = _group_masks(Y, n_splits, grouping, thresholds)
    dirs = _moments(view, X, Y, n_layers, n_splits, grouping, thresholds, batch_size)
    P_m, preds = _project(view, X, dirs, batch_size)

    w_list, b_list, P_list = [], [], []
    for i in range(n_layers):
        b, sgn = _bias_and_sign(P_m[i], masks, n_splits)
        w_list.append(dirs[i] * sgn[:, None])
        b_list.append(b)
        P_list.append(P_m[i] * sgn[None, :])
    if verbose:
        print(f"[stream] base directions solved for {n_layers} layers")
    return w_list, b_list, P_list, preds


# ------------------------------------------------------------------ accuracy
def _chunk_slices(n):
    for s in range(0, n, _CHUNK):
        yield s, min(s + _CHUNK, n)


def base_accuracy(P_list, b_list, Y_list, select_list, n_splits):
    """Base-set accuracy: uses the already computed projections P, so no data reload is needed."""
    N = P_list[0].shape[0]
    sel_masks = [np.zeros(N, dtype=bool) for _ in range(n_splits)]
    for k in range(n_splits):
        sel_masks[k][np.asarray(list(select_list[k][0]) + list(select_list[k][1]), dtype=np.int64)] = True

    Ymat = np.stack([np.asarray(y).reshape(-1) for y in Y_list], axis=1)
    acc = np.zeros((len(P_list), n_splits))
    t = 0
    for s, e in _chunk_slices(N):
        m = np.stack([x[s:e] for x in sel_masks], axis=1)
        if not m.any():
            continue
        for i, P in enumerate(P_list):
            logits = P[s:e] + b_list[i][None, :]
            xi = (2.0 * Ymat[s:e] - 1.0) * logits
            ok = (xi >= 0) & m
            tot = m.sum(axis=0)
            acc[i] += np.divide(ok.sum(axis=0), tot, out=np.zeros(n_splits), where=tot > 0)
        t += 1
    return acc / max(t, 1)


def eval_condition(view, X, Y, w_list, b_list, select_list, Y_list,
                   batch_size=_BATCH):
    """A single perturbation condition: one forward pass; returns (len(w_list), n_splits) accuracy and the full-network predictions."""
    N = len(Y)
    n_splits = len(Y_list)
    sel = [np.zeros(N, dtype=bool) for _ in range(n_splits)]
    for k in range(n_splits):
        sel[k][np.asarray(list(select_list[k][0]) + list(select_list[k][1]), dtype=np.int64)] = True
    Ymat = np.stack([np.asarray(y).reshape(-1) for y in Y_list], axis=1)

    acc = np.zeros((len(w_list), n_splits))
    t = 0
    preds = None
    is_reg = False

    for s, e in _chunk_slices(N):
        m = np.stack([x[s:e] for x in sel], axis=1)
        chunk_acc = np.zeros((len(w_list), n_splits))
        # Run the forward pass even without selected samples: the full-network predictions must cover all samples
        for bs_ in range(s, e, batch_size):
            be = min(bs_ + batch_size, e)
            outs = [o.numpy() for o in view(X[bs_:be], training=False)]
            out_last = outs[-1]
            if preds is None:
                is_reg = len(out_last.shape) == 2 and int(out_last.shape[-1]) == 1
                preds = np.zeros(N, np.float32) if is_reg else np.zeros(N, np.int32)
            preds[bs_:be] = (np.squeeze(out_last, axis=-1) if is_reg
                             else np.argmax(out_last, axis=1))
            if not m.any():
                continue
            mm = m[bs_ - s:be - s]
            ym = Ymat[bs_:be]
            for i, w in enumerate(w_list):
                P = outs[i] @ w.T
                xi = (2.0 * ym - 1.0) * (P + b_list[i][None, :])
                chunk_acc[i] += ((xi >= 0) & mm).sum(axis=0)
        if not m.any():
            continue
        tot = m.sum(axis=0)
        for i in range(len(w_list)):
            acc[i] += np.divide(chunk_acc[i], tot, out=np.zeros(n_splits), where=tot > 0)
        t += 1

    return acc / max(t, 1), preds


# ------------------------------------------------------------------ final layer (full network)
def final_from_preds(task, preds, select_list, thresholds):
    """Compute the last row directly from the full-network predictions (regression = threshold, classification = one-vs-rest)."""
    N = len(preds)
    if task == "regression":
        out = []
        for k, sel in enumerate(select_list):
            thr = float(thresholds[k])
            idx0 = np.asarray(sel[0], dtype=np.int64)
            idx1 = np.asarray(sel[1], dtype=np.int64)
            correct = 0
            if idx0.size:
                correct += int(np.sum(preds[idx0] >= thr))
            if idx1.size:
                correct += int(np.sum(preds[idx1] < thr))
            out.append(correct / max(idx0.size + idx1.size, 1))
        return np.asarray(out, dtype=np.float32)

    accs, t_k = [], 0
    targets = list(range(len(select_list)))
    for sel in select_list:
        idx0 = np.asarray(sel[0], dtype=np.int64)
        idx1 = np.asarray(sel[1], dtype=np.int64)
        idx0 = idx0[(idx0 >= 0) & (idx0 < N)]
        idx1 = idx1[(idx1 >= 0) & (idx1 < N)]
        if idx0.size == 0 or idx1.size == 0:
            accs.append(np.nan)
            continue
        target = targets[t_k]
        t_k += 1
        correct = int(np.sum(preds[idx1] == target)) + int(np.sum(preds[idx0] != target))
        accs.append(correct / (idx0.size + idx1.size))
    return np.asarray(accs, dtype=np.float32)


# ------------------------------------------------------------------ selection
def select_from_base(P_last, b_last, preds, Y, thresholds, Y_list, grouping):
    """Select samples on the base set: final-layer projection plus full-network prediction, same logic as evaluate.evalu_select."""
    n_splits = len(Y_list)
    logits = P_last + b_last[None, :]
    select_list = []
    for k in range(n_splits):
        labels_k = np.asarray(Y_list[k]).astype(int).reshape(-1)
        values_k = (2.0 * labels_k - 1.0) * logits[:, k]
        if grouping == "quantile":
            select_list.append(select_indices_by_pred_threshold(
                labels_k, values_k, preds, thresholds[k], Y))
        else:
            select_list.append(select_indices_by_pred_match(
                labels_k, values_k, np.asarray(preds), Y))
    return select_list
