"""Probe evaluation.

Shared parts: facet label construction, per-batch xi/correctness accounting, the
main evaluation over the selected samples, and result aggregation.
Task-specific parts:
  - selection rule: regression picks in two stages by |y-thr|, classification picks
    by whether the predicted class is correct;
  - final-layer (full-network) accuracy: regression thresholds the output,
    classification uses one-vs-rest.
"""
import os

import numpy as np
import tensorflow as tf

from .cache import load_layer_outputs_and_labels


# ------------------------------------------------------------------ facet labels
def evalu_prepare(Y, n=9, grouping="quantile"):
    """Build n binary facets.

    grouping="quantile": thresholds cut at the (j+1)/(n+1) quantile of Y (regression)
    grouping="class"   : Y == j is the positive class (classification, one-vs-rest)
    Returns (threshold_list, Y_list); for classification threshold_list holds class ids.
    """
    Y = np.asarray(Y).reshape(-1)
    N = len(Y)
    Y_list = []

    if grouping == "quantile":
        sortY = np.sort(Y)
        NN = int(N / (n + 1))
        threshold_list = []
        for j in range(n):
            nY = float(sortY[(j + 1) * NN])
            threshold_list.append(nY)
            Y_ = np.zeros(N, dtype=np.float32)
            Y_[np.where(Y < nY)[0]] = 1
            Y_list.append(Y_)
        return threshold_list, Y_list

    threshold_list = list(range(n))
    for j in range(n):
        Y_ = np.zeros(N, dtype=np.float32)
        Y_[np.where(Y == j)[0]] = 1
        Y_list.append(Y_)
    return threshold_list, Y_list


def evalu_stream_no_index(X_chunk, Y_list, w_l_list, b_list):
    """Compute xi = (2Y-1)(x.w+b) and the accuracy for every facet."""
    n = len(Y_list)
    correct = np.zeros(n)
    xi_list = []

    for j in range(n):
        logits = np.dot(X_chunk[j], w_l_list[j]) + b_list[j]
        xi = (2 * Y_list[j] - 1) * logits
        correct[j] = np.sum(xi >= 0) / len(xi)
        xi_list.append(xi)

    return xi_list, correct


# ------------------------------------------------------------------ main evaluation
def normalize_select(sel):
    """Normalize select (ndarray/list/dict/set) into set[int]."""
    if sel is None:
        return set()
    if isinstance(sel, set):
        return sel
    if isinstance(sel, (list, tuple, np.ndarray)):
        return set(map(int, sel))
    if isinstance(sel, dict):
        idx = []
        for v in sel.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                idx.extend(v)
            else:
                idx.append(v)
        return set(map(int, idx))
    raise TypeError(f"Unsupported select type: {type(sel)}")


def evalu_stream_main_selected(layer_list, Y, eva_w, eva_b, select_list, save_dir,
                               n=9, grouping="quantile"):
    """Restrict samples to select_list, then compute per-layer accuracy over the n facets; returns (len(layer_list), n)."""
    threshold_list, Y_list = evalu_prepare(Y, n=n, grouping=grouping)
    select_set = [normalize_select(sel) for sel in select_list]

    correct = np.zeros((len(layer_list), n))

    for i, layer_id in enumerate(layer_list):
        print(f"Layer {i}")
        w_list = eva_w[i]
        b_list = eva_b[i]

        t = 0
        correct_i = [0.0 for _ in range(n)]

        for batch_id, (X_chunk, Y_chunk) in enumerate(
            load_layer_outputs_and_labels(layer_id, save_dir=save_dir)
        ):
            batch_size = len(Y_chunk)
            start = batch_id * batch_size
            end = start + batch_size
            global_idx = np.arange(start, end)

            X_sel_list, Y_sel_list = [], []
            for j in range(n):
                mask = np.array([idx in select_set[j] for idx in global_idx], dtype=bool)
                X_sel_list.append(X_chunk[mask])
                Y_sel_list.append(Y_list[j][global_idx[mask]])

            if all(len(x) == 0 for x in X_sel_list):
                continue

            _, correct_batch = evalu_stream_no_index(X_sel_list, Y_sel_list, w_list, b_list)
            correct_i = correct_i + correct_batch
            t += 1

        correct[i] = correct_i / t
        print("accuracy:", correct[i])

    return correct


# ------------------------------------------------------------------ sample selection
def select_indices_by_pred_threshold(labels, values, preds, threshold, y_true,
                                     pro_1=0.05, pro_2=0.25):
    """Regression selection: drop the pro_1 fraction with the smallest |y-thr|, then take the pro_2 fraction with the smallest xi."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    values = np.asarray(values).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if not (len(labels) == len(values) == len(preds) == len(y_true)):
        raise ValueError(
            f"length mismatch: labels={len(labels)}, values={len(values)}, "
            f"preds={len(preds)}, y_true={len(y_true)}")

    thr = float(threshold)
    values_1 = np.abs(y_true - thr)

    def _two_stage_pick(mask):
        idx = np.where(mask)[0]
        num = len(idx)
        if idx.size == 0:
            return []
        start = int(num * pro_1)
        idx_s1 = idx[np.argsort(values_1[idx], kind="mergesort")][start:]
        if idx_s1.size == 0:
            return []
        k = int(num * pro_2)
        idx_s2 = idx_s1[np.argsort(values[idx_s1], kind="mergesort")]
        return idx_s2[:k].tolist()

    result = {0: [], 1: []}
    result[0] = _two_stage_pick((labels == 0) & (preds >= thr))
    result[1] = _two_stage_pick((labels == 1) & (preds < thr))
    return result


def select_indices_by_pred_match(labels, values, preds_argmax, y_true,
                                 n0_cap=3000, n1_cap=1000):
    """Classification selection: among samples whose predicted class equals the true class, take the first n0/n1 in ascending xi."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    values = np.asarray(values).reshape(-1)
    preds = np.asarray(preds_argmax).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if not (len(labels) == len(values) == len(preds)):
        raise ValueError(
            f"length mismatch: labels={len(labels)}, values={len(values)}, preds={len(preds)}")

    result = {0: [], 1: []}

    m0 = (labels == 0) & (preds == y_true)
    idx0 = np.where(m0)[0]
    result[0] = idx0[np.argsort(values[idx0])][0:n0_cap].tolist()

    m1 = (labels == 1) & (preds == y_true)
    idx1 = np.where(m1)[0]
    result[1] = idx1[np.argsort(values[idx1])][0:n1_cap].tolist()
    return result


def _reshape_for_pred_model(X_chunk, pred_model):
    """A 4-D input layer (convolutional paths are saved flattened) must be restored to (B, H, W, C)."""
    if len(pred_model.input_shape) == 4:
        return X_chunk.reshape((-1,) + tuple(pred_model.input_shape[1:]))
    return X_chunk


def evalu_select(layer_name, w, b, Y, pred_model, save_dir, n=9, grouping="quantile"):
    """Pick the "hardest/easiest" sample sets on the base samples so all later perturbation conditions can reuse them.

    layer_name/w/b must be the layer that matches the truncated output of pred_model.
    """
    threshold_list, Y_list = evalu_prepare(Y, n=n, grouping=grouping)
    task = "regression" if grouping == "quantile" else "classification"

    xi_list = [[] for _ in range(n)]
    pred_list = [[] for _ in range(n)]

    w_list = np.asarray(w)
    b_list = np.asarray(b)

    for batch_id, (X_chunk, Y_chunk) in enumerate(
        load_layer_outputs_and_labels(layer_name, save_dir=save_dir)
    ):
        X_chunk = np.asarray(X_chunk, dtype=np.float32)
        B = X_chunk.shape[0]
        start = batch_id * B
        end = start + B

        X_sel_list = [X_chunk] * n
        Y_sel_list = [Y_list[k][start:end] for k in range(n)]

        xi_batch_list, _ = evalu_stream_no_index(X_sel_list, Y_sel_list, w_list, b_list)
        for k in range(n):
            xi_list[k].append(np.asarray(xi_batch_list[k]).reshape(-1))

        X_pred = _reshape_for_pred_model(X_chunk, pred_model)
        pred_out = pred_model(X_pred, training=False)

        if task == "regression":
            prob = tf.squeeze(pred_out, axis=-1).numpy()
        else:
            prob = pred_out.numpy()
        for k in range(n):
            pred_list[k].append(prob)

    select_list = []
    for k in range(n):
        labels_k = np.asarray(Y_list[k]).astype(int).reshape(-1)
        values_k = np.concatenate(xi_list[k], axis=0)
        preds_k = np.concatenate(pred_list[k], axis=0)

        if task == "regression":
            select = select_indices_by_pred_threshold(
                labels_k, values_k, preds_k, threshold_list[k], Y)
        else:
            select = select_indices_by_pred_match(
                labels_k, values_k, preds_k.argmax(axis=1), Y)
        select_list.append(select)

    return select_list


# ------------------------------------------------------------------ full-network (final layer) accuracy
def load_x_from_any(data_or_path, split="train", mmap=True):
    if isinstance(data_or_path, np.ndarray):
        return data_or_path
    if isinstance(data_or_path, dict):
        return data_or_path[f"x_{split}"]
    if isinstance(data_or_path, (str, os.PathLike)):
        obj = np.load(str(data_or_path), allow_pickle=True, mmap_mode="r" if mmap else None)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            return obj.item()[f"x_{split}"]
        return obj
    raise TypeError(f"Unsupported input type: {type(data_or_path)}")


def _sel_to_indices(sel, N):
    arr = np.asarray(sel)
    if arr.ndim == 1 and arr.shape[0] == N and arr.dtype != object:
        mask = arr.astype(bool) if arr.dtype == bool else (arr.astype(np.int8) != 0)
        return np.flatnonzero(mask).astype(np.int64, copy=False)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        idx = arr.astype(np.int64, copy=False)
        if idx.size > 0 and (idx.min() < 0 or idx.max() >= N):
            raise IndexError(f"indices out of range [0,{N - 1}]")
        return idx
    raise ValueError(f"Unsupported sel shape={arr.shape}, dtype={arr.dtype}")


def eval_acc_select_list_single_thresholds(model, data_or_path, split, select_list,
                                           thresholds_list, batch_size=256,
                                           mmap=True, prefetch=tf.data.AUTOTUNE):
    """Regression full-network accuracy: each facet is judged 0/1 with the same threshold; returns the per-k accuracy array."""
    X = load_x_from_any(data_or_path, split=split, mmap=mmap)
    N = int(X.shape[0])

    if len(thresholds_list) != len(select_list):
        raise ValueError("thresholds_list must have the same length as select_list.")

    thr_list = []
    for t in thresholds_list:
        t = tf.convert_to_tensor(tf.cast(t, tf.float32))
        if t.shape.rank != 0:
            t = tf.reshape(t, [])
        thr_list.append(t)

    def make_ds(indices_np):
        indices_np = indices_np.astype(np.int64, copy=False)

        def gen():
            for idx in indices_np:
                yield X[idx]

        ds = tf.data.Dataset.from_generator(
            gen, output_signature=tf.TensorSpec(shape=X.shape[1:], dtype=tf.as_dtype(X.dtype)))
        return ds.batch(batch_size, drop_remainder=False).prefetch(prefetch)

    per_k = []
    for k, (sel_dict, thr) in enumerate(zip(select_list, thr_list)):
        idx0 = _sel_to_indices(sel_dict[0], N)
        idx1 = _sel_to_indices(sel_dict[1], N)

        k_c0 = k_c0_n = k_c1 = k_c1_n = 0

        if idx0.size > 0:
            for xb in make_ds(idx0):
                pred = tf.squeeze(model(xb, training=False), axis=-1)
                ok = tf.greater_equal(pred, thr)
                k_c0 += int(tf.reduce_sum(tf.cast(ok, tf.int32)).numpy())
                k_c0_n += int(ok.shape[0])

        if idx1.size > 0:
            for xb in make_ds(idx1):
                pred = tf.squeeze(model(xb, training=False), axis=-1)
                ok = tf.less(pred, thr)
                k_c1 += int(tf.reduce_sum(tf.cast(ok, tf.int32)).numpy())
                k_c1_n += int(ok.shape[0])

        total_correct = k_c0 + k_c1
        total_count = k_c0_n + k_c1_n
        per_k.append(total_correct / max(total_count, 1))

    return np.array(per_k)


def per_group_ovr_accuracy(model, x, y_true_multiclass, dict_list, batch_size=256,
                           return_target_classes=False):
    """Classification full-network accuracy: group g treats class g as positive in a one-vs-rest decision."""
    y = tf.convert_to_tensor(y_true_multiclass)
    y_int = tf.argmax(y, axis=1, output_type=tf.int32) if len(y.shape) == 2 else tf.cast(y, tf.int32)
    N = int(tf.shape(y_int)[0])

    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    preds = [tf.argmax(model(xb, training=False), axis=1, output_type=tf.int32) for xb in ds]
    y_pred_int = tf.concat(preds, axis=0)

    accs = []
    targets = list(range(len(dict_list)))
    t_k = 0

    for g in dict_list:
        idx0 = np.asarray(g[0], dtype=np.int64)
        idx1 = np.asarray(g[1], dtype=np.int64)
        idx0 = idx0[(idx0 >= 0) & (idx0 < N)]
        idx1 = idx1[(idx1 >= 0) & (idx1 < N)]

        if idx0.size == 0 and idx1.size == 0:
            accs.append(np.nan)
            targets.append(-1)
            continue
        if idx1.size == 0:
            accs.append(np.nan)
            targets.append(-1)
            continue

        target = targets[t_k]
        t_k += 1

        correct = total = 0
        p1 = tf.gather(y_pred_int, idx1).numpy()
        correct += int(np.sum(p1 == target))
        total += int(idx1.size)

        p0 = tf.gather(y_pred_int, idx0).numpy()
        correct += int(np.sum(p0 != target))
        total += int(idx0.size)

        accs.append(correct / total if total > 0 else np.nan)

    accs = np.asarray(accs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)

    if return_target_classes:
        return accs, targets
    return accs


# ------------------------------------------------------------------ aggregation
def compute_stats(arr):
    """Average the 9 accuracy columns over 3 groups and report mean/std."""
    if arr.shape[1] != 9:
        raise ValueError("Input array must have 9 columns")

    stats_m = np.zeros((arr.shape[0], 3))
    stats_s = np.zeros((arr.shape[0], 3))
    for g in range(3):
        block = arr[:, g * 3:(g + 1) * 3]
        stats_m[:, g] = block.mean(axis=1)
        stats_s[:, g] = block.std(axis=1)
    return stats_m, stats_s


def stats_minmax_from_matrix_dict(matrix_dict, check_num=6):
    """Compute row-wise mean/std/min/max for every condition matrix."""
    if check_num is not None and len(matrix_dict) != check_num:
        raise ValueError(f"Expected {check_num} matrices, but received {len(matrix_dict)}")

    shapes = [np.asarray(v).shape for v in matrix_dict.values()]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent matrix shapes: {shapes}")

    stats = {}
    for name, X in matrix_dict.items():
        X = np.asarray(X)
        stats[name] = {
            "mean": X.mean(axis=1),
            "std": X.std(axis=1),
            "min": X.min(axis=1),
            "max": X.max(axis=1),
        }
    return stats
