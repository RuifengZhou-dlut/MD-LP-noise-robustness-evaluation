import numpy as np
import tensorflow as tf
from save_load import load_layer_outputs_and_labels

def evalu_prepare(Y, n=9):
    """Preprocess Y, output thresholds nY and corresponding Y_ for each segment."""
    sortY = tf.sort(Y)
    N = len(Y)

    threshold_list = []
    Y_list = []

    for j in range(n):
        NN = int(N / (n + 1))
        lab = (j + 1) * NN
        nY = sortY[lab]                     # threshold
        threshold_list.append(nY)

        # generate Y_
        mask_1 = np.where(Y < nY)[0]
        Y_ = np.zeros(N, dtype=np.float32)
        Y_[mask_1] = 1
        Y_list.append(Y_)

    return threshold_list, Y_list


def evalu_stream_no_index(X_chunk, Y_list, w_l_list, b_list):
    """
    X_chunk: [B, D]    hidden layer output of current batch
    Y_chunk_index_range: (start, end) global sample range for current batch
    Y_list: length n, each is a global binary label array [N]
    w_l_list: n weights for each layer
    b_list: same as above
    """

    n = len(Y_list)

    correct = np.zeros(n)
    xi_list = []

    for j in range(n):
        # use global index slicing directly, not chunk_indices
        Y_ = Y_list[j]

        w = w_l_list[j]
        b = b_list[j]

        logits = np.dot(X_chunk[j], w) + b
        xi = (2 * Y_ - 1) * logits

        correct[j] = np.sum(xi >= 0) / len(xi)
        xi_list.append(xi)

    return xi_list, correct

def normalize_select(sel):
    """
    Convert sel (could be ndarray / list / dict / set)
    to a unified set[int].
    """
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

def evalu_stream_main_selected(layer_list, Y, eva_w, eva_b, select_list, save_dir, n=9):
    """
    Fully follows the original evalu_stream_main framework:
    - Y is regression label
    - evalu_prepare generates 9 groups of classification labels
    - evalu_stream_no_index handles logits / xi / correct
    - only filters samples within batch by select_list
    """

    # ===== generate labels for 9 classification surfaces =====
    threshold_list, Y_list = evalu_prepare(Y, n=n)

    # convert to set for faster lookup
    select_set = [normalize_select(sel) for sel in select_list]

    correct = np.zeros((len(layer_list),n))

    for i, layer_id in enumerate(layer_list):
        print(f"Layer {i}")

        w_list = eva_w[i]   # [n, D]
        b_list = eva_b[i]   # [n]

        t = 0
        correct_i = [0.0 for i in range(n)]

        # ===== streaming read =====
        for batch_id, (X_chunk, Y_chunk) in enumerate(
            load_layer_outputs_and_labels(layer_id, save_dir=save_dir)
        ):
            batch_size = len(Y_chunk)
            start = batch_id * batch_size
            end   = start + batch_size

            # ===== filter samples for each classification surface =====
            X_sel_list = []
            Y_sel_list = []

            global_idx = np.arange(start, end)

            for j in range(n):
                mask = np.array(
                    [idx in select_set[j] for idx in global_idx],
                    dtype=bool
                )

                X_sel_list.append(X_chunk[mask])
                Y_sel_list.append(Y_list[j][global_idx[mask]])

            # skip if all surfaces have no samples
            if all(len(x) == 0 for x in X_sel_list):
                continue

            # ===== call original evalu_stream_no_index =====
            xi_list, correct_batch = evalu_stream_no_index(
                X_sel_list,
                Y_sel_list,
                w_list,
                b_list
            )

            correct_i = correct_i + correct_batch
            t += 1

        correct[i] = correct_i / t
        print("accuracy:", correct[i])

    return correct


def select_indices_by_pred_threshold(labels, values, preds, threshold, y_true, pro_1=0.05, pro_2=0.25):
    """
    Two-stage filtering (executed independently for each class):
      1) First sort by values_1 = |y_true - y_threshold| in ascending order;
         discard the first [0 : start) after sorting (e.g., start=200).
      2) Among the remaining candidates, sort again by values in ascending order;
         take the first (end-start) items (e.g., 500 items).

    labels: (N,) 0/1
    values: (N,) used for second stage sorting (xi)
    preds : (N,) sigmoid probability, used for pred_threshold comparison to enter candidate pool
    pred_threshold: scalar, classification threshold
    y_true: (N,) true regression label (float)
    y_threshold: scalar, used to construct values_1 = |y_true - y_threshold|
    n_0, n_1: (start, end) two-stage logic corresponds to the effect of original slice [start:end]
    Rules:
      y=0: pred >= pred_threshold to enter candidate pool
      y=1: pred <  pred_threshold to enter candidate pool
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    values = np.asarray(values).reshape(-1)
    preds  = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if not (len(labels) == len(values) == len(preds) == len(y_true)):
        raise ValueError(
            f"length mismatch: labels={len(labels)}, values={len(values)}, "
            f"preds={len(preds)}, y_true={len(y_true)}"
        )

    thr = float(threshold)

    # values_1 = |y_true - y_thr|
    values_1 = np.abs(y_true - thr)

    def _two_stage_pick(mask, pro_1=pro_1, pro_2=pro_2):
        """Two-stage filtering for samples satisfying mask, return final index list."""
        idx = np.where(mask)[0]
        num = len(idx)
        if idx.size == 0:
            return []
        start = int(num*pro_1)
        # ---- Stage 1: sort by values_1 ascending, discard first start items ----
        idx_s1 = idx[np.argsort(values_1[idx], kind="mergesort")]
        idx_s1 = idx_s1[start:]  # discard first start items
        if idx_s1.size == 0:
            return []
        k = int(num*pro_2)
        # ---- Stage 2: among remaining, sort by values ascending, take (end-start) items ----
        #k = max(0, int(end) - int(start))
        idx_s2 = idx_s1[np.argsort(values[idx_s1], kind="mergesort")]
        return idx_s2[:k].tolist()

    result = {0: [], 1: []}

    # y=0: pred >= pred_thr
    m0 = (labels == 0) & (preds >= thr)
    result[0] = _two_stage_pick(m0)

    # y=1: pred < pred_thr
    m1 = (labels == 1) & (preds < thr)
    result[1] = _two_stage_pick(m1)

    return result

def evalu_select(layer_list, Y, eva_w, eva_b, pred_model, save_dir, layer_i=-1, n=9):
    threshold_list, Y_list = evalu_prepare(Y, n=n)

    xi_list   = [[] for _ in range(n)]   # values: used for sorting (xi)
    pred_list = [[] for _ in range(n)]   # preds : sigmoid probability, used for threshold judgment

    i = layer_i
    w_list = np.asarray(eva_w[i])   # [n, D]
    b_list = np.asarray(eva_b[i])   # [n]

    #a = [[50,550], [100,1100], [150,1650], [200,2200], [250,2750], [300,3300], [350,3850], [400,4400], [450,4950]]
    #b = [[450,4950], [400,4400], [350,3850], [300,3300], [250,2750], [200,2200], [150,1650], [100,1100], [50,550]]

    for batch_id, (X_chunk, Y_chunk) in enumerate(load_layer_outputs_and_labels(layer_list[i], save_dir=save_dir)):
        # X_chunk: (B,D)
        X_chunk = np.asarray(X_chunk, dtype=np.float32)
        B, D = X_chunk.shape

        start = batch_id * B
        end   = start + B

        X_sel_list = []
        Y_sel_list = []
        for k in range(n):
            X_sel_list.append(X_chunk)
            Y_sel_list.append(Y_list[k][start:end])

        # still use original evalu_stream_no_index to compute xi (sorting basis)
        xi_batch_list, correct_i_batch = evalu_stream_no_index(X_sel_list, Y_sel_list, w_list, b_list)

        # append xi
        for k in range(n):
            xi_list[k].append(np.asarray(xi_batch_list[k]).reshape(-1))

        # compute preds here (sigmoid probability) as alternative to (values>0) condition
        # logit: (B, n) = (B,D) @ (D,n) + (n,)
        # X_chunk: (B, D)
        #print(X_chunk.shape)
        if len(pred_model.input_shape) == 4:
            X_chunk = X_chunk.reshape(5000, 2, 2, 256)
        pred_prob = pred_model(X_chunk, training=False)  # (B, 1)
        pred_prob = tf.squeeze(pred_prob, axis=-1).numpy()  # (B,)


        for k in range(n):
            pred_list[k].append(pred_prob)

    # concatenate to full length N
    select_list = []
    for k in range(n):
        labels_k = np.asarray(Y_list[k]).astype(int).reshape(-1)
        values_k = np.concatenate(xi_list[k], axis=0)      # (N,)
        preds_k = np.concatenate(pred_list[k], axis=0)
        thr_k    = threshold_list[k]                       # scalar

        select = select_indices_by_pred_threshold(labels_k, values_k, preds_k, thr_k, Y)
        select_list.append(select)

    return select_list

def load_x_from_any(data_or_path, split="train", mmap=True):
    """Support dict dataset / npy(dict or ndarray) / ndarray"""
    if isinstance(data_or_path, np.ndarray):
        return data_or_path
    if isinstance(data_or_path, dict):
        key = f"x_{split}"
        if key not in data_or_path:
            raise KeyError(f"dict does not contain '{key}'")
        return data_or_path[key]
    if isinstance(data_or_path, str):
        obj = np.load(data_or_path, allow_pickle=True, mmap_mode="r" if mmap else None)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            d = obj.item()
            key = f"x_{split}"
            if key not in d:
                raise KeyError(f"{data_or_path} does not contain '{key}'")
            return d[key]
        return obj
    raise TypeError(f"Unsupported input type: {type(data_or_path)}")


def _sel_to_indices(sel, N):
    """
    sel supports:
      - indices: [K]
      - mask:    [N] bool/0-1
    Returns indices (int64)
    """
    arr = np.asarray(sel)

    # mask
    if arr.ndim == 1 and arr.shape[0] == N and arr.dtype != object:
        mask = arr.astype(bool) if arr.dtype == bool else (arr.astype(np.int8) != 0)
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    # indices
    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        idx = arr.astype(np.int64, copy=False)
        # optional out-of-bounds check
        if idx.size > 0 and (idx.min() < 0 or idx.max() >= N):
            raise IndexError(f"indices out of range [0,{N-1}]")
        return idx

    raise ValueError(f"Unsupported sel shape={arr.shape}, dtype={arr.dtype}")


def eval_acc_select_list_single_thresholds(
    model,
    data_or_path,
    split,
    select_list,        # list of dicts: [{0:sel0,1:sel1}, ...] length = n
    thresholds_list,    # list of tf scalars length = n, each thr used for both classes
    batch_size=256,
    mmap=True,
    prefetch=tf.data.AUTOTUNE,
    return_details=True
):
    """
    For each k:
      - use threshold thr[k]
      - use sample set select_list[k][0], select_list[k][1]
      - rules:
          class0: pred >= thr[k] considered correct
          class1: pred <  thr[k] considered correct
    Finally accumulate correct/count from all k to get overall accuracy.

    model output: (B,1)
    """
    X = load_x_from_any(data_or_path, split=split, mmap=mmap)
    N = int(X.shape[0])

    if not isinstance(select_list, (list, tuple)) or len(select_list) == 0:
        raise ValueError("select_list must be a non-empty list of dicts.")
    if len(thresholds_list) != len(select_list):
        raise ValueError("thresholds_list must have the same length as select_list.")

    # thresholds -> tf scalar float32
    thr_list = []
    for t in thresholds_list:
        t = tf.cast(t,dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        if t.shape.rank != 0:
            t = tf.reshape(t, [])
        thr_list.append(t)

    def make_ds(indices_np):
        indices_np = indices_np.astype(np.int64, copy=False)

        def gen():
            for idx in indices_np:
                yield X[idx]

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(shape=X.shape[1:], dtype=tf.as_dtype(X.dtype))
        )
        return ds.batch(batch_size, drop_remainder=False).prefetch(prefetch)

    c0_correct = c0_count = 0
    c1_correct = c1_count = 0

    # optionally track accuracy per k
    per_k = []

    for k, (sel_dict, thr) in enumerate(zip(select_list, thr_list)):
        if not isinstance(sel_dict, dict) or (0 not in sel_dict) or (1 not in sel_dict):
            raise ValueError(f"select_list[{k}] must be a dict containing keys 0 and 1.")

        idx0 = _sel_to_indices(sel_dict[0], N)
        idx1 = _sel_to_indices(sel_dict[1], N)

        k_c0_correct = k_c0_count = 0
        k_c1_correct = k_c1_count = 0

        # --- class 0: pred >= thr ---
        if idx0.size > 0:
            ds0 = make_ds(idx0)
            for xb in ds0:
                pred = model(xb, training=False)      # (B,1)
                pred = tf.squeeze(pred, axis=-1)      # (B,)
                ok = tf.greater_equal(pred, thr)
                k_c0_correct += int(tf.reduce_sum(tf.cast(ok, tf.int32)).numpy())
                k_c0_count   += int(ok.shape[0])

        # --- class 1: pred < thr ---
        if idx1.size > 0:
            ds1 = make_ds(idx1)
            for xb in ds1:
                pred = model(xb, training=False)      # (B,1)
                pred = tf.squeeze(pred, axis=-1)      # (B,)
                ok = tf.less(pred, thr)
                k_c1_correct += int(tf.reduce_sum(tf.cast(ok, tf.int32)).numpy())
                k_c1_count   += int(ok.shape[0])

        # accumulate to global
        c0_correct += k_c0_correct
        c0_count   += k_c0_count
        c1_correct += k_c1_correct
        c1_count   += k_c1_count

        if return_details:
            k_total_correct = k_c0_correct + k_c1_correct
            k_total_count   = k_c0_count + k_c1_count
            per_k.append({
                "k": k,
                "threshold": float(thr.numpy()),
                "class0": {"correct": k_c0_correct, "count": k_c0_count,
                           "acc": (k_c0_correct / max(k_c0_count, 1)) if k_c0_count else np.nan},
                "class1": {"correct": k_c1_correct, "count": k_c1_count,
                           "acc": (k_c1_correct / max(k_c1_count, 1)) if k_c1_count else np.nan},
                "total":  {"correct": k_total_correct, "count": k_total_count,
                           "acc": (k_total_correct / max(k_total_count, 1)) if k_total_count else np.nan},
            })

    total_correct = c0_correct + c1_correct
    total_count   = c0_count + c1_count
    overall_acc   = total_correct / max(total_count, 1)

    per_acc=[p_k['total']['acc'] for p_k in per_k]

    return np.array(per_acc)


def compute_stats(arr):
    if arr.shape[1] != 9:
        raise ValueError("Input array must have 9 columns")

    n_rows = arr.shape[0]
    stats_m = np.zeros((n_rows, 3))
    stats_s = np.zeros((n_rows, 3))
    first_3 = arr[:, :3]
    stats_m[:, 0] = first_3.mean(axis=1)  # mean of first 3 columns
    stats_s[:, 0] = first_3.std(axis=1)  # std of first 3 columns

    middle_3 = arr[:, 3:6]
    stats_m[:, 1] = middle_3.mean(axis=1)  # mean of middle 3 columns
    stats_s[:, 1] = middle_3.std(axis=1)  # std of middle 3 columns

    # last 3 columns (columns 6-8)
    last_3 = arr[:, 6:9]
    stats_m[:, 2] = last_3.mean(axis=1)  # mean of last 3 columns
    stats_s[:, 2] = last_3.std(axis=1)  # std of last 3 columns
    return stats_m, stats_s  # return stats[:, 3:] if std is needed