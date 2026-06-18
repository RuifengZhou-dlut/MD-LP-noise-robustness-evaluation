import numpy as np
import tensorflow as tf
from save_load import load_layer_outputs_and_labels

def evalu_prepare(Y, n=4):
    """Preprocess Y, output threshold nY for each segment and corresponding Y_"""
    sortY = tf.sort(Y)
    N = len(Y)

    #threshold_list = []
    Y_list = []

    for j in range(n):
        #NN = int(N / (n + 1))
        #lab = (j + 1) * NN
        #nY = sortY[lab]                     # threshold
        #threshold_list.append(nY)

        # Generate Y_
        mask_1 = np.where(Y == j)[0]
        Y_ = np.zeros(N, dtype=np.float32)
        Y_[mask_1] = 1
        Y_list.append(Y_)

    return Y_list


def evalu_stream_no_index(X_chunk, Y_list, w_l_list, b_list):
    """
    X_chunk: [B, D]    Hidden layer outputs of current batch
    Y_chunk_index_range: (start, end) Global sample range for current batch
    Y_list: length n, each is a global binary label array [N]
    w_l_list: n weights per layer
    b_list: same as above
    """

    n = len(Y_list)

    correct = np.zeros(n)
    xi_list = []

    for j in range(n):
        # Directly slice using global indices, not chunk_indices
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
    to a unified set[int]
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

def evalu_stream_main_selected(layer_list, Y, eva_w, eva_b, select_list, save_dir, n=4):
    """
    Follow original evalu_stream_main framework exactly:
    - Y is regression label
    - evalu_prepare generates 9 sets of classification labels
    - evalu_stream_no_index handles logits / xi / correct
    - Filter samples within batch by select_list only
    """

    # ===== Generate labels for 9 classification planes =====
    Y_list = evalu_prepare(Y, n=n)

    # Convert to set for faster lookup
    select_set = [normalize_select(sel) for sel in select_list]

    correct = np.zeros((len(layer_list),n))

    for i, layer_id in enumerate(layer_list):
        print(f"Layer {i}")

        w_list = eva_w[i]   # [n, D]
        b_list = eva_b[i]   # [n]

        t = 0
        correct_i = [0.0 for i in range(n)]

        # ===== Streaming read =====
        for batch_id, (X_chunk, Y_chunk) in enumerate(
            load_layer_outputs_and_labels(layer_id, save_dir=save_dir)
        ):
            batch_size = len(Y_chunk)
            start = batch_id * batch_size
            end   = start + batch_size

            # ===== Filter for each classification plane =====
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

            # Skip if no samples in any plane
            if all(len(x) == 0 for x in X_sel_list):
                continue

            # ===== Call original evalu_stream_no_index =====
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


def select_indices_by_pred_threshold(labels, values, preds, Y):
    """
    labels: (N,) 0/1
    values: (N,) for sorting (xi)
    preds : (N,) sigmoid output probabilities for threshold comparison
    threshold: scalar (float / np / tf scalar)
    n_0, n_1: [start, end] slice from sorted candidates
    Rules:
      y=0: only pred >= thr enters candidates
      y=1: only pred < thr enters candidates
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    values = np.asarray(values).reshape(-1)
    preds = np.asarray(preds).argmax(axis=1)
    print(preds)
    if not (len(labels) == len(values) == len(preds)):
        raise ValueError(f"length mismatch: labels={len(labels)}, values={len(values)}, preds={len(preds)}")

    result = {0: [], 1: []}

    # y=0: pred >= thr

    # print((preds == Y.reshape(len(Y))).shape,labels.shape)

    m0 = (labels == 0) & (preds == Y.reshape(len(Y)))
    idx0 = np.where(m0)[0]
    idx0_sorted = idx0[np.argsort(values[idx0])]  # 仍按 values 升序排序
    result[0] = idx0_sorted[0:3000].tolist()

    # y=1: pred < thr
    m1 = (labels == 1) & (preds == Y.reshape(len(Y)))
    idx1 = np.where(m1)[0]
    idx1_sorted = idx1[np.argsort(values[idx1])]
    result[1] = idx1_sorted[0:1000].tolist()

    return result

def evalu_select(layer_list, Y, eva_w, eva_b, pred_model, save_dir, layer_i=-1, n=4):
    Y_list = evalu_prepare(Y, n=n)

    xi_list   = [[] for _ in range(n)]   # values: 用于排序（xi）
    pred_list = [[] for _ in range(n)]   # preds : sigmoid概率，用于阈值判断

    i = layer_i
    w_list = np.asarray(eva_w[i])   # [n, D]
    b_list = np.asarray(eva_b[i])   # [n]


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

        # 仍然用你原来的 evalu_stream_no_index 算 xi（排序依据）
        xi_batch_list, correct_i_batch = evalu_stream_no_index(X_sel_list, Y_sel_list, w_list, b_list)

        # 追加 xi
        for k in range(n):
            xi_list[k].append(np.asarray(xi_batch_list[k]).reshape(-1))

        # 在这里自己算 preds（sigmoid概率），作为替代 (values>0) 的条件
        # logit: (B, n) = (B,D) @ (D,n) + (n,)
        # X_chunk: (B, D)
        #print(X_chunk.shape)

        if len(pred_model.input_shape)==4:
            X_chunk = X_chunk.reshape(5000,2,2,256)

        pred_prob = pred_model(X_chunk, training=False)  # (B, 4)
        pred_prob = pred_prob.numpy()  # (B,4)
        #pred_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
        #pred_model.evaluate(X_chunk,tf.keras.utils.to_categorical(Y_chunk,num_classes=4))


        for k in range(n):
            pred_list[k].append(pred_prob)

    # 拼接成全长 N
    select_list = []
    for k in range(n):
        labels_k = np.asarray(Y_list[k]).astype(int).reshape(-1)
        values_k = np.concatenate(xi_list[k], axis=0)      # (N,)
        preds_k = np.concatenate(pred_list[k], axis=0)
        #thr_k    = threshold_list[k]                       # 标量

        select = select_indices_by_pred_threshold(labels_k, values_k, preds_k, Y)
        select_list.append(select)

    return select_list

def per_group_ovr_accuracy(
    model,
    x,
    y_true_multiclass,
    dict_list,
    batch_size=256,
    return_target_classes=False,
):
    """
    Compute one-vs-rest accuracy for each group based on dict_list (length=4, each element like {0:[...], 1:[...]}):
      - Samples in class 1: correct only if prediction == target_class
      - Samples in class 0: correct only if prediction != target_class

    target_class is automatically inferred from the true labels of class 1 samples in y_true_multiclass (default: unique value/mode).

    Parameters
    ----------
    model : tf.keras.Model
    x : np.ndarray or tf.Tensor, shape (N, ...)
    y_true_multiclass : np.ndarray or tf.Tensor, shape (N,) or (N,C)
        Multiclass true labels (integer or one-hot), used to infer target_class
    dict_list : list[dict]
        Length 4; each dict must contain keys 0 and 1, corresponding to sample index lists
    batch_size : int
    return_target_classes : bool
        Whether to also return target_class for each group (for verification)

    Returns
    -------
    accs : np.ndarray, shape (len(dict_list),)
        One-vs-rest accuracy for each dict
    (optional) targets : np.ndarray, shape (len(dict_list),)
        Inferred target_class for each group
    """

    # ---- Handle true labels as integers (N,) ----
    y = tf.convert_to_tensor(y_true_multiclass)
    if len(y.shape) == 2:  # one-hot
        y_int = tf.argmax(y, axis=1, output_type=tf.int32)
    else:
        y_int = tf.cast(y, tf.int32)

    N = int(tf.shape(y_int)[0])

    # ---- Get predicted classes for all samples in batches ----
    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    preds = []
    for xb in ds:
        logits = model(xb, training=False)
        preds.append(tf.argmax(logits, axis=1, output_type=tf.int32))
    y_pred_int = tf.concat(preds, axis=0)  # (N,)

    # ---- Compute one-vs-rest accuracy for each group ----
    accs = []
    targets = [0,1,2,3]
    t_k=0

    for g in dict_list:
        #print(len(g[1]))
        idx0 = np.asarray(g[0], dtype=np.int64)
        idx1 = np.asarray(g[1], dtype=np.int64)

        # Safety: filter out-of-bound indices
        idx0 = idx0[(idx0 >= 0) & (idx0 < N)]
        idx1 = idx1[(idx1 >= 0) & (idx1 < N)]


        if idx0.size == 0 and idx1.size == 0:
            accs.append(np.nan)
            targets.append(-1)
            continue

        # ---- Infer target_class: get unique value/mode from idx1 true labels ----
        if idx1.size == 0:
            # If class 1 is empty, cannot infer target; assigning -1 and using rule "only idx0: predict != -1 is always correct" is meaningless
            # More reasonable: return nan
            accs.append(np.nan)
            targets.append(-1)
            continue

        #y1 = tf.gather(y_int, idx1).numpy()
        # Mode (more robust: works even with small amount of noise)
        #vals, counts = np.unique(y1, return_counts=True)
        target = targets[t_k]
        t_k += 1
        #targets.append(target)


        correct = 0
        total = 0

        # Class 1: must predict target to be correct
        if idx1.size > 0:
            p1 = tf.gather(y_pred_int, idx1).numpy()
            correct += int(np.sum(p1 == target))
            total += int(idx1.size)

        # Class 0: must not predict target to be correct
        if idx0.size > 0:
            p0 = tf.gather(y_pred_int, idx0).numpy()
            correct += int(np.sum(p0 != target))
            total += int(idx0.size)

        accs.append(correct / total if total > 0 else np.nan)

    accs = np.asarray(accs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)

    if return_target_classes:
        return accs, targets
    return accs