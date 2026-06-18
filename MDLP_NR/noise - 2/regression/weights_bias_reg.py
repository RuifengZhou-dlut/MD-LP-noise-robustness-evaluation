import numpy as np
import tensorflow as tf
import scipy.optimize as opt
import os
from save_load import load_layer_outputs_and_labels

def W_streaming_with_inline_JW_stream(layer_name, all_Y, save_dir, block=100, L=9):

    weight_list = []
    J = 0

    sortY = tf.sort(all_Y)

    for i in range(L):
        NN = int(len(all_Y) / (L+1))
        nY = sortY[(i+1)*NN]
        print(f"\n==== split {i}, threshold={(i+1)*NN} ====")

        # ============================================
        # Pass 1 - compute mean1 / mean2
        # ============================================
        sum1 = 0.0
        sum2 = 0.0
        n1 = 0
        n2 = 0

        for X_b, Y_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

            if len(X_b.shape) > 2:
                X_b = X_b.reshape(X_b.shape[0], -1)

            mask1 = (Y_b < nY)
            mask2 = ~mask1

            if tf.reduce_any(mask1):
                sum1 += tf.reduce_sum(tf.boolean_mask(X_b, mask1), axis=0)
                n1 += int(tf.reduce_sum(tf.cast(mask1, tf.int32)))

            if tf.reduce_any(mask2):
                sum2 += tf.reduce_sum(tf.boolean_mask(X_b, mask2), axis=0)
                n2 += int(tf.reduce_sum(tf.cast(mask2, tf.int32)))

        mean1 = sum1 / (n1 + 1e-8)
        mean2 = sum2 / (n2 + 1e-8)

        m_i = n2 * mean1 - n1 * mean2
        m_i = m_i / (tf.linalg.norm(m_i) + 1e-8)

        LL = m_i.shape[0]

        # ============================================
        # Pass 2 - compute row_norm_sq
        # ============================================
        row_norm_sq = tf.zeros([LL], dtype=tf.float32)

        # Outer loop: X1 batches (streaming read)
        for X1_b, Y1_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

            if len(X1_b.shape) > 2:
                X1_b = X1_b.reshape(X1_b.shape[0], -1)

            mask1 = (Y1_b < nY)
            X1_b = tf.boolean_mask(X1_b, mask1)
            if X1_b.shape[0] == 0:
                continue

            # chunk X1
            n1b = X1_b.shape[0]
            for i1 in range(0, n1b, block):
                X1_block = X1_b[i1:i1+block]

                # Inner loop: X2 batches (streaming read again)
                for X2_b, Y2_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

                    if len(X2_b.shape) > 2:
                        X2_b = X2_b.reshape(X2_b.shape[0], -1)

                    mask2 = (Y2_b >= nY)
                    X2_b = tf.boolean_mask(X2_b, mask2)
                    if X2_b.shape[0] == 0:
                        continue

                    # chunk X2
                    n2b = X2_b.shape[0]
                    for i2 in range(0, n2b, block):
                        X2_block = X2_b[i2:i2+block]

                        diff = X1_block[:, None, :] - X2_block[None, :, :]
                        diff2 = tf.reshape(diff, [-1, LL])
                        diff2 = tf.transpose(diff2)

                        row_norm_sq += tf.reduce_sum(diff2 * diff2, axis=1)

        # ============================================
        # Pass 3 - use row_norm_sq to compute m_weighted and L1_total
        # ============================================
        reciprocal = tf.where(row_norm_sq > 0,
                              1.0/row_norm_sq,
                              tf.zeros_like(row_norm_sq))

        m_weighted = tf.reshape(m_i * reciprocal, [1, LL])

        L1_total = 0.0
        L1_abs_total = 0.0

        # Outer loop: X1 batches
        for X1_b, Y1_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

            if len(X1_b.shape) > 2:
                X1_b = X1_b.reshape(X1_b.shape[0], -1)

            mask1 = (Y1_b < nY)
            X1_b = tf.boolean_mask(X1_b, mask1)
            if X1_b.shape[0] == 0:
                continue

            n1b = X1_b.shape[0]
            for i1 in range(0, n1b, block):
                X1_block = X1_b[i1:i1+block]

                # Inner loop: X2 batches
                for X2_b, Y2_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

                    if len(X2_b.shape) > 2:
                        X2_b = X2_b.reshape(X2_b.shape[0], -1)

                    mask2 = (Y2_b >= nY)
                    X2_b = tf.boolean_mask(X2_b, mask2)
                    if X2_b.shape[0] == 0:
                        continue

                    n2b = X2_b.shape[0]
                    for i2 in range(0, n2b, block):
                        X2_block = X2_b[i2:i2+block]

                        diff = X1_block[:, None, :] - X2_block[None, :, :]
                        diff2 = tf.reshape(diff, [-1, LL])
                        diff2 = tf.transpose(diff2)

                        mM = tf.matmul(m_weighted, tf.cast(diff2, tf.float32))

                        L1_total += tf.reduce_sum(mM)
                        L1_abs_total += tf.reduce_sum(tf.abs(mM))

        J_i = tf.abs(L1_total) / (L1_abs_total + 1e-8)
        weight_list.append(tf.reshape(tf.sign(L1_total) * m_weighted, [LL]))

        J += J_i

    return J.numpy()/L, tf.stack(weight_list).numpy()


def F_b_streaming(layer_name, Y, w, save_dir):
    """
    Returns a function F(b) that can be called by scipy.optimize.
    layer_name: layer name for loading data
    Y: global Y (size N)
    w: current weight, shape [D]
    """

    Y = Y.astype(np.float32)

    def F_of_b(b):
        total = 0.0
        b = float(b)

        index = 0
        for X_b, Y_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir, chunk_size=500):
            # reshape batch
            if len(X_b.shape) == 4:
                B, H, W, C = X_b.shape
                X_b = X_b.reshape(B, H * W * C)
            else:
                X_b = X_b.reshape(X_b.shape[0], -1)

            logits = np.dot(X_b, w) + b
            Y_batch = Y[index:index + len(logits)]
            term = (1 - 2 * Y_batch).astype(np.float32) * logits
            total += np.sum(np.maximum(term, 0))
            index = index + len(logits)

        return total

    return F_of_b

def F_one_streaming(layer_name, Y, w, save_dir):
    # initial b
    b0 = 0.0

    # create F(b)
    Fb = F_b_streaming(layer_name, Y, w,  save_dir=save_dir)

    # optimization
    result = opt.minimize(Fb, b0)
    return float(result['x'])


def F_streaming(layer_name, Y, save_dir, L=9):
    """
    layer_name: layer name for loading data
    Y: global Y array
    save_dir: directory to load layer outputs from
    L: number of quantiles
    """
    # -------------------
    # first compute w_list (using your streaming W)
    # -------------------
    J, w_list = W_streaming_with_inline_JW_stream(layer_name, Y, save_dir=save_dir, L=L)

    # -------------------
    # prepare Y
    # -------------------
    # collect Y (small memory footprint, can be saved)
    total_size = len(Y)
    sortY = np.sort(Y)

    # -------------------
    # output b_list and xi_list
    # -------------------
    b_list = []
    # xi = np.zeros(total_size, dtype=np.float32)
    xi_list = []

    # -------------------
    # main loop over L quantiles
    # -------------------
    for i in range(L):

        xi = np.zeros(total_size, dtype=np.float32)

        NN = total_size // (L + 1)
        lab = (i + 1) * NN
        nY = sortY[lab]

        mask_1 = np.where(Y < nY)[0]
        mask_2 = np.where(Y >= nY)[0]

        # construct Y_ (0/1 binary label)
        Y_ = np.zeros(total_size)
        Y_[mask_1] = 1

        # current weight
        w_l = w_list[i]

        # compute b
        b_l = F_one_streaming(layer_name, Y_, w_l, save_dir=save_dir,)
        b_list.append(b_l)

        # --------------------------
        # compute xi = (2Y_-1)(X·w + b)
        # --------------------------
        index = 0
        for X_b, Y_b in load_layer_outputs_and_labels(layer_name, save_dir=save_dir):

            # reshape
            if len(X_b.shape) == 4:
                B, H, W, C = X_b.shape
                X_flat = X_b.reshape(B, H * W * C)
            else:
                X_flat = X_b.reshape(X_b.shape[0], -1)

            logits = np.dot(X_flat, w_l) + b_l
            Y_batch = Y_[index:index + len(logits)]
            xi_batch = (2 * Y_batch - 1) * logits

            xi[index:index + len(logits)] = xi_batch
            index += len(logits)

        print("xi>=0 count:", np.sum(xi >= 0))
        xi_list.append(xi)

    return w_list, b_list, xi



def save_wb_per_layer(w, b, lname, CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{lname}_w.npy"), w, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, f"{lname}_b.npy"), b, allow_pickle=True)
    #print("Saved all w,b into separate files.")

def load_wb_if_exists(Y, layer_list, CACHE_DIR, save_dir):
    """
    Try to load w and b for each layer.
    If any file is missing, return None and recompute.
    If all files exist, return (eva_w, eva_b)
    """

    eva_w = []
    eva_b = []

    for lname in layer_list:
        w_path = os.path.join(CACHE_DIR, f"{lname}_w.npy")
        b_path = os.path.join(CACHE_DIR, f"{lname}_b.npy")

        # if any file doesn't exist -> recompute
        if not os.path.exists(w_path) or not os.path.exists(b_path):
            w,b,xi=F_streaming(lname, Y, save_dir=save_dir)
            save_wb_per_layer(w, b, lname, CACHE_DIR)
        else:
            w = np.load(w_path, allow_pickle=True)
            b = np.load(b_path, allow_pickle=True)

        eva_w.append(w)
        eva_b.append(b)

    return eva_w, eva_b