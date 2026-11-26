"""Persisting and streaming reads of intermediate-layer outputs.

The task variants differ only in the Windows retry logic and in whether labels
are reshaped with reshape(-1); here both are applied uniformly (the reshape is
an identity transform for 1-D labels).
"""
import os
import time

import numpy as np
import tensorflow as tf


def _makedirs_retry(path, attempts=6, delay=3.0):
    for k in range(attempts):
        try:
            os.makedirs(path, exist_ok=True)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(delay)


def _np_save_retry(file_path, arr, attempts=6, delay=3.0):
    """Write to a temporary file, then replace atomically to avoid stray Windows Access Denied."""
    tmp = file_path + ".tmp"
    for k in range(attempts):
        try:
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, file_path)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(delay)


def save_layer_outputs_and_labels(model, X, Y, layer_list, save_dir, batch_size=5000):
    """Write the requested layers' outputs (flattened) and labels into save_dir; skip if they exist."""
    save_dir = str(save_dir)
    _makedirs_retry(save_dir)

    all_exist = True
    for lname in layer_list:
        if not (os.path.exists(os.path.join(save_dir, f"{lname}.npy"))
                and os.path.exists(os.path.join(save_dir, f"{lname}_labels.npy"))):
            all_exist = False
            break
    if all_exist:
        print(f"[SKIP] All layer files already exist in {save_dir}")
        return

    print(f"[SAVE] Generating layer outputs for: {save_dir}")

    layer_models = {
        lname: tf.keras.Model(inputs=model.input, outputs=model.get_layer(lname).output)
        for lname in layer_list
    }
    buffers = {lname: [] for lname in layer_list}
    label_buffers = {lname: [] for lname in layer_list}

    N = X.shape[0]
    steps = (N + batch_size - 1) // batch_size

    for step in range(steps):
        s = step * batch_size
        e = min((step + 1) * batch_size, N)
        X_batch, Y_batch = X[s:e], Y[s:e]

        for lname in layer_list:
            out = layer_models[lname].predict(X_batch, verbose=0)
            if len(out.shape) > 2:
                out = out.reshape(out.shape[0], -1)
            buffers[lname].append(out)
            label_buffers[lname].append(Y_batch)

    for lname in layer_list:
        arr = np.concatenate(buffers[lname], axis=0)
        lab = np.concatenate(label_buffers[lname], axis=0)
        _np_save_retry(os.path.join(save_dir, f"{lname}.npy"), arr)
        _np_save_retry(os.path.join(save_dir, f"{lname}_labels.npy"), lab)
        print(f"[Saved] {lname}: outputs {arr.shape}, labels {lab.shape}")


def load_layer_outputs_and_labels(layer_name, save_dir, chunk_size=5000):
    """Stream one layer's outputs with the matching labels, yielding (X_chunk, Y_chunk)."""
    save_dir = str(save_dir)
    arr = np.load(os.path.join(save_dir, f"{layer_name}.npy"))
    lab = np.load(os.path.join(save_dir, f"{layer_name}_labels.npy"))

    N = arr.shape[0]
    steps = (N + chunk_size - 1) // chunk_size

    for step in range(steps):
        s = step * chunk_size
        e = min((step + 1) * chunk_size, N)
        yield arr[s:e], lab[s:e].reshape(-1)
