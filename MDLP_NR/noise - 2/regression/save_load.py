import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os

def load_noise_if_exists(path):

    g_path = os.path.join(path, "gauss.npy")
    s_path = os.path.join(path, "salt.npy")
    m_path = os.path.join(path, "move.npy")
    o_path = os.path.join(path, "occ.npy")


    xg = np.load(g_path, allow_pickle=True)
    xs = np.load(s_path, allow_pickle=True)
    xm = np.load(m_path, allow_pickle=True)
    xo = np.load(o_path, allow_pickle=True)

    return xg, xs, xm, xo

def save_layer_outputs_and_labels(model, X, Y, layer_list, save_dir, batch_size=5000):
    """
    Save hidden layer outputs only if corresponding files don't exist:
        save_dir/layer_name.npy
        save_dir/layer_name_labels.npy

    Skip computation if files already exist.
    """

    os.makedirs(save_dir, exist_ok=True)

    # check if all layer files exist
    all_exist = True
    for lname in layer_list:
        arr_path = os.path.join(save_dir, f"{lname}.npy")
        lab_path = os.path.join(save_dir, f"{lname}_labels.npy")
        if not (os.path.exists(arr_path) and os.path.exists(lab_path)):
            all_exist = False
            break

    if all_exist:
        print(f"[SKIP] All layer files already exist in {save_dir}")
        return

    print(f"[SAVE] Generating layer outputs for: {save_dir}")

    # create forward model for each layer
    layer_models = {
        lname: tf.keras.Model(inputs=model.input, outputs=model.get_layer(lname).output)
        for lname in layer_list
    }

    # prepare buffers for each layer
    buffers = {lname: [] for lname in layer_list}
    label_buffers = {lname: [] for lname in layer_list}

    N = X.shape[0]
    steps = (N + batch_size - 1) // batch_size

    for step in range(steps):
        s = step * batch_size
        e = min((step + 1) * batch_size, N)

        X_batch = X[s:e]
        Y_batch = Y[s:e]

        for lname in layer_list:
            out = layer_models[lname].predict(X_batch, verbose=0)

            # flatten conv output
            if len(out.shape) > 2:
                out = out.reshape(out.shape[0], -1)

            buffers[lname].append(out)
            label_buffers[lname].append(Y_batch)

    # save to disk
    for lname in layer_list:
        arr = np.concatenate(buffers[lname], axis=0)
        lab = np.concatenate(label_buffers[lname], axis=0)

        np.save(os.path.join(save_dir, f"{lname}.npy"), arr)
        np.save(os.path.join(save_dir, f"{lname}_labels.npy"), lab)

        print(f"[Saved] {lname}: outputs {arr.shape}, labels {lab.shape}")

def load_layer_outputs_and_labels(layer_name, save_dir, chunk_size=5000):
    """
    Stream read hidden layer outputs and corresponding labels (sample ids not saved).
    Returns:
        X_L_chunk: (chunk, D)
        Y_chunk: (chunk,)
    """
    arr = np.load(os.path.join(save_dir, f"{layer_name}.npy"))
    lab = np.load(os.path.join(save_dir, f"{layer_name}_labels.npy"))

    N = arr.shape[0]
    steps = (N + chunk_size - 1) // chunk_size

    for step in range(steps):
        s = step * chunk_size
        e = min((step + 1) * chunk_size, N)

        yield arr[s:e], lab[s:e]