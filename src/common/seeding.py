"""Global random seeds: training, perturbation generation and PGD attacks draw
deterministic seeds from here.

Usage:
    set_global_seed(seed)                 # call once at the process entry point
    s = derive_seed("perturb", task, i)   # derive a stable sub-seed from the experiment dims

Note: the evaluation path (reading persisted npy / layer_cache files) has no
randomness of its own; seeds only affect "generating data" and "training".
"""
import hashlib
import os
import random

import numpy as np
import tensorflow as tf

# Default master seed; changing the number yields a different fully reproducible run
BASE_SEED = 20240101

# Noise generation and training are seeded separately: the 5 perturbations of one
# group (PGD/gauss/salt/move/occ) share a single group seed
NOISE_SEED = 20250101


def set_global_seed(seed=BASE_SEED, deterministic=False):
    """Seed python / numpy / tensorflow.

    With deterministic=True, op determinism is enabled as well (training is slower,
    but it is bit-for-bit reproducible in the same environment).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        if hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
    return seed


def derive_seed(*tags, base=BASE_SEED):
    """Derive a 32-bit seed from tags: constant for equal tags, different otherwise (process independent)."""
    key = "|".join(str(t) for t in tags).encode("utf-8")
    h = hashlib.md5(key).digest()
    return (int.from_bytes(h[:4], "little") + int(base)) % (2 ** 31 - 1)


def noise_group_seed(task, repeat, base=NOISE_SEED):
    """Seed of perturbation group `repeat`: the 5 noises in it (PGD/gauss/salt/move/occ) are derived from it."""
    return derive_seed("noise-group", task, repeat, base=base)
