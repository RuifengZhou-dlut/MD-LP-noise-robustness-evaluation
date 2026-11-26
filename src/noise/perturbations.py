"""Four input perturbations: Gaussian (gauss) / salt-and-pepper (salt) / motion blur (move) / occlusion (occ).

Randomness comes from the global TF/random seeds, so these functions must be
called in a fixed order for runs to be reproducible.
"""
import math
import os
import random

import numpy as np
import tensorflow as tf

from ..common.config import perturbation_file

PERTURBATIONS = ("gauss", "salt", "move", "occ")


# ------------------------------------------------------------------ generators
def gauss_noise(X, a):
    """Additive Gaussian noise with intensity `a`."""
    noise = tf.random.normal(shape=tf.shape(X), dtype=X.dtype)
    X_a = X + a * noise
    return tf.clip_by_value(X_a, 0.0, 1.0)


def salt_noise(X, n):
    """Salt-and-pepper noise: per image, a fraction n/total_pixels of the pixels is set to 0/1 at random."""
    a, b, c, d = tf.unstack(tf.shape(X))
    total_pixels = b * c

    ratio = tf.cast(n, tf.float32) / tf.cast(total_pixels, tf.float32)
    rand_mask = tf.random.uniform((a, b, c), dtype=tf.float32)
    noise_mask = rand_mask < ratio

    noise_values = tf.cast(tf.random.uniform((a, b, c, 1), maxval=2, dtype=tf.int32), X.dtype)
    noise_mask = tf.expand_dims(noise_mask, axis=-1)

    return tf.where(noise_mask, noise_values, X)


def motion_blur(X, degree, padding_mode="REFLECT"):
    """Motion blur with a random angle.

    Pure numpy implementation (per-image random angle + 15-point line kernel
    convolution). It accumulates shifted, kernel-weighted copies instead of a
    per-image TF depthwise_conv2d, which could randomly crash large eager-mode
    runs with 0xC0000005; the result is equivalent, REFLECT padding included.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.max() <= 1.0:
        X = X * 255.0

    N, H, W, C = X.shape
    pad = degree // 2
    pad_mode = {"REFLECT": "reflect", "SYMMETRIC": "symmetric", "REPLICATE": "edge"}[padding_mode]
    out = np.empty((N, H, W, C), dtype=np.float32)

    for i in range(N):
        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[degree // 2, :] = 1.0

        angle = random.uniform(0, 180)
        rad = math.radians(angle)
        center = (degree - 1) / 2.0
        y, x = np.ogrid[:degree, :degree]
        xr = (x - center) * np.cos(rad) + (y - center) * np.sin(rad) + center
        yr = -(x - center) * np.sin(rad) + (y - center) * np.cos(rad) + center
        rotated = np.zeros_like(kernel)
        valid = (xr >= 0) & (xr < degree) & (yr >= 0) & (yr < degree)
        rotated[valid] = kernel[np.clip(yr[valid].astype(int), 0, degree - 1),
                                np.clip(xr[valid].astype(int), 0, degree - 1)]

        s = np.sum(rotated)
        if s > 0:
            rotated /= s

        padded = np.pad(X[i], ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
        acc = np.zeros((H, W, C), dtype=np.float32)
        for ky, kx in zip(*np.nonzero(rotated)):
            acc += rotated[ky, kx] * padded[ky:ky + H, kx:kx + W]
        out[i] = acc

    return np.clip(out / 255.0, 0.0, 1.0)


def occlusion_noise(X, v, random_color=False):
    """Random rectangular occlusion with a fixed area.

    Fully vectorized numpy implementation.
    """
    X = np.asarray(X, dtype=np.float32)
    N, H, W, C = X.shape

    aspect_ratios = np.random.uniform(0.5, 2.0, size=N).astype(np.float32)
    h = np.round(np.sqrt(v / aspect_ratios)).astype(np.int64)
    w = np.round(np.sqrt(v * aspect_ratios)).astype(np.int64)
    h = np.clip(h, 1, H)
    w = np.clip(w, 1, W)

    top = (np.random.uniform(0, 1, size=N) * (H - h).astype(np.float32)).astype(np.int64)
    left = (np.random.uniform(0, 1, size=N) * (W - w).astype(np.float32)).astype(np.int64)

    yy = np.arange(H)[None, :, None]                                # (1,H,1)
    xx = np.arange(W)[None, None, :]                                # (1,1,W)
    rect_y = (yy >= top[:, None, None]) & (yy < np.minimum(top + h, H)[:, None, None])
    rect_x = (xx >= left[:, None, None]) & (xx < np.minimum(left + w, W)[:, None, None])
    mask = (rect_y & rect_x).astype(np.float32)[..., None]          # (N,H,W,1)

    if random_color:
        color = np.random.uniform(0.5, 1.0, size=X.shape).astype(np.float32)
    else:
        color = np.ones_like(X)
    X_v = X * (1.0 - mask) + color * mask
    return np.clip(X_v, 0.0, 1.0)


# ------------------------------------------------------------------ save / load
def load_perturbations(task, repeat):
    """Read the four corruption arrays of one repeat (gauss/salt/move/occ)."""
    return tuple(np.load(perturbation_file(task, name, repeat), allow_pickle=True)
                 for name in PERTURBATIONS)


def save_perturbations(task, repeat, arrays):
    """Write arrays: dict(name -> ndarray), name in PERTURBATIONS."""
    for name in PERTURBATIONS:
        path = perturbation_file(task, name, repeat)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arrays[name], allow_pickle=True)
