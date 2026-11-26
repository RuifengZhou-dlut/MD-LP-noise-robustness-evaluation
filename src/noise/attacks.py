"""L2-constrained PGD adversarial example generation (regression and classification).

A single BaseAdversaryL2 covers both tasks: the `task` argument selects the loss
(cross-entropy for classification, MSE for regression).
"""
import random

import tensorflow as tf


class BaseAdversaryL2:
    def __init__(self, model, task="regression", num_classes=None,
                 clip_min=0.0, clip_max=1.0, seed=None,
                 from_logits=None, label_smoothing=0.0):
        self.model = model
        self.task = task
        self.loss_kind = "mse" if task == "regression" else "ce"
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rng = tf.random.Generator.from_seed(
            int(seed) if seed is not None else random.randint(1, 100000)
        )
        self.label_smoothing = float(label_smoothing)

        if self.loss_kind == "ce":
            if from_logits is None:
                act = None
                try:
                    act = getattr(model.layers[-1], "activation", None)
                except Exception:
                    act = None
                self.from_logits = False if (act is not None and act.__name__ == "softmax") else True
            else:
                self.from_logits = bool(from_logits)

            self._ce_obj = tf.keras.losses.CategoricalCrossentropy(
                from_logits=self.from_logits,
                label_smoothing=self.label_smoothing,
                reduction=tf.keras.losses.Reduction.NONE,
            )

    # ---------- classification: labels / loss ----------
    def _to_onehot(self, y):
        y = tf.convert_to_tensor(y)
        if y.shape.rank == 1:
            return tf.one_hot(tf.cast(y, tf.int32), depth=self.num_classes, dtype=tf.float32)
        return tf.cast(y, tf.float32)

    def _ce_per_sample(self, y_pred, y_true):
        return self._ce_obj(self._to_onehot(y_true), y_pred)

    def loss_mean(self, x, y):
        """Batch-mean loss for the PGD gradient ascent: MSE for regression, cross-entropy for classification."""
        pred = self.model(x, training=False)
        if self.loss_kind == "mse":
            y_pred = tf.reshape(pred, [-1])
            y_true = tf.reshape(y, [-1])
            return tf.reduce_mean(tf.square(y_pred - y_true))
        return tf.reduce_mean(self._ce_per_sample(pred, y))

    # ---------- shared utilities ----------
    @staticmethod
    def l2_normalize_per_sample(t):
        flat = tf.reshape(t, [tf.shape(t)[0], -1])
        n = tf.norm(flat, ord=2, axis=1, keepdims=True)
        return tf.reshape(flat / (n + 1e-12), tf.shape(t))

    def clip_and_project_l2(self, x, x0, eps):
        x = tf.cast(x, tf.float32)
        x0 = tf.cast(x0, tf.float32)

        delta = x - x0
        delta_flat = tf.reshape(delta, [tf.shape(delta)[0], -1])
        l2 = tf.norm(delta_flat, ord=2, axis=1, keepdims=True)

        scale = tf.minimum(1.0, eps / (l2 + 1e-12))
        x_proj = x0 + tf.reshape(delta_flat * scale, tf.shape(delta))

        if self.clip_min is not None or self.clip_max is not None:
            cmin = -1e9 if self.clip_min is None else self.clip_min
            cmax = 1e9 if self.clip_max is None else self.clip_max
            x_proj = tf.clip_by_value(x_proj, cmin, cmax)
        return x_proj

    def random_l2_perturb(self, x_shape, eps):
        noise = self.rng.normal(x_shape, dtype=tf.float32)
        noise_flat = tf.reshape(noise, [tf.shape(noise)[0], -1])
        norm = tf.norm(noise_flat, ord=2, axis=1, keepdims=True)
        direction = noise_flat / (norm + 1e-12)

        D = tf.cast(tf.shape(noise_flat)[1], tf.float32)
        u = self.rng.uniform([tf.shape(noise_flat)[0], 1], 0.0, 1.0, dtype=tf.float32)
        r = tf.pow(u, 1.0 / D) * eps
        return tf.reshape(direction * r, x_shape)

    # ---------- dataset-level generation ----------
    def generate_adversarial_dataset(self, x_all, y_all, batch_size=128, shuffle=False,
                                     seed=1234, drop_remainder=False, prefetch=True,
                                     **attack_kwargs):
        x_all = tf.convert_to_tensor(x_all, dtype=tf.float32)
        y_all = tf.cast(tf.convert_to_tensor(y_all), tf.float32) if self.loss_kind == "mse" \
            else tf.convert_to_tensor(y_all)
        N = tf.shape(x_all)[0]

        idx = tf.range(N, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((x_all, y_all, idx))
        if shuffle:
            buf = int(x_all.shape[0]) if x_all.shape[0] is not None else 10000
            ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=False)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)

        adv_chunks, idx_chunks = [], []
        for xb, yb, ib in ds:
            adv_chunks.append(self.attack_batch(xb, yb, **attack_kwargs))
            idx_chunks.append(ib)

        x_adv_all = tf.concat(adv_chunks, axis=0)
        idx_all = tf.concat(idx_chunks, axis=0)

        if shuffle:
            order = tf.argsort(idx_all, stable=True)
            x_adv_all = tf.gather(x_adv_all, order, axis=0)
        return x_adv_all

    def attack_batch(self, xb, yb, **kwargs):
        raise NotImplementedError


class PGDAdversaryL2(BaseAdversaryL2):
    """White-box PGD attack (L2-ball constrained, untargeted)."""

    @tf.function
    def attack_batch(self, xb, yb, eps=1.0, alpha=0.2, steps=20, random_start=True):
        x0 = tf.cast(xb, tf.float32)
        if self.loss_kind == "mse":
            yb = tf.cast(yb, tf.float32)

        if random_start:
            noise = self.random_l2_perturb(tf.shape(x0), eps)
            x_adv = self.clip_and_project_l2(x0 + noise, x0, eps)
        else:
            x_adv = tf.identity(x0)

        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                loss = self.loss_mean(x_adv, yb)

            grad = tape.gradient(loss, x_adv)
            grad = tf.zeros_like(x_adv) if grad is None else tf.cast(grad, tf.float32)

            grad_unit = self.l2_normalize_per_sample(grad)
            x_adv = x_adv + alpha * grad_unit
            x_adv = self.clip_and_project_l2(x_adv, x0, eps)

        return x_adv


def make_pgd(task, model, num_classes=None, seed=None):
    return PGDAdversaryL2(model, task=task, num_classes=num_classes, seed=seed)
