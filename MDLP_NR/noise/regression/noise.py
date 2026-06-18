import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
import math

def noise_a(X, a):
    noise = tf.random.normal(shape=tf.shape(X), dtype=X.dtype)
    X_a = X + a * noise
    X_a = tf.clip_by_value(X_a, 0.0, 1.0)
    return X_a

def noise_p(X, n):
    a, b, c, d = tf.unstack(tf.shape(X))
    total_pixels = b * c

    # compute the ratio of pixels to be noised per image
    ratio = tf.cast(n, tf.float32) / tf.cast(total_pixels, tf.float32)

    # generate random noise mask [0,1) for each pixel
    rand_mask = tf.random.uniform((a, b, c), dtype=tf.float32)

    # True means this pixel will be noised
    noise_mask = rand_mask < ratio  # shape = (a,b,c)

    # generate random 0/1 for selected pixels
    noise_values = tf.cast(tf.random.uniform((a, b, c, 1), maxval=2, dtype=tf.int32), X.dtype)

    # expand mask to channel dimension
    noise_mask = tf.expand_dims(noise_mask, axis=-1)

    # use tf.where for direct replacement (GPU parallel)
    X_p = tf.where(noise_mask, noise_values, X)
    return X_p

def noise_m(X, degree, padding_mode="REFLECT"):
    """
    GPU-accelerated random motion blur (no black edge version)
    Args:
        X: [N, H, W, C] Tensor, 0~1 or 0~255
        degree: blur kernel size
        padding_mode: edge padding mode, optional {"REFLECT", "SYMMETRIC", "REPLICATE"}
    """
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
    if tf.reduce_max(X) <= 1.0:
        X = X * 255.0

    N, H, W, C = X.shape
    pad = degree // 2
    X_m = []

    for _ in range(N):
        # Step 1: generate line kernel
        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[degree // 2, :] = 1.0

        # Step 2: random rotation
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

        # Step 3: normalize
        s = np.sum(rotated)
        if s > 0:
            rotated /= s

        # Step 4: depthwise kernel
        kernel_tf = tf.constant(rotated, dtype=tf.float32)
        kernel_tf = tf.reshape(kernel_tf, [degree, degree, 1, 1])
        kernel_tf = tf.tile(kernel_tf, [1, 1, C, 1])

        # Step 5: edge padding
        if padding_mode == "REFLECT":
            xi = tf.pad(X[_:_+1], [[0,0], [pad,pad], [pad,pad], [0,0]], mode="REFLECT")
        elif padding_mode == "SYMMETRIC":
            xi = tf.pad(X[_:_+1], [[0,0], [pad,pad], [pad,pad], [0,0]], mode="SYMMETRIC")
        elif padding_mode == "REPLICATE":
            xi = tf.pad(X[_:_+1], [[0,0], [pad,pad], [pad,pad], [0,0]], mode="CONSTANT")
            xi = tf.concat([xi[:, pad:pad+1, :, :]]*pad + [xi] + [xi[:, -pad-1:-pad, :, :]]*pad, axis=1)
        else:
            raise ValueError("Unsupported padding mode")

        # Step 6: convolution blur (VALID)
        blurred = tf.nn.depthwise_conv2d(
            xi,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID"
        )
        X_m.append(blurred)

    X_m = tf.concat(X_m, axis=0)
    X_m = tf.clip_by_value(X_m / 255.0, 0.0, 1.0)
    return X_m


def noise_v(X, v, random_color=False):
    """
    GPU-accelerated version: add a fixed-area random rectangular noise block on input batch images.

    Args:
        X: input image tensor, shape (N, H, W, C), value range [0,1]
        v: target rectangular area (pixel count)
        random_color: if True, use random color block, otherwise use white block (1.0)

    Returns:
        X_v: image with rectangular noise added, shape same as X
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    N, H, W, C = X.shape

    # --- Step 1: random aspect ratio but keep area fixed ---
    aspect_ratios = tf.random.uniform([N], 0.5, 2.0)  # random aspect ratio w/h
    h = tf.sqrt(v / aspect_ratios)  # height
    w = tf.sqrt(v * aspect_ratios)  # width
    h = tf.cast(tf.round(h), tf.int32)
    w = tf.cast(tf.round(w), tf.int32)

    # clip to image bounds
    h = tf.clip_by_value(h, 1, H)
    w = tf.clip_by_value(w, 1, W)

    # --- Step 2: randomly generate rectangle top-left corner ---
    top = tf.random.uniform([N], 0, tf.cast(H - tf.cast(h, tf.float32), tf.float32))
    left = tf.random.uniform([N], 0, tf.cast(W - tf.cast(w, tf.float32), tf.float32))
    top = tf.cast(top, tf.int32)
    left = tf.cast(left, tf.int32)

    # --- Step 3: construct mask ---
    mask_array = tf.TensorArray(dtype=tf.float32, size=N)

    def draw_rect(i, mask_array):
        mask_i = tf.zeros([H, W], dtype=tf.float32)
        yi = tf.range(top[i], top[i] + h[i])
        xi = tf.range(left[i], left[i] + w[i])
        yi = tf.clip_by_value(yi, 0, H - 1)
        xi = tf.clip_by_value(xi, 0, W - 1)

        yy, xx = tf.meshgrid(yi, xi, indexing="ij")
        coords = tf.stack([yy, xx], axis=-1)
        coords = tf.reshape(coords, [-1, 2])
        updates = tf.ones([tf.shape(coords)[0]], dtype=tf.float32)

        mask_i = tf.tensor_scatter_nd_update(mask_i, coords, updates)
        mask_array = mask_array.write(i, tf.expand_dims(mask_i, -1))
        return mask_array

    for i in tf.range(N):
        mask_array = draw_rect(i, mask_array)

    mask = mask_array.stack()  # [N, H, W, 1]
    mask = tf.broadcast_to(mask, tf.shape(X))

    # --- Step 4: apply noise block ---
    if random_color:
        color = tf.random.uniform(tf.shape(X), 0.5, 1.0)
    else:
        color = tf.ones_like(X)

    X_v = X * (1.0 - mask) + color * mask
    X_v = tf.clip_by_value(X_v, 0.0, 1.0)

    return X_v

class BaseClassificationAdversaryL2:
    """
    Multi-class classification model adversarial attack base class (L2 constraint, untargeted: maximize Cross-Entropy).
    Applicable input: images [B,H,W,C]
    Compatible y:
      - sparse label: [B] (int32/int64)
      - one-hot:      [B,num_class] (float32/float64)
    """

    def __init__(self, model, num_classes=4, clip_min=0.0, clip_max=1.0, seed=None,
                 from_logits=None, label_smoothing=0.0):
        self.model = model
        self.num_classes = int(num_classes)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rng = tf.random.Generator.from_seed(
            int(seed) if seed is not None else random.randint(1, 100000)
        )
        self.label_smoothing = float(label_smoothing)

        # automatically infer logits/prob: if user doesn't specify from_logits, try to detect last layer activation
        if from_logits is None:
            act = None
            try:
                last = model.layers[-1]
                act = getattr(last, "activation", None)
            except Exception:
                act = None
            # softmax => probs; other/unknown => treat as logits (more stable)
            self.from_logits = False if (act is not None and act.__name__ == "softmax") else True
        else:
            self.from_logits = bool(from_logits)

        # cross entropy object (reduction=None to get per-sample loss)
        self._ce_obj = tf.keras.losses.CategoricalCrossentropy(
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    # ---------- label handling ----------
    def _to_onehot(self, y):
        """
        Convert y to one-hot float32: [B,num_classes]
        """
        y = tf.convert_to_tensor(y)
        if y.shape.rank == 1:
            y = tf.cast(y, tf.int32)
            y_oh = tf.one_hot(y, depth=self.num_classes, dtype=tf.float32)
            return y_oh
        else:
            # assume already one-hot / soft label
            y = tf.cast(y, tf.float32)
            # if dimensions are incorrect, this will be exposed during training
            return y

    # ---------- loss ----------
    def _ce_per_sample(self, y_pred, y_true):
        """
        y_pred: [B,num_classes] (logits or probs)
        y_true: [B] or [B,num_classes]
        Returns: [B]
        """
        y_true_oh = self._to_onehot(y_true)
        return self._ce_obj(y_true_oh, y_pred)  # [B]

    def loss_mean(self, x, y):
        """
        Batch average CE (scalar), used for PGD gradient ascent
        """
        pred = self.model(x, training=False)
        per = self._ce_per_sample(pred, y)
        return tf.reduce_mean(per)

    # ---------- l2 normalize ----------
    @staticmethod
    def l2_normalize_per_sample(t):
        """Per-sample L2 normalization: t [B,...] -> [B,...]"""
        flat = tf.reshape(t, [tf.shape(t)[0], -1])
        n = tf.norm(flat, ord=2, axis=1, keepdims=True)
        return tf.reshape(flat / (n + 1e-12), tf.shape(t))

    # ---------- projection ----------
    def clip_and_project_l2(self, x, x0, eps):
        """Project to L2 ball centered at x0 with radius eps (per-sample), and clip to input range"""
        x = tf.cast(x, tf.float32)
        x0 = tf.cast(x0, tf.float32)

        delta = x - x0
        delta_flat = tf.reshape(delta, [tf.shape(delta)[0], -1])
        l2 = tf.norm(delta_flat, ord=2, axis=1, keepdims=True)

        scale = tf.minimum(1.0, eps / (l2 + 1e-12))
        delta_proj = delta_flat * scale
        x_proj = x0 + tf.reshape(delta_proj, tf.shape(delta))

        if self.clip_min is not None or self.clip_max is not None:
            cmin = -1e9 if self.clip_min is None else self.clip_min
            cmax =  1e9 if self.clip_max is None else self.clip_max
            x_proj = tf.clip_by_value(x_proj, cmin, cmax)

        return x_proj

    # ---------- random start ----------
    def random_l2_perturb(self, x_shape, eps):
        """Generate random perturbation within L2 ball (per-sample)"""
        noise = self.rng.normal(x_shape, dtype=tf.float32)
        noise_flat = tf.reshape(noise, [tf.shape(noise)[0], -1])
        norm = tf.norm(noise_flat, ord=2, axis=1, keepdims=True)
        direction = noise_flat / (norm + 1e-12)

        D = tf.cast(tf.shape(noise_flat)[1], tf.float32)
        u = self.rng.uniform([tf.shape(noise_flat)[0], 1], 0.0, 1.0, dtype=tf.float32)
        r = tf.pow(u, 1.0 / D) * eps
        return tf.reshape(direction * r, x_shape)

    # ---------- dataset-level generation ----------
    def generate_adversarial_dataset(
        self, x_all, y_all, batch_size=128, shuffle=False, seed=1234,
        drop_remainder=False, prefetch=True, **attack_kwargs
    ):
        """
        Automatically split into batches, call self.attack_batch to generate adversarial samples and concatenate.
        Returns: x_adv_all
        """
        x_all = tf.convert_to_tensor(x_all, dtype=tf.float32)
        y_all = tf.convert_to_tensor(y_all)
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
        y_chunks = []
        for xb, yb, ib in ds:
            xb_adv = self.attack_batch(xb, yb, **attack_kwargs)
            adv_chunks.append(xb_adv)
            y_chunks.append(yb)
            idx_chunks.append(ib)

        x_adv_all = tf.concat(adv_chunks, axis=0)
        idx_all = tf.concat(idx_chunks, axis=0)
        y_out = tf.concat(y_chunks, axis=0)

        if shuffle:
            order = tf.argsort(idx_all, stable=True)
            x_adv_all = tf.gather(x_adv_all, order, axis=0)
            y_out = tf.gather(y_out, order, axis=0)

        return x_adv_all  # if you need to return y_out as well, change to return (x_adv_all, y_out)

    def attack_batch(self, xb, yb, **kwargs):
        raise NotImplementedError

class PGDClassificationAdversaryL2(BaseClassificationAdversaryL2):
    """
    PGD white-box attack (L2 constraint, untargeted: maximize Cross-Entropy).
    - Compatible with PGDRegressionAdversaryL2 interface for seamless replacement.
    """

    @tf.function
    def attack_batch(
        self,
        xb,
        yb,
        eps=1.0,
        alpha=0.2,
        steps=20,
        random_start=True,
    ):
        """
        Generate adversarial samples for one batch (white-box: 1 forward + 1 backward per step).
        xb: [B,H,W,C]
        yb: [B] / [B,1] / [B,num_classes]
        """
        x0 = tf.cast(xb, tf.float32)

        # random start (within L2 ball)
        if random_start:
            noise = self.random_l2_perturb(tf.shape(x0), eps)
            x_adv = self.clip_and_project_l2(x0 + noise, x0, eps)
        else:
            x_adv = tf.identity(x0)

        # PGD gradient ascent to maximize CE
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                loss = self.loss_mean(x_adv, yb)  # scalar CE mean

            grad = tape.gradient(loss, x_adv)
            grad = tf.zeros_like(x_adv) if grad is None else tf.cast(grad, tf.float32)

            grad_unit = self.l2_normalize_per_sample(grad)
            x_adv = x_adv + alpha * grad_unit
            x_adv = self.clip_and_project_l2(x_adv, x0, eps)

        return x_adv

class BaseRegressionAdversaryL2:
    """
    Single-output regression model adversarial attack base class (L2 constraint, untargeted: maximize MSE).
    Applicable input: images [B, 32, 32, C] or general [B, H, W, C]
    """

    def __init__(self, model, clip_min=0.0, clip_max=1.0, seed=random.randint(1,100)):
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rng = tf.random.Generator.from_seed(seed)

    # ---------- loss ----------
    @staticmethod
    def _mse_per_sample(y_pred, y_true):
        """Per-sample scalar MSE: returns [B]"""
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        return tf.square(y_pred - y_true)

    def loss_mean(self, x, y):
        """Batch average MSE (scalar)"""
        pred = self.model(x, training=False)
        return tf.reduce_mean(self._mse_per_sample(pred, y))

    # ---------- l2 normalize ----------
    @staticmethod
    def l2_normalize_per_sample(t):
        """Per-sample L2 normalization: t [B,...] -> [B,...]"""
        flat = tf.reshape(t, [tf.shape(t)[0], -1])
        n = tf.norm(flat, ord=2, axis=1, keepdims=True)
        return tf.reshape(flat / (n + 1e-12), tf.shape(t))

    # ---------- projection ----------
    def clip_and_project_l2(self, x, x0, eps):
        """
        Project to L2 ball centered at x0 with radius eps (per-sample), and clip to input range.
        """
        x = tf.cast(x, tf.float32)
        x0 = tf.cast(x0, tf.float32)

        delta = x - x0
        delta_flat = tf.reshape(delta, [tf.shape(delta)[0], -1])
        l2 = tf.norm(delta_flat, ord=2, axis=1, keepdims=True)

        scale = tf.minimum(1.0, eps / (l2 + 1e-12))
        delta_proj = delta_flat * scale
        x_proj = x0 + tf.reshape(delta_proj, tf.shape(delta))

        if self.clip_min is not None or self.clip_max is not None:
            cmin = -1e9 if self.clip_min is None else self.clip_min
            cmax = 1e9 if self.clip_max is None else self.clip_max
            x_proj = tf.clip_by_value(x_proj, cmin, cmax)

        return x_proj


    # ---------- random start ----------
    def random_l2_perturb(self, x_shape, eps):
        """
        Generate random perturbation within L2 ball (per-sample).
        """
        noise = self.rng.normal(x_shape, dtype=tf.float32)
        noise_flat = tf.reshape(noise, [tf.shape(noise)[0], -1])
        norm = tf.norm(noise_flat, ord=2, axis=1, keepdims=True)
        direction = noise_flat / (norm + 1e-12)

        D = tf.cast(tf.shape(noise_flat)[1], tf.float32)
        u = self.rng.uniform([tf.shape(noise_flat)[0], 1], 0.0, 1.0, dtype=tf.float32)
        r = tf.pow(u, 1.0 / D) * eps

        return tf.reshape(direction * r, x_shape)

    # ---------- dataset-level generation ----------
    def generate_adversarial_dataset(
        self,
        x_all,
        y_all,
        batch_size=128,
        shuffle=False,
        seed=1234,
        drop_remainder=False,
        prefetch=True,
        **attack_kwargs,
    ):
        """
        Input entire dataset (x_all, y_all), automatically split into batches, call self.attack_batch to generate adversarial samples and concatenate.
        Subclass must implement: attack_batch(xb, yb, **attack_kwargs) -> xb_adv

        Returns:
          x_adv_all, y_out (y_out is the same as y_all).
        """
        x_all = tf.convert_to_tensor(x_all, dtype=tf.float32)
        y_all = tf.convert_to_tensor(y_all, dtype=tf.float32)
        N = tf.shape(x_all)[0]

        idx = tf.range(N, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((x_all, y_all, idx))
        if shuffle:
            buf = int(x_all.shape[0]) if x_all.shape[0] is not None else 10000
            ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=False)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)

        adv_chunks, y_chunks, idx_chunks = [], [], []
        for xb, yb, ib in ds:
            xb_adv = self.attack_batch(xb, yb, **attack_kwargs)
            adv_chunks.append(xb_adv)
            y_chunks.append(yb)
            idx_chunks.append(ib)

        x_adv_all = tf.concat(adv_chunks, axis=0)
        y_out = tf.concat(y_chunks, axis=0)
        idx_all = tf.concat(idx_chunks, axis=0)

        if shuffle:
            order = tf.argsort(idx_all, stable=True)
            x_adv_all = tf.gather(x_adv_all, order, axis=0)
            y_out = tf.gather(y_out, order, axis=0)

        return x_adv_all

    # Subclass must override
    def attack_batch(self, xb, yb, **kwargs):
        raise NotImplementedError

class PGDRegressionAdversaryL2(BaseRegressionAdversaryL2):
    """
    PGD white-box attack (L2 constraint, untargeted: maximize MSE).
    """

    @tf.function
    def attack_batch(
        self,
        xb,
        yb,
        eps=1.0,
        alpha=0.2,
        steps=20,
        random_start=True,
    ):
        """
        Generate adversarial samples for one batch.
        """
        x0 = tf.cast(xb, tf.float32)
        yb = tf.cast(yb, tf.float32)

        # random start within L2 ball
        if random_start:
            noise = self.random_l2_perturb(tf.shape(x0), eps)
            x_adv = self.clip_and_project_l2(x0 + noise, x0, eps)
        else:
            x_adv = tf.identity(x0)

        # PGD: gradient ascent to maximize MSE
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                loss = self.loss_mean(x_adv, yb)  # scalar

            grad = tape.gradient(loss, x_adv)
            if grad is None:
                grad = tf.zeros_like(x_adv)
            else:
                grad = tf.cast(grad, tf.float32)

            grad_unit = self.l2_normalize_per_sample(grad)
            x_adv = x_adv + alpha * grad_unit
            x_adv = self.clip_and_project_l2(x_adv, x0, eps)

        return x_adv
