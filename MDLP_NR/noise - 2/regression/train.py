import tensorflow as tf
import tensorflow.keras as keras
import math
from collections import deque

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, warmup_lr=0.0):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
    def __call__(self, step):
        if step is None:
            step = tf.constant(0)
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_percent_done = step / warmup_steps
        learning_rate = tf.where(
            step < warmup_steps,
            self.warmup_lr + (self.base_lr - self.warmup_lr) * warmup_percent_done,
            self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        )
        return learning_rate
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_lr": self.warmup_lr,
        }

class CustomWeightDecaySGD(tf.keras.optimizers.SGD):
    def __init__(self, weight_decay, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay
    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)
        for grad, var in grads_and_vars:
            if ('kernel' in var.name) and ('bn' not in var.name.lower()):
                var.assign_sub(self.weight_decay * var)
    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_decay": float(self.weight_decay),  # ensure float
        })
        return config

class AdamW(tf.keras.optimizers.Adam):
    def __init__(self, weight_decay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        super().apply_gradients(grads_and_vars, name, **kwargs)
        for g, v in grads_and_vars:
            if g is None: continue
            if ('kernel' in v.name) and ('norm' not in v.name.lower()):
                v.assign_sub(self.weight_decay * v)
    def get_config(self):
        return {
            "weight_decay": self.weight_decay,
        }

class LastNSaver(tf.keras.callbacks.Callback):
    def __init__(self, n=10):
        super().__init__()
        self.n = n
        self.history = deque(maxlen=n)  # store last N (val_acc, weights)

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if val_acc is not None:
            # save (val_acc, current weights)
            weights = self.model.get_weights()
            self.history.append((val_acc, weights))

    def on_train_end(self, logs=None):
        # select best among last N
        if not self.history:
            return
        best_acc, best_weights = max(self.history, key=lambda x: x[0])
        print(f" Using best val_acc={best_acc:.4f} from last {self.n} epochs")
        self.model.set_weights(best_weights)  # restore best weights

AUTOTUNE = tf.data.AUTOTUNE
def cifar_pad_crop_flip(x, y, image_size=32, pad=4):
    # x: [H,W,C] in [0,1] float32
    x = tf.image.resize_with_crop_or_pad(x, image_size + 2*pad, image_size + 2*pad)  # 40x40
    x = tf.image.random_crop(x, size=[image_size, image_size, 3])
    x = tf.image.random_flip_left_right(x)
    return x, y

@tf.function
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    b = tf.shape(x)[0]

    g1 = tf.random.gamma([b], alpha)
    g2 = tf.random.gamma([b], alpha)
    lam = g1 / (g1 + g2)  # (B,)

    idx = tf.random.shuffle(tf.range(b))
    x2 = tf.gather(x, idx)
    y2 = tf.gather(y, idx)

    # ---- construct lam_x for x: shape = (B, 1, 1, 1, ...) ----
    rx = tf.rank(x)
    lam_x_shape = tf.concat([[b], tf.ones([rx - 1], dtype=tf.int32)], axis=0)
    lam_x = tf.reshape(lam, lam_x_shape)  # (B,1,1,1...) broadcast compatible with x

    x_mix = lam_x * x + (1.0 - lam_x) * x2

    # ---- construct lam_y for y: shape = (B, 1, 1, ...) ----
    ry = tf.rank(y)
    lam_y_shape = tf.concat([[b], tf.ones([ry - 1], dtype=tf.int32)], axis=0)
    lam_y = tf.reshape(lam, lam_y_shape)

    y_mix = lam_y * y + (1.0 - lam_y) * y2

    return x_mix, y_mix

def randaugment_mild(x, y, strong_aug=False):
    # mild color jitter (keep hue/saturation unchanged)
    if tf.random.uniform([]) < (0.6 if strong_aug else 0.3):
        x = tf.image.random_brightness(x, max_delta=0.1)
    if tf.random.uniform([]) < (0.6 if strong_aug else 0.3):
        x = tf.image.random_contrast(x, lower=0.8, upper=1.2)

    # cutout (mild)
    if strong_aug and tf.random.uniform([]) < 0.3:
        h = tf.shape(x)[0]
        w = tf.shape(x)[1]
        cutout_frac = tf.random.uniform([], 0.15, 0.35)
        ch = tf.cast(cutout_frac * tf.cast(h, tf.float32), tf.int32)
        cw = tf.cast(cutout_frac * tf.cast(w, tf.float32), tf.int32)
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)
        y1 = tf.clip_by_value(cy - ch // 2, 0, h)
        y2 = tf.clip_by_value(cy + ch // 2, 0, h)
        x1 = tf.clip_by_value(cx - cw // 2, 0, w)
        x2 = tf.clip_by_value(cx + cw // 2, 0, w)

        mask = tf.ones([h, w, 3], dtype=x.dtype)
        cut = tf.pad(tf.zeros([y2 - y1, x2 - x1, 3], dtype=x.dtype),
                     paddings=[[y1, h - y2], [x1, w - x2], [0, 0]])
        x = x * (mask - cut)

    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y

def make_train_ds(x_train, y_train_onehot, batch_size=128,
                  image_size=32, pad=4, mixup_alpha=0.0, strong_aug=False):
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
    ds = ds.shuffle(min(len(x_train), 20000), reshuffle_each_iteration=True)

    def to_float(x, y):
        x = tf.cast(x, tf.float32)
        # uncomment next line if input is uint8 0..255
        # x = x / 255.0
        return x, y

    ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: cifar_pad_crop_flip(x, y, image_size=image_size, pad=pad),
                num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: randaugment_mild(x, y, strong_aug=strong_aug),
                num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)

    if mixup_alpha and mixup_alpha > 0:
        ds = ds.map(lambda xb, yb: mixup_batch(xb, yb, alpha=mixup_alpha),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_test_ds(x_test, y_test_onehot, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_onehot))
    def to_float(x, y):
        x = tf.cast(x, tf.float32)
        # if input is 0..255: x /= 255.0
        return x, y
    ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

class RSquared(tf.keras.metrics.Metric):
    def __init__(self, name="r_squared", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sq_res = self.add_weight(name="ss_res", initializer="zeros")
        self.sum_sq_tot = self.add_weight(name="ss_tot", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.mean_y = self.add_weight(name="mean_y", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # update y_true mean
        batch_mean = tf.reduce_mean(y_true)
        self.mean_y.assign_add(batch_mean)
        self.count.assign_add(1.0)

        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - batch_mean))

        self.sum_sq_res.assign_add(ss_res)
        self.sum_sq_tot.assign_add(ss_tot)

    def result(self):
        return 1.0 - (self.sum_sq_res / (self.sum_sq_tot + 1e-7))

    def reset_state(self):
        self.sum_sq_res.assign(0.0)
        self.sum_sq_tot.assign(0.0)
        self.mean_y.assign(0.0)
        self.count.assign(0.0)