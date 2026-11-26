"""Train the four backbone networks.

Hyperparameters (the same for every backbone) live in config.MODELS[model][task]
and config.TRAIN_DEFAULTS.
"""
import argparse
from pathlib import Path

import tensorflow as tf

from ..common.config import EXPS, MODELS, TASKS, TASK_CFG, TRAIN_DEFAULTS, checkpoint_path
from ..data.datasets import load_train_xy, to_onehot
from ..models.network import build_model
from ..common.seeding import BASE_SEED, derive_seed, set_global_seed
from ..models.trainer import LastNSaver, RSquared, build_optimizer, make_test_ds, make_train_ds


def run(task, model_key, epochs=None, batch_size=None, out_path=None, noise_aug=False,
        seed=None, deterministic=False, exp="clean"):
    # classification_full reuses the training hyperparameters of classification
    cfg = MODELS[model_key].get(task) or MODELS[model_key]["classification"]
    d = TRAIN_DEFAULTS
    epochs = epochs or d["epochs"]
    batch_size = batch_size or d["batch_size"]

    base_seed = BASE_SEED if seed is None else int(seed)
    data_seed = derive_seed("train", exp, task, model_key,
                            "noise" if noise_aug else "clean", base=base_seed)
    set_global_seed(data_seed, deterministic=deterministic)

    x_train, y_train, x_test, y_test = load_train_xy(task, exp)
    if task in ("classification", "classification_full"):
        n_cls = TASK_CFG[task]["num_classes"]
        y_train = to_onehot(y_train, n_cls)
        y_test = to_onehot(y_test, n_cls)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = ["accuracy"]
    else:
        loss = "mse"
        metrics = [RSquared()]

    model = build_model(model_key, task)

    total_steps = epochs * (x_train.shape[0] // batch_size)
    warmup_steps = d["warmup_epochs"] * (x_train.shape[0] // batch_size)
    optimizer = build_optimizer(cfg, total_steps, warmup_steps)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_ds = make_train_ds(x_train, y_train, batch_size=batch_size,
                             image_size=d["image_size"], pad=d["pad"],
                             mixup_alpha=cfg["mixup"],
                             use_noise=noise_aug, seed=derive_seed("shuffle", data_seed))
    test_ds = make_test_ds(x_test, y_test, batch_size=batch_size)

    model.fit(train_ds, epochs=epochs, validation_data=test_ds, verbose=2,
              callbacks=[LastNSaver(n=d["last_n"])])

    out_path = Path(out_path) if out_path else checkpoint_path(task, model_key, exp=exp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"saved -> {out_path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--exp", choices=EXPS, default="clean",
                    help="clean=main experiment / noise=noise-augmented training")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--noise-aug", action="store_true",
                    help="enable noise-augmented training (gauss/salt; every batch is doubled with clean+noisy copies)")
    ap.add_argument("--seed", type=int, default=BASE_SEED,
                    help="master random seed (same seed + same environment reproduces the same weights)")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable TF op determinism (stricter reproducibility, slower training)")
    args = ap.parse_args()
    run(args.task, args.model, args.epochs, args.batch_size, out_path=args.out,
        noise_aug=args.noise_aug, seed=args.seed, deterministic=args.deterministic,
        exp=args.exp)


if __name__ == "__main__":
    main()
