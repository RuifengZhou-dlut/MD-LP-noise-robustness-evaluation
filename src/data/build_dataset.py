"""Build the processed datasets from the original CIFAR-10 / UTKFace sources.

The repository already ships the exact paper splits for the four-class CIFAR
subset and for UTKFace (``dataset/clean/``).  This script is only needed when
those files are missing or when the full ten-class CIFAR-10 split has to be
recreated (Section 5.3.1).

Examples::

    python -m src.data.build_dataset --dataset cifar10_full
    python -m src.data.build_dataset --dataset cifar10_subset --n-per-class 5000
    python -m src.data.build_dataset --dataset utkface \
        --utkface-tar dataset/raw/UTKFace.tar.gz \
        --crop-tar dataset/raw/crop_part1.tar.gz
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from ..common.config import CLEAN_ROOT, REPO_ROOT
from ..common.seeding import BASE_SEED, derive_seed

CLASSES_4 = (0, 1, 2, 3)          # airplane, automobile, ship, truck
IMAGE_SIZE = 32


def _write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[saved] {path} ({path.stat().st_size / 1e6:.1f} MB)")


def build_cifar_full():
    """Full ten-class CIFAR-10 (50000 train / 10000 test) as uint8."""
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    out = CLEAN_ROOT / "classification_full" / "data.npz"
    _write_npz(out, x_train=x_train.astype(np.uint8),
               y_train=y_train.reshape(-1).astype(np.int32),
               x_test=x_test.astype(np.uint8),
               y_test=y_test.reshape(-1).astype(np.int32))
    print("[note] this file is ~150 MB; it is ignored by git and can be rebuilt on demand.")


def build_cifar_subset(n_per_class=5000, seed=None):
    """Four-class CIFAR-10 subset used by the main experiment.

    The archived split in ``dataset/clean/classification`` is the authoritative
    one; this builder reproduces the same class counts with a deterministic
    per-class subsample.
    """
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    rng = np.random.default_rng(BASE_SEED if seed is None else seed)

    tr_idx, te_idx = [], []
    for c in CLASSES_4:
        pool = np.where(y_train == c)[0]
        tr_idx.append(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False))
        te_idx.append(np.where(y_test == c)[0])
    tr_idx = np.sort(np.concatenate(tr_idx))
    te_idx = np.sort(np.concatenate(te_idx))

    out = CLEAN_ROOT / "classification" / "data.npz"
    _write_npz(out, x_train=x_train[tr_idx].astype(np.uint8),
               y_train=y_train[tr_idx].astype(np.int32),
               x_test=x_test[te_idx].astype(np.uint8),
               y_test=y_test[te_idx].astype(np.int32))


def _load_utkface(tars, target_size=128):
    images, ages = [], []
    for tar_path in tars:
        tar_path = Path(tar_path)
        if not tar_path.exists():
            raise FileNotFoundError(f"{tar_path} not found (see dataset/raw/README.md)")
        with tarfile.open(str(tar_path), "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".png")):
                    continue
                parts = member.name.split("/")[-1].split("_")
                if len(parts) < 4:
                    continue
                try:
                    age = int(parts[0])
                except ValueError:
                    continue
                from PIL import Image
                img = Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB")
                images.append(np.asarray(img.resize((target_size, target_size)), dtype=np.float32) / 255.0)
                ages.append(age)
        print(f"[load] {tar_path.name}: {len(images)} images in total")
    return np.stack(images), np.asarray(ages, dtype=np.int64)


def _downsample(x, size=IMAGE_SIZE):
    import tensorflow as tf
    out = np.empty((x.shape[0], size, size, x.shape[-1]), dtype=np.float32)
    for s in range(0, x.shape[0], 2000):
        e = min(s + 2000, x.shape[0])
        out[s:e] = tf.image.resize(x[s:e], (size, size), method="area").numpy()
    return out


def build_utkface(tars, n_train=20000, n_test=2000):
    """Stratified UTKFace split, written as two <100 MB shards."""
    from sklearn.model_selection import StratifiedShuffleSplit

    x, age = _load_utkface(tars)
    bins = (age // 5).astype(np.int32)
    seed = derive_seed("split", "regression", "v2", base=BASE_SEED)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, test_size=n_test,
                                 random_state=seed)
    tr, te = next(sss.split(x, bins))

    x_train = _downsample(x[tr])
    x_test = _downsample(x[te])
    y_train = (age[tr] + 1) / 120.0
    y_test = (age[te] + 1) / 120.0

    outdir = CLEAN_ROOT / "regression" / "parts"
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        s, e = i * 10000, (i + 1) * 10000
        kw = dict(x_train=x_train[s:e].astype(np.float16),
                  y_train=y_train[s:e].astype(np.float64))
        if i == 0:
            kw.update(x_test=x_test.astype(np.float16), y_test=y_test.astype(np.float64))
        _write_npz(outdir / f"part-{i:03d}.npz", **kw)
    meta = {"seed": int(seed), "n_train": n_train, "n_test": n_test,
            "raw_tars": [str(t) for t in tars]}
    (outdir.parent / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("cifar10_full", "cifar10_subset", "utkface"), required=True)
    ap.add_argument("--n-per-class", type=int, default=5000)
    ap.add_argument("--utkface-tar", default=str(REPO_ROOT / "dataset" / "raw" / "UTKFace.tar.gz"))
    ap.add_argument("--crop-tar", default=str(REPO_ROOT / "dataset" / "raw" / "crop_part1.tar.gz"))
    args = ap.parse_args()

    if args.dataset == "cifar10_full":
        build_cifar_full()
    elif args.dataset == "cifar10_subset":
        build_cifar_subset(args.n_per_class)
    else:
        build_utkface([args.utkface_tar, args.crop_tar])


if __name__ == "__main__":
    main()
