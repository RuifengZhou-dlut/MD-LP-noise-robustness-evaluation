"""End-to-end runner of Section 5.3.1 (full ten-class CIFAR-10).

The main pipeline is dataset-agnostic: it only needs ``--task
classification_full``.  This wrapper runs the five stages in order so that the
experiment can be reproduced with one command.

Usage::

    python -m src.pipelines.run_full_cifar10 --models VGG-11 Res-18 ViT-8 Mix-8
    python -m src.pipelines.run_full_cifar10 --skip-train --skip-data   # evaluation only

Before running, build the data (``python -m src.data.build_dataset --dataset
cifar10_full``) and make sure the full CIFAR-10 models exist under
``model/classification_full/clean``.
"""
import argparse
import subprocess
import sys

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]


def run(module, args):
    cmd = [sys.executable, "-m", module] + args
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-attack", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        for m in args.models:
            extra = [] if args.epochs is None else ["--epochs", str(args.epochs)]
            run("src.pipelines.run_train", ["--task", "classification_full", "--model", m] + extra)

    if not args.skip_data:
        run("src.noise.build_noise", ["--task", "classification_full", "--what", "perturb",
                                      "--repeats", str(args.repeats)])
        if not args.skip_attack:
            run("src.noise.build_noise", ["--task", "classification_full", "--what", "attack",
                                          "--repeats", str(args.repeats), "--model"] + args.models)

    for m in args.models:
        run("src.pipelines.run_probe_eval", ["--task", "classification_full", "--model", m,
                                             "--exp", "clean"])
    run("src.pipelines.plot_clean", ["--task", "classification_full"])


if __name__ == "__main__":
    main()
