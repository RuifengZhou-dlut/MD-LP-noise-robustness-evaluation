#!/usr/bin/env python3
"""Regenerate the machine-readable seed manifest (tools/seeds.json).

The manifest is derived from the same functions used by the pipeline, so it can
be used to audit that no seed has drifted.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import MODELS, TASKS  # noqa: E402
from src.common.seeding import (BASE_SEED, NOISE_SEED, derive_seed,  # noqa: E402
                                noise_group_seed)


def build():
    training = {}
    for task in TASKS:
        training[task] = {}
        for model in MODELS:
            exps = ["clean"] if task == "classification_full" else ["clean", "noise"]
            training[task][model] = {
                exp: derive_seed("train", exp, task, model,
                                 "noise" if exp == "noise" else "clean", base=BASE_SEED)
                for exp in exps
            }

    noise_group, pgd = {}, {}
    for task in TASKS:
        noise_group[task] = {str(i): noise_group_seed(task, i, base=NOISE_SEED)
                             for i in range(10)}
        pgd[task] = {}
        for model in MODELS:
            pgd[task][model] = {
                str(i): derive_seed("pgd", model, base=noise_group[task][str(i)])
                for i in range(10)
            }

    return {
        "base_seed": BASE_SEED,
        "noise_seed": NOISE_SEED,
        "training": training,
        "noise_group": noise_group,
        "pgd": pgd,
        "split_regression": derive_seed("split", "regression", "v2", base=BASE_SEED),
        "add_experiments": {
            "rotation_seed": 20261108,
            "weight_vs_samples_seed": 20261114,
            "weight_vs_noise_seed": 20261116,
            "feature_sites_seed": 20261108,
        },
    }


def main():
    out = ROOT / "tools" / "seeds.json"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
