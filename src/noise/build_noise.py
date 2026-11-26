"""Generate the four perturbation sets and the PGD adversarial examples.

Default parameters:
    gauss: a=0.2   salt: n=64   move: degree=15   occ: v=192   repeats=10
    PGD  : eps=2.0 alpha=0.2 steps=20 repeats=10

Seed convention: perturbation group i has one group seed noise_group_seed(task, i);
all 5 disturbances of that group (PGD/gauss/salt/move/occ) are derived from it and
recorded in that group's seed.json. Noise seeds are kept separate from training
seeds (NOISE_SEED vs BASE_SEED), so changing the training seed leaves the noise
data unchanged.
"""
import argparse
import json

import numpy as np

from ..common.config import (EXPS, MODELS, NOISE_ROOT, TASKS, TASK_CFG, attack_dir,
                             attack_npy, perturbation_file)
from ..noise.attacks import PGDAdversaryL2
from ..data.datasets import load_train_xy
from ..noise.perturbations import (PERTURBATIONS, gauss_noise, motion_blur,
                                  occlusion_noise, salt_noise)
from ..common.seeding import NOISE_SEED, derive_seed, noise_group_seed, set_global_seed

PERTURB_PARAMS = dict(gauss_a=0.2, salt_n=64, move_degree=15, occ_v=192)


def _write_seed(d: "Path", task, repeat, group_seed):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "seed.json", "w", encoding="utf-8") as f:
        json.dump({"task": task, "repeat": repeat, "group_seed": int(group_seed)}, f,
                  ensure_ascii=False, indent=2)


def build_perturbations(task, repeats=10, params=None, overwrite=False,
                        exp="clean", seed=NOISE_SEED, start=0):
    p = dict(PERTURB_PARAMS)
    if params:
        p.update(params)

    x_train, _, _, _ = load_train_xy(task, exp)

    for i in range(start, start + repeats):
        paths = {n: perturbation_file(task, n, i) for n in PERTURBATIONS}
        if not overwrite and all(p.exists() for p in paths.values()):
            print(f"[perturbation] {task} repeat {i} exists, skipped (--overwrite to redo)")
            continue

        gseed = noise_group_seed(task, i, base=seed)
        set_global_seed(gseed)
        arrays = {
            "gauss": gauss_noise(x_train, p["gauss_a"]),
            "salt": salt_noise(x_train, p["salt_n"]),
            "move": motion_blur(x_train, p["move_degree"]),
            "occ": occlusion_noise(x_train, p["occ_v"]),
        }
        for name, arr in arrays.items():
            path = paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(arr), allow_pickle=True)
        seed_dir = NOISE_ROOT / "_seeds" / task
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / f"{i:02d}.json").write_text(
            json.dumps({"task": task, "repeat": i, "group_seed": int(gseed)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[perturbation] {task} repeat {i} saved (seed={gseed})")


def build_attacks(task, repeats=10, model_keys=None, eps=2.0, alpha=0.2, steps=20,
                  seed=NOISE_SEED, overwrite=False, exp="clean", start=0):
    from ..models.loader import load_model

    model_keys = model_keys or list(MODELS)
    x_train, y_train, _, _ = load_train_xy(task, exp)
    models = {key: load_model(task, key, exp=exp) for key in model_keys}

    for i in range(start, start + repeats):
        d = attack_dir(task, i, exp=exp)
        d.mkdir(parents=True, exist_ok=True)
        gseed = noise_group_seed(task, i, base=seed)
        _write_seed(d, task, i, gseed)

        for key in model_keys:
            out = attack_npy(task, key, i, exp=exp)
            if not overwrite and out.exists():
                print(f"[attack] {task} repeat {i} {key} exists, skipped")
                continue

            set_global_seed(derive_seed("pgd", key, base=gseed))
            attacker = PGDAdversaryL2(models[key], task=task,
                                      num_classes=None if task == "regression" else TASK_CFG[task]["num_classes"],
                                      seed=derive_seed("pgd-start", key, base=gseed))
            x_adv = attacker.generate_adversarial_dataset(
                x_train, y_train, eps=eps, alpha=alpha, steps=steps,
                batch_size=128, shuffle=True, random_start=True)
            np.save(out, x_adv.numpy(), allow_pickle=True)
            print(f"[attack] {task} repeat {i} {key} saved (group_seed={gseed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--exp", choices=EXPS, default="clean")
    ap.add_argument("--what", choices=("perturb", "attack", "all"), default="all")
    ap.add_argument("--model", nargs="*", default=None,
                    help="generate PGD only for the listed networks (default: all 4)")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="start group index (with --repeats 1 it generates group by group, so a crash does not lose progress)")
    ap.add_argument("--seed", type=int, default=NOISE_SEED,
                    help="base seed of the noise groups (separate from the training seed)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing perturbation/attack npy files (default: skip, so existing data is kept)")
    args = ap.parse_args()

    if args.what in ("perturb", "all"):
        build_perturbations(args.task, repeats=args.repeats or 10,
                            overwrite=args.overwrite, exp=args.exp, seed=args.seed,
                            start=args.start)
    if args.what in ("attack", "all"):
        build_attacks(args.task, repeats=args.repeats or 10,
                      model_keys=args.model, overwrite=args.overwrite,
                      exp=args.exp, seed=args.seed, start=args.start)


if __name__ == "__main__":
    main()
