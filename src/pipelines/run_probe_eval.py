"""Compute the condition x all-layers x n-facets accuracy matrix (streaming, no intermediate layer cache) and save it.

    exp=clean  6 conditions (base/attack/gauss/salt/move/occ), 10 repeats
    exp=noise  3 conditions (base/gauss/salt, single repeat)

Outputs (probe and robustness are stored separately):

    result/probe/<exp>/<task>/<Model>/probe_acc.pkl        [base] per-layer probe accuracy
    result/probe/<exp>/<task>/<Model>/<layer>_w.npy|_b.npy  probe weights/bias (same level)
    result/criteria/<exp>/<task>/<Model>_robustness.pkl    accuracy per perturbation condition
    result/criteria/<exp>/<task>/<Model>_std.pkl           std across repeats
    result/criteria/<exp>/<task>/<Model>_drop.pkl          base - cond (performance drop)
    result/criteria/<exp>/<task>.csv                       summary table

Each dataset is forwarded only a few times (base set: 2 passes, moment statistics
then projection; each perturbation condition: 1 pass), and only the N x n_splits
projections and predictions are kept in memory. On disk only w/b (MB scale) and
the result pkl remain.
"""
import argparse
import math
import pickle
from pathlib import Path

import numpy as np

from ..common.config import (CONDITIONS, MODELS, NOISE_CONDITIONS, TASKS, TASK_CFG,
                             attack_npy, drop_pkl, normalize_exp, perturbation_file,
                             probe_acc_pkl, probe_weights_dir, robust_pkl,
                             robust_se_pkl, robust_std_pkl, repeat_raw_dir, summary_csv)
from ..data.datasets import load_probe_labels, load_train_xy
from ..models.loader import layer_names, load_model
from ..probe.evaluate import evalu_prepare
from ..probe.streaming import (base_accuracy, eval_condition, final_from_preds,
                               make_view, select_from_base, solve_base)
from ..probe.weights import save_wb_per_layer


def _wb_ready(wb_dir, layer_list):
    return all((Path(wb_dir) / f"{name}_{s}.npy").exists()
               for name in layer_list for s in ("w", "b"))


def _dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _write_summary(task, exp):
    """Rebuild the summary CSV from the probe_acc/robustness files already under criteria/<exp>/.

    Column meaning:
      std_final : std across repeats of a single measurement, s (ddof=0); how much the
                  value moves when a different noise draw is used;
      se_final  : standard error of the mean, SE = s/sqrt(n); the uncertainty of the
                  tabulated value (average over n repeats).
    Significance of a metric should be judged from se_final.
    """
    rows = ["model,condition,n_layers,layer_mean,final,std_final,drop_layer_mean,drop_final,se_final"]
    for model in MODELS:
        pp, rp = probe_acc_pkl(task, model, exp), robust_pkl(task, model, exp)
        if not pp.exists():
            continue
        base = np.asarray(pickle.load(open(pp, "rb"))["base"], dtype=np.float64)
        rows.append("%s,base,%d,%.4f,%.4f,,0.0000,0.0000," % (
            model, base.shape[0] - 1, base[:-1].mean(), base[-1].mean()))
        robust = pickle.load(open(rp, "rb")) if rp.exists() else {}
        stds = {}
        sp = robust_std_pkl(task, model, exp)
        if sp.exists():
            stds = pickle.load(open(sp, "rb"))
        ses = {}
        ep = robust_se_pkl(task, model, exp)
        if ep.exists():
            ses = pickle.load(open(ep, "rb"))
        for cond, mat in robust.items():
            m = np.asarray(mat, dtype=np.float64)
            sd = stds.get(cond)
            se = ses.get(cond)
            rows.append("%s,%s,%d,%.4f,%.4f,%s,%.4f,%.4f,%s" % (
                model, cond, m.shape[0] - 1, m[:-1].mean(), m[-1].mean(),
                "" if sd is None else "%.4f" % np.asarray(sd)[-1].mean(),
                (base[:-1] - m[:-1]).mean(), (base[-1] - m[-1]).mean(),
                "" if se is None else "%.4f" % np.asarray(se)[-1].mean()))
    path = summary_csv(task, exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def run(task, model_key, exp="clean", conditions=None, max_layers=None, save=True,
        max_repeats=None, out_dir=None, batch_size=256):
    exp = normalize_exp(exp)
    cfg = TASK_CFG[task]
    n = cfg["n_splits"]
    grouping = cfg["grouping"]
    conditions = conditions or (NOISE_CONDITIONS if exp == "noise" else CONDITIONS)

    y = load_probe_labels(task, exp)
    x_train, y_train, _, _ = load_train_xy(task, exp)

    model = load_model(task, model_key, exp=exp)
    all_layers = layer_names(model, model_key)
    layer_list = all_layers[:max_layers] if max_layers else all_layers
    k_eval = len(layer_list)

    thresholds, Y_list = evalu_prepare(y, n=n, grouping=grouping)
    view = make_view(model, all_layers)

    wb_dir = probe_weights_dir(task, model_key, exp)
    need_save_wb = not _wb_ready(wb_dir, all_layers)

    # Base set: moment statistics -> (w, b), then one projection pass for P and the full-network predictions
    w_list, b_list, P_list, base_preds = solve_base(
        view, x_train, y, all_layers, n, grouping, thresholds,
        wb_cache=wb_dir, batch_size=batch_size)

    if need_save_wb:
        for name, w, b in zip(all_layers, w_list, b_list):
            save_wb_per_layer(w, b, name, str(wb_dir))
        print(f"[stream] w/b saved -> {wb_dir}")

    # Sample selection: final-layer projection + full-network prediction (same convention as evaluate.evalu_select)
    select_list = select_from_base(P_list[-1], b_list[-1], base_preds, y,
                                   thresholds, Y_list, grouping)

    out = {}
    out_std = {}   # std across repeats of the same shape (ddof=0); base has no repeats -> None
    out_se = {}    # standard error of the mean SE = s/sqrt(n); base has no repeats -> None
    used_n = {}    # number of repeats actually used per condition

    for condition in conditions:
        if condition == "base":
            mat = base_accuracy(P_list[:k_eval], b_list[:k_eval], Y_list, select_list, n)
            final = final_from_preds(task, base_preds, select_list, thresholds)
            out[condition] = np.concatenate([mat, final.reshape(1, n)], axis=0)
            out_std[condition] = None
            out_se[condition] = None
            used_n[condition] = None
            print(f"[{condition}] done, shape={out[condition].shape}")
            continue

        repeats = TASK_CFG[task]["repeats"][condition]
        if max_repeats:
            repeats = min(repeats, max_repeats)
        mats, finals = [], []

        for i in range(repeats):
            if condition == "attack":
                raw = attack_npy(task, model_key, i, exp=exp)
            else:
                raw = perturbation_file(task, condition, i)

            X = np.load(str(raw), allow_pickle=True)
            mat, preds = eval_condition(view, X, y_train, w_list[:k_eval], b_list[:k_eval],
                                        select_list, Y_list, batch_size=batch_size)
            mats.append(mat)
            finals.append(final_from_preds(task, preds, select_list, thresholds))
            del X

        stack = np.stack(mats, axis=0)                 # (repeats, L, n)
        final_arr = np.stack(finals, axis=0)           # (repeats, n)
        try:
            d = repeat_raw_dir(task)
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{model_key}_{condition}.npz", stack=stack, final=final_arr)
        except Exception as _e:
            print("dump failed", _e)
        out[condition] = np.concatenate([stack.mean(axis=0),
                                         final_arr.mean(axis=0).reshape(1, n)], axis=0)
        std = np.concatenate([stack.std(axis=0, ddof=0),
                              final_arr.std(axis=0, ddof=0).reshape(1, n)], axis=0)
        out_std[condition] = std
        # SE = s_{n-1}/sqrt(n) = s_n/sqrt(n-1) (the stored s uses ddof=0)
        out_se[condition] = std / math.sqrt(max(repeats - 1, 1))
        used_n[condition] = repeats
        print(f"[{condition}] done, shape={out[condition].shape}, "
              f"repeats={repeats}, mean_std={std.mean():.4f}, "
              f"mean_se={out_se[condition].mean():.4f}")

    if not save:
        return out, out_std

    base_mat = np.asarray(out["base"], dtype=np.float32)
    probe = {"base": base_mat}
    robust = {c: np.asarray(v, dtype=np.float32) for c, v in out.items() if c != "base"}
    robust_std = {c: (None if v is None else np.asarray(v, dtype=np.float32))
                  for c, v in out_std.items() if c != "base"}
    robust_se = {c: (None if v is None else np.asarray(v, dtype=np.float32))
                 for c, v in out_se.items() if c != "base"}
    drop = {c: base_mat - v for c, v in robust.items()}

    if out_dir:
        d = Path(out_dir)
        _dump(probe, d / f"{model_key}_probe_acc.pkl")
        _dump(robust, d / f"{model_key}_robustness.pkl")
        _dump(robust_std, d / f"{model_key}_robustness_std.pkl")
        _dump(robust_se, d / f"{model_key}_robustness_se.pkl")
        _dump(drop, d / f"{model_key}_robustness_drop.pkl")
        print(f"saved -> {d}/{model_key}_*.pkl")
        return out, out_std

    p_probe = probe_acc_pkl(task, model_key, exp)
    p_rob = robust_pkl(task, model_key, exp)
    _dump(probe, p_probe)
    _dump(robust, p_rob)
    _dump(robust_std, robust_std_pkl(task, model_key, exp))
    _dump(robust_se, robust_se_pkl(task, model_key, exp))
    _dump(drop, drop_pkl(task, model_key, exp))

    sc = _write_summary(task, exp)
    print(f"saved -> {p_probe}\nsaved -> {p_rob}\nsaved -> {sc}")
    return out, out_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--exp", default="clean",
                    help="clean=main experiment without noise training, noise=noise-augmented training branch")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="defaults: all 6 conditions for clean, base/gauss/salt for noise")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--max-repeats", type=int, default=None,
                    help="debug only: run the first k repeats (results differ from the official matrix)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run(args.task, args.model, args.exp, args.conditions, args.max_layers,
        max_repeats=args.max_repeats, out_dir=args.out_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
