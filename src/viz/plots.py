"""Unified plotting entry point: every figure is written under figure/generated/ by default.

The main experiment figures share one layout:
  - each figure draws the six conditions in a single coordinate system
    (x = substructure depth, y = ACC)
  - fixed condition colors: base black / gamma_p orange / gamma_g red /
    gamma_s blue / gamma_m green / gamma_o purple
  - each column (facet) yields one large figure plus one mean-only summary figure:
    regression prefix r0..r9, classification prefix c0..c4
  - uniform size (22,12), ylim(-0.15,1.05), large fonts, legend outside at the
    lower right, dashed grid

Data sources (probe and robustness results are stored separately and merged into a
single figure here):
  result/probe/<exp>/<task>/<Model>/probe_acc.pkl -> base curves
  result/criteria/<exp>/<task>/<Model>_robustness.pkl -> curves per perturbation
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.config import figure_dir

# ---------------------------------------------------------------- plotting constants
MODEL_XTICK_LABELS = {
    "VGG":    ["1-C", "2-C", "3-C", "4-C", "5-C", "6-C",
               "7-C", "8-C", "9-D", "10-D", "O"],
    "ResNet": ["1-S", "2-B", "3-B", "4-B", "5-B", "6-B",
               "7-B", "8-B", "9-B", "O"],
    "ViT":    ["1-E", "2-T", "3-T", "4-T", "5-T", "6-T",
               "7-T", "8-T", "9-T", "10-H", "O"],
    "Mix":    ["1-E", "2-M", "3-M", "4-M", "5-M", "6-M",
               "7-M", "8-M", "9-M", "10-H", "O"],
}

CONDITION_COLORS = {
    "base":   "black",
    "attack": "orange",
    "gauss":  "red",
    "salt":   "blue",
    "move":   "green",
    "occ":    "purple",
}
CONDITION_LABELS = {
    "base":   "base",
    "attack": chr(947) + "$_{p}$",
    "gauss":  chr(947) + "$_{g}$",
    "salt":   chr(947) + "$_{s}$",
    "move":   chr(947) + "$_{m}$",
    "occ":    chr(947) + "$_{o}$",
}


def load_result_pkl(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_combined(task, exp, model):
    """Merge probe (base) and robustness (perturbed) results into a plotting dict
    {condition: matrix}."""
    from ..common.config import probe_acc_pkl, robust_pkl

    d = {}
    p = probe_acc_pkl(task, model, exp)
    if p.exists():
        d.update(load_result_pkl(p))
    r = robust_pkl(task, model, exp)
    if r.exists():
        d.update(load_result_pkl(r))
    return d


def _apply_common_style(ax, x, xtick_labels, y_bottom=0.0):
    ax.grid(True, linestyle="--", linewidth=2.0)
    ax.set_xlabel("substructures", size=50)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("ACC", size=50)
    # keep the lower bound close to the data minimum so the axis keeps no large
    # blank region below 0.3
    ax.set_ylim(y_bottom, 1.02)
    ax.tick_params(labelsize=35)


def _add_figure_legend(fig, ax):
    """Place the legend below the axes so it does not take up blank space inside them."""
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=35,
                     bbox_to_anchor=(0.5, 0.0), framealpha=0)
    leg.get_frame().set_alpha(0)


def plot_curves(matrices, model_name, save_dir, prefix):
    """Draw the curve figures: matrices[cond] has shape (n_substructures, n_cols).

    Writes prefix0.pdf (mean over facets for every condition) and prefix1..n.pdf
    (one figure per facet) and returns the list of files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in CONDITION_COLORS if k in matrices]
    xtick_labels = MODEL_XTICK_LABELS[model_name]

    num_sub = matrices[keys[0]].shape[0]
    num_cols = matrices[keys[0]].shape[1]
    x = np.arange(num_sub)

    # summary figure: facet mean of every condition
    fig = plt.figure(1, figsize=(22, 12))
    ax0 = plt.gca()
    means = np.stack([np.asarray(matrices[k], dtype=np.float64).mean(axis=1) for k in keys])
    for key, mean in zip(keys, means):
        ax0.plot(x, mean, color=CONDITION_COLORS[key],
                 linewidth=6.0, marker="o", markersize=20, label=CONDITION_LABELS[key])
    _apply_common_style(ax0, x, xtick_labels, max(0.0, means.min() - 0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    _add_figure_legend(fig, ax0)

    out = [save_dir / f"{prefix}0.pdf"]
    plt.savefig(out[0], dpi=200, bbox_inches="tight")
    plt.savefig(str(out[0]).replace(".pdf", ".png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # one figure per facet
    for col in range(num_cols):
        fig = plt.figure(col + 2, figsize=(22, 12))
        ax = plt.gca()
        col_vals = np.stack([np.asarray(matrices[k], dtype=np.float64)[:, col] for k in keys])
        for key, vals in zip(keys, col_vals):
            ax.plot(x, vals, color=CONDITION_COLORS[key],
                    linewidth=6.0, marker="o", markersize=20, label=CONDITION_LABELS[key])
        _apply_common_style(ax, x, xtick_labels, max(0.0, col_vals.min() - 0.02))
        fig.tight_layout(rect=(0, 0.10, 1, 0.98))
        _add_figure_legend(fig, ax)
        p = save_dir / f"{prefix}{col + 1}.pdf"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.savefig(str(p).replace(".pdf", ".png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def _figure_dir(task, exp):
    """Single figure location: figure/generated/<exp>/<task>."""
    from ..common.config import normalize_exp
    return figure_dir(task, exp)


def run(task, exp="clean", model_keys=None):
    """Read probe + robustness data under criteria/<exp> and render the figures."""
    from ..common.config import MODELS, probe_acc_pkl

    task_prefix = "r" if task == "regression" else "c"
    model_prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}
    outputs = []
    keys = [k for k in (model_keys or list(MODELS)) if probe_acc_pkl(task, k, exp).exists()]
    for k in (model_keys or list(MODELS)):
        if k not in keys:
            print(f"[skip] {probe_acc_pkl(task, k, exp)} not found")
    for key in keys:
        subdir = _figure_dir(task, exp) / key
        outs = plot_curves(load_combined(task, exp, key),
                                  model_prefix[key], subdir, task_prefix)
        outputs += outs
        print(f"[ok] {key}: {len(outs)} figures -> {subdir}")
    return outputs


def plot_summary_grid(task, exp="clean", model_keys=None):
    """Combine the mean summary figure (the r0/c0 content) of the four networks into
    a single 2x2 panel and save it as png."""
    from ..common.config import MODELS

    task_prefix = "r" if task == "regression" else "c"
    model_prefix = {"VGG-11": "VGG", "Res-18": "ResNet", "ViT-8": "ViT", "Mix-8": "Mix"}
    keys = model_keys or list(MODELS)

    fig, axes = plt.subplots(2, 2, figsize=(40, 24))
    loaded = {k: load_combined(task, exp, k) for k in keys}
    # all four panels share one lower bound (taken from the facet means actually
    # plotted), so they stay comparable without leaving blank space
    means_by_model = {
        k: {c: np.asarray(loaded[k][c], dtype=np.float64).mean(axis=1)
            for c in CONDITION_COLORS if c in loaded[k]}
        for k in keys
    }
    y_bottom = max(0.0, min(v.min() for m in means_by_model.values() for v in m.values()) - 0.02)
    for ax, key in zip(axes.ravel(), keys):
        conds = [k for k in CONDITION_COLORS if k in loaded[key]]
        xtick_labels = MODEL_XTICK_LABELS[model_prefix[key]]
        x = np.arange(loaded[key][conds[0]].shape[0])

        for cond in conds:
            ax.plot(x, means_by_model[key][cond], color=CONDITION_COLORS[cond],
                    linewidth=6.0, marker="o", markersize=18, label=CONDITION_LABELS[cond])
        ax.grid(True, linestyle="--", linewidth=2.0)
        ax.set_title(f"{model_prefix[key]} ({task}, {exp})", size=40)
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("substructures", size=36)
        ax.set_ylabel("ACC", size=36)
        ax.set_ylim(y_bottom, 1.02)
        ax.tick_params(labelsize=26)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    leg = fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=34,
                     bbox_to_anchor=(0.5, 0.0), framealpha=0)
    leg.get_frame().set_alpha(0)

    out_dir = _figure_dir(task, exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task_prefix}0_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] -> {out}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("regression", "classification"), required=True)
    ap.add_argument("--exp", choices=("clean", "noise"), default="clean")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--grid", action="store_true",
                    help="only build the 2x2 grid of the four network mean summaries "
                         "(png); skip the per-facet figures")
    args = ap.parse_args()
    if args.grid:
        plot_summary_grid(args.task, args.exp, args.model)
    else:
        run(args.task, args.exp, args.model)


if __name__ == "__main__":
    main()


