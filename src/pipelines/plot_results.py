"""Render the probe/robustness matrices under result/criteria/<exp> as figures under figure/generated/<exp>/<task>/.

    python -m src.pipelines.plot_results --task regression                 # clean experiment
    python -m src.pipelines.plot_results --task classification --exp noise # noise experiment

Layout: one large panel per facet plus a mean overview (r0..r9 / c0..c4);
output goes to figure/generated/<task>[ -noise]/<Model>/, so different experiments and
models never overwrite each other.
"""
import argparse

from ..viz.plots import plot_summary_grid, run
from ..common.config import EXPS, TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--exp", choices=EXPS, default="clean",
                    help="clean=main experiment without noise training, noise=noise-augmented training branch")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--grid", action="store_true",
                    help="only build the 2x2 grid (png) of the four-network mean overview, skip the per-facet panels")
    args = ap.parse_args()
    if args.grid:
        plot_summary_grid(args.task, args.exp, args.model)
    else:
        run(args.task, args.exp, args.model)


if __name__ == "__main__":
    main()
