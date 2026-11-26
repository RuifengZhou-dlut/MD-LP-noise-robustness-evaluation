"""Plots for the clean branch: 2x2 overall metrics + 1x4 per facet, with a single legend row below.

    python -m src.pipelines.plot_clean --task classification
    python -m src.pipelines.plot_clean --task regression
"""
import argparse

from ..viz.clean_plots import run
from ..common.config import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--exp", default="clean")
    ap.add_argument("--model", nargs="*", default=None)
    args = ap.parse_args()
    run(args.task, args.exp, args.model)


if __name__ == "__main__":
    main()
