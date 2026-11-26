"""Plots specific to the noise branch: clean/noise network accuracy radar + substructure mean noise-metric comparison.

    python -m src.pipelines.plot_noise --task classification
    python -m src.pipelines.plot_noise --task regression --model VGG-11

Outputs: figure/generated/noise/<task>/<Model>/radar_accuracy.png + noise_metric_compare.png,
plus figure/generated/noise/<task>/<task>_radar_grid.png.
"""
import argparse

from ..viz.noise_plots import run
from ..common.config import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--exp", default="noise",
                    help="figures are written under figure/generated/<exp>/<task>/; defaults to noise")
    ap.add_argument("--model", nargs="*", default=None)
    args = ap.parse_args()
    run(args.task, args.exp, args.model)


if __name__ == "__main__":
    main()
