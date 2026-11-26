#!/usr/bin/env bash
# Run the add-on experiments that only need the stored main results.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.add_exp.probe_stability.weight_vs_samples.plot_stability
python -m src.add_exp.probe_stability.weight_vs_noise.plot_stability
python -m src.add_exp.s_sensitivity.make_figure
python -m src.add_exp.s_sensitivity.make_rank_table
python -m src.add_exp.s_sensitivity.make_rank_table_alt
