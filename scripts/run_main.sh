#!/usr/bin/env bash
# Run the full main experiment (train -> noise -> probes -> figures -> SE).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"VGG-11 Res-18 ViT-8 Mix-8"}
TASKS=${TASKS:-"classification regression"}

for task in $TASKS; do
  for m in $MODELS; do
    python -m src.pipelines.run_train --task "$task" --model "$m"
  done
  python -m src.noise.build_noise --task "$task" --what all --repeats 10
  for m in $MODELS; do
    python -m src.pipelines.run_probe_eval --task "$task" --model "$m" --exp clean
  done
  python -m src.pipelines.plot_clean --task "$task"
done

python -m src.eval.aggregate
python -m src.eval.se_heatmap
python -m src.eval.scoring
python -m src.eval.make_score_table
python -m src.pipelines.plot_noise --task classification
python -m src.pipelines.plot_noise --task regression
