#!/usr/bin/env bash
# Fast self-check: data shapes, model loading, one probe evaluation, one add-on.
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "from src.data.datasets import load_train_xy; x,y,_,_=load_train_xy('classification'); print('data', x.shape, y.shape)"
python -c "from src.models.loader import load_model; m=load_model('classification','VGG-11'); print('model', m.output_shape)"
rm -rf /tmp/mdlp_smoke && mkdir -p /tmp/mdlp_smoke
python -m src.pipelines.run_probe_eval --task classification --model VGG-11 \
  --conditions base --max-layers 2 --out-dir /tmp/mdlp_smoke
python -m src.eval.scoring --out /tmp/mdlp_smoke
echo "[ok] smoke test passed"
