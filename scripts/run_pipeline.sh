#!/usr/bin/env bash
set -euo pipefail

echo "[pipeline] starting..."

# TODO: wire these to your real module entrypoints
# Example placeholders (replace with your actual commands):
# python -m src.data.audit
# python -m src.models.train --config configs/train_custom.yaml
# python -m src.models.train --config configs/train_effnet.yaml
# python -m src.evaluation.compare
# python -m src.xai.gradcam
# python -m src.inference.predict --config configs/inference.yaml

echo "[pipeline] done."
