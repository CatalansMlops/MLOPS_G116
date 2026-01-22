#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ ! -d data/processed ]]; then
    echo "data/processed not found. Build the image with data/processed present." >&2
    exit 1
fi

if [[ ! -f models/model.pth ]]; then
    echo "models/model.pth not found. Mount models/ or rebuild with a model present." >&2
    exit 1
fi

RUN_SNAKEVIZ="${RUN_SNAKEVIZ:-0}" \
RUN_TENSORBOARD="${RUN_TENSORBOARD:-0}" \
OPEN_TENSORBOARD="${OPEN_TENSORBOARD:-0}" \
exec python -u src/mlops_g116/evaluate.py "$@"
