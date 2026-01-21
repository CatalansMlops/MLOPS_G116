#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ ! -f data/processed.dvc ]]; then
    echo "data/processed.dvc not found. Run the data container first." >&2
    exit 1
fi

dvc pull data/processed.dvc

extra_args=()
output_root="/tmp/outputs"
export OUTPUT_ROOT="$output_root"
mkdir -p "$output_root"
hydra_run_dir="${output_root}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
extra_args+=("hydra.run.dir=${hydra_run_dir}")

RUN_SNAKEVIZ="${RUN_SNAKEVIZ:-0}" \
RUN_TENSORBOARD="${RUN_TENSORBOARD:-0}" \
OPEN_TENSORBOARD="${OPEN_TENSORBOARD:-0}" \
exec python -u src/mlops_g116/train_hydra.py "${extra_args[@]}" "$@"
