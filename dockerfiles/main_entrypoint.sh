#!/usr/bin/env bash
set -euo pipefail

cd /app

data_root="${DATA_ROOT:-data/processed}"
if [[ "$data_root" == /gcs/* ]]; then
    echo "Using mounted GCS data at $data_root; skipping dvc pull."
else
    if [[ ! -f data/processed.dvc ]]; then
        echo "data/processed.dvc not found. Run the data container first." >&2
        exit 1
    fi
    dvc_cache_dir="${DVC_CACHE_DIR:-/tmp/dvc-cache}"
    mkdir -p "$dvc_cache_dir"
    dvc config cache.dir "$dvc_cache_dir"
    if ! dvc pull data/processed.dvc; then
        echo "dvc pull failed. Check GCS credentials and GOOGLE_CLOUD_PROJECT." >&2
        exit 1
    fi
fi

extra_args=()
output_root="/tmp/outputs"
export OUTPUT_ROOT="$output_root"
mkdir -p "$output_root"
hydra_run_dir="${output_root}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
extra_args+=("hydra.run.dir=${hydra_run_dir}")

RUN_SNAKEVIZ="${RUN_SNAKEVIZ:-0}" \
RUN_TENSORBOARD="${RUN_TENSORBOARD:-0}" \
OPEN_TENSORBOARD="${OPEN_TENSORBOARD:-0}" \
exec python -u src/mlops_g116/main.py "${extra_args[@]}" "$@"
