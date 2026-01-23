#!/usr/bin/env bash

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

pre-commit install --install-hooks

if command -v pip >/dev/null 2>&1; then
    pip install -e . || echo "pip install -e . failed."
fi

if command -v gcloud >/dev/null 2>&1; then
    if [ ! -f /root/.config/gcloud/application_default_credentials.json ]; then
        echo "gcloud ADC not found. Run: gcloud auth application-default login"
    fi
fi

if command -v dvc >/dev/null 2>&1; then
    dvc pull || echo "dvc pull failed. Configure gcloud auth and rerun."
fi
