#!/usr/bin/env bash

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

pre-commit install --install-hooks

if command -v dvc >/dev/null 2>&1; then
    dvc pull || echo "dvc pull failed. Configure gcloud auth and rerun."
fi
