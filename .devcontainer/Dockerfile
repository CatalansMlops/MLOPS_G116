# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc git curl gnupg ca-certificates && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install --no-install-recommends -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY requirements_dev.txt /tmp/requirements_dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements_dev.txt

ENV WANDB_DIR=/workspaces/MLOps_G116/outputs/wandb
