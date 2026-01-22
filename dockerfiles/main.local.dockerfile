# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

ARG GOOGLE_CLOUD_PROJECT
ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
ENV WANDB_PROJECT=mlops_g116
ENV WANDB_ENTITY=sergi-luponsantacana-danmarks-tekniske-universitet-dtu
ENV WANDB_REGISTRY_ENTITY=sergi-luponsantacana-danmarks-tekniske-universitet-dtu-org
ENV WANDB_MODE=online
ENV WANDB_COLLECTION_MAIN=mlops_g116-main-models
ENV WANDB_COLLECTION_MAIN_EVAL=mlops_g116-main-evals

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src src/
COPY configs configs/
COPY data/processed data/processed
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY dockerfiles/main_local_entrypoint.sh /usr/local/bin/main_entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --verbose
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . --no-deps --verbose

RUN chmod +x /usr/local/bin/main_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/main_entrypoint.sh"]
