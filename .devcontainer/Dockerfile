FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY requirements_dev.txt /tmp/requirements_dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements_dev.txt

WORKDIR /workspaces/MLOps_G116

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data/*.dvc data/
RUN dvc config core.no_scm true
RUN dvc pull
