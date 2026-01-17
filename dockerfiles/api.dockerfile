FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Optional: Document that we listen on a variable port
EXPOSE 8080

# Use exec to enable variable expansion AND graceful shutdowns
# We use ${PORT:-8080} which means: "Use $PORT if it exists, otherwise default to 8080"
CMD exec uvicorn src.mlops_g116.api:app --host 0.0.0.0 --port ${PORT:-8080}
