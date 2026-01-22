FROM python:3.11-slim

# 1. Install system tools
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy Requirements & Install
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# 3. Install PyTorch CPU-only
RUN pip install --no-cache-dir torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
# 4. Copy the Backend Code
# Assuming your terminal is at project root and backend.py is in src/mlops_g116/
COPY src src/
COPY models/model.pth models/model.pth

# 6. Environment & Run
# Cloud run
#ENV PORT=8000
CMD uvicorn mlops_g116.backend:app --host 0.0.0.0 --port $PORT --app-dir src

