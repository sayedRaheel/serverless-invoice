# InvoAIce Serverless — Qwen2.5-VL vision pipeline on RunPod Serverless
# Slim image: no Docling, PaddleOCR, LayoutLMv3, FastAPI, or frontend.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MODELS=/root/.ollama/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        poppler-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Ollama (Qwen2.5-VL backend) — install binary directly (install.sh fails in Docker build)
ARG OLLAMA_VERSION=v0.5.7
RUN curl -fsSL -o /tmp/ollama.tgz \
        https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz \
 && tar -C /usr -xzf /tmp/ollama.tgz \
 && rm /tmp/ollama.tgz \
 && ollama --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Pre-pull qwen2.5vl:7b into the image so cold-start workers don't re-download ~6 GB
RUN set -eux; \
    ollama serve > /tmp/ollama.log 2>&1 & \
    OLLAMA_PID=$!; \
    for i in $(seq 1 30); do \
        if curl -sf http://127.0.0.1:11434/api/version > /dev/null; then break; fi; \
        sleep 1; \
    done; \
    curl -sf http://127.0.0.1:11434/api/version || (echo "ollama serve failed to start"; cat /tmp/ollama.log; exit 1); \
    ollama pull qwen2.5vl:7b; \
    ls -la /root/.ollama/models/blobs/ | head; \
    kill $OLLAMA_PID; \
    wait $OLLAMA_PID 2>/dev/null || true

COPY app ./app
COPY handler.py ./handler.py

# Start script: Ollama in background, warm the model, then handler
RUN cat > /start.sh <<'SH'
#!/usr/bin/env bash
set -e
ollama serve &
sleep 3
(ollama run qwen2.5vl:7b "hi" > /dev/null 2>&1 &)
exec python -u handler.py
SH
RUN chmod +x /start.sh

CMD ["/start.sh"]
