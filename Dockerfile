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
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        poppler-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

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

# Note: qwen2.5vl:7b is downloaded on first cold start per worker (~2-3 min, ~6 GB).
# Pre-pulling at build time isn't viable on GitHub Actions runners (no GPU, ollama
# serve behavior inconsistent). OLLAMA_KEEP_ALIVE=-1 keeps the model resident once loaded.

COPY app ./app
COPY handler.py ./handler.py

# Start script: Ollama in background, wait for server, ensure model present, then handler
RUN cat > /start.sh <<'SH'
#!/usr/bin/env bash
set -e
ollama serve &
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:11434/api/version > /dev/null && break
    sleep 1
done
if ! ollama list 2>/dev/null | grep -q 'qwen2.5vl:7b'; then
    echo "[start.sh] pulling qwen2.5vl:7b (first-run download, ~6 GB)"
    ollama pull qwen2.5vl:7b
fi
# Warm the model into VRAM in background
(ollama run qwen2.5vl:7b "hi" > /dev/null 2>&1 &)
exec python -u handler.py
SH
RUN chmod +x /start.sh

CMD ["/start.sh"]
