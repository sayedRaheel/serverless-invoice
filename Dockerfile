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

# Ollama (Qwen2.5-VL backend)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Pre-pull qwen2.5vl:7b into the image so cold-start workers don't re-download ~6 GB
RUN ollama serve & \
    sleep 5 && \
    ollama pull qwen2.5vl:7b && \
    pkill -f "ollama serve" || true

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
