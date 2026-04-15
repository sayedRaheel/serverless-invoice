# InvoAIce Serverless — Qwen2.5-VL vision pipeline on RunPod Serverless.
# Base image: ollama/ollama (has ollama binary + GPU libs + LD paths pre-wired).
# Known-working base per the RunPod-Ollama reference repo.
FROM ollama/ollama:0.5.7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        poppler-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

COPY app ./app
COPY handler.py ./handler.py

# Start script: ollama serve in background, wait for "Listening", pull model, run handler.
RUN cat > /start.sh <<'SH'
#!/usr/bin/env bash
set -e
echo "[start.sh] booting"
ollama serve 2>&1 | tee /tmp/ollama.log &
for i in $(seq 1 120); do
    if grep -q "Listening" /tmp/ollama.log 2>/dev/null; then
        echo "[start.sh] ollama listening after ${i}s"
        break
    fi
    sleep 1
done
grep -q "Listening" /tmp/ollama.log || { echo "[start.sh] ERROR ollama never listened"; tail -50 /tmp/ollama.log; exit 1; }
if ollama list 2>&1 | grep -q 'qwen2.5vl:7b'; then
    echo "[start.sh] qwen2.5vl:7b already present"
else
    echo "[start.sh] pulling qwen2.5vl:7b (~6 GB, one-time per worker)"
    ollama pull qwen2.5vl:7b
    echo "[start.sh] pull complete"
fi
(ollama run qwen2.5vl:7b "hi" > /dev/null 2>&1 &)
echo "[start.sh] starting handler"
exec python -u handler.py
SH
RUN chmod +x /start.sh

# ollama/ollama base sets ENTRYPOINT=/bin/ollama — override it
ENTRYPOINT []
CMD ["/start.sh"]
