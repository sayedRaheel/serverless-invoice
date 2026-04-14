# InvoAIce Serverless

Vision-only invoice OCR via Qwen2.5-VL on RunPod Serverless.

## Pipeline

```
PDF → PyMuPDF classifier (is_digital?) → Qwen2.5-VL via Ollama → confidence gate → JSON
```

## Deploy

RunPod builds the Dockerfile from this repo automatically via its native GitHub integration. No registry setup, no workflows, no image management.

### 1. Create the Serverless endpoint

RunPod Console → **Serverless** → **New Endpoint** → **GitHub** tab:

- **Connect GitHub** → authorize RunPod → select `sayedRaheel/serverless-invoice`
- **Branch**: `main`
- **Dockerfile path**: `Dockerfile` (repo root — RunPod auto-detects)
- **GPU**: RTX 4090 (24 GB) or A5000 (24 GB) — needs ≥16 GB VRAM for Qwen2.5-VL 7B
- **Min workers**: `0` (true pay-per-use)
- **Max workers**: `3`
- **Idle timeout**: `5` seconds
- **Execution timeout**: `600` seconds
- **Container disk**: `30` GB

Click **Deploy**. RunPod clones the repo, builds the image, and caches it in their registry. First build takes ~15–20 min (torch + Ollama + qwen model pre-pull).

Subsequent pushes to `main` trigger a rebuild automatically.

## Request format

```json
{
  "input": {
    "file": "<base64-encoded PDF or image>",
    "filename": "invoice.pdf",
    "review_threshold": 0.85,
    "request_id": "optional-client-id"
  }
}
```

## Response format

```json
{
  "request_id": "optional-client-id",
  "file_hash": "sha256 hex",
  "filename": "invoice.pdf",
  "is_digital": true,
  "text_chars": 842,
  "pages": 1,
  "classifier_ms": 12,
  "total_ms": 15240,
  "needs_review": false,
  "review_reasons": [],
  "review_threshold": 0.85,
  "field_completeness": 1.0,
  "ocr_result": {
    "invoice_number": "INV-001",
    "invoice_date": "2026-04-01T00:00:00",
    "vendor_name": "Acme Corp",
    "total_amount": 1234.56,
    "line_items": [...],
    "overall_confidence": 0.91
  }
}
```

## Review rules (any one triggers `needs_review: true`)

1. `overall_confidence < review_threshold` (default 0.85)
2. Any critical field missing: `invoice_number`, `invoice_date`, `total_amount`, `vendor_name`
3. `subtotal + tax ≠ total` (tolerance: 2% or $1)
4. `sum(line_items) ≠ subtotal` (tolerance: 5% or $1)
5. `field_completeness < 1.0`

## Test requests

```bash
export RUNPOD_ENDPOINT_ID=xxxxxxxx   # from RunPod endpoint page
export RUNPOD_API_KEY=xxxxxxxx       # from RunPod → Settings → API Keys
python test_request.py sample_invoice.pdf
```

Or via curl:

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"file\":\"$(base64 -i sample.pdf)\",\"filename\":\"sample.pdf\"}}"
```

## Cost notes

- RunPod Serverless bills by GPU-second: ~$0.00024/s on RTX 4090
- Typical invoice: ~15s GPU time = ~$0.0036/invoice
- Cold start (worker spinup + Ollama warmup): ~30–60s first request, then warm for idle timeout
- Min workers `0` means you pay $0 when idle
- Image is ~10 GB (Qwen model baked in) — RunPod caches it per-region after first pull

## Notes

- Qwen2.5-VL handles both digital and scanned PDFs in vision mode
- The `is_digital` flag is captured but not used for routing — future optimization: text-mode path for digital PDFs to save GPU time
- Multi-page PDFs currently process page 1 only (matching the non-serverless pipeline behavior)
