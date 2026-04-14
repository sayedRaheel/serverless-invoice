# InvoAIce Serverless

Vision-only invoice OCR via Qwen2.5-VL on RunPod Serverless.

## Pipeline

```
PDF → PyMuPDF classifier (is_digital?) → Qwen2.5-VL via Ollama → confidence gate → JSON
```

## Deploy

### 1. Push to GitHub

```bash
gh repo create serverless-invoice --private --source=. --remote=origin --push
```

GitHub Actions will auto-build and push `ghcr.io/sayedraheel/serverless-invoice:latest` (~20 min).

### 2. Make GHCR image public

`github.com/users/sayedraheel/packages/container/serverless-invoice/settings` → Danger Zone → Change visibility → **Public**.

### 3. Create a RunPod Serverless endpoint

RunPod Console → **Serverless** → **New Endpoint**:

- **Container image**: `ghcr.io/sayedraheel/serverless-invoice:latest`
- **GPU**: RTX 4090 (24 GB) or A5000 (24 GB) — needs ≥ 16 GB VRAM for Qwen2.5-VL 7B
- **Min workers**: 0 (true pay-per-use)
- **Max workers**: 3
- **Idle timeout**: 5 seconds
- **Execution timeout**: 600 seconds
- **Container disk**: 30 GB

No env vars required. Model is baked into the image.

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
    "overall_confidence": 0.91,
    ...
  }
}
```

## Review rules (any one triggers `needs_review: true`)

1. `overall_confidence < review_threshold` (default 0.85)
2. Any critical field missing: `invoice_number`, `invoice_date`, `total_amount`, `vendor_name`
3. `subtotal + tax ≠ total` (tolerance: 2% or $1)
4. `sum(line_items) ≠ subtotal` (tolerance: 5% or $1)
5. `field_completeness < 1.0`

## Test locally

```bash
export RUNPOD_ENDPOINT_ID=xxxxxxxx
export RUNPOD_API_KEY=xxxxxxxx
python test_request.py sample_invoice.pdf
```

## Cost notes

- RunPod Serverless bills by GPU-second: ~$0.00024/s on RTX 4090
- Typical invoice: ~15s GPU time = ~$0.0036/invoice
- Cold start (worker spinup + Ollama warmup): ~30–60s first request, then warm for idle timeout
- Image is ~10 GB (baked Qwen model) — RunPod caches it per-region after first pull

## Notes

- Qwen2.5-VL handles both digital and scanned PDFs in vision mode
- The `is_digital` flag is captured but not used for routing yet — future optimization: text-mode path for digital PDFs to save GPU
- Multi-page PDFs currently process page 1 only (matching the non-serverless pipeline behavior)
