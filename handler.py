"""RunPod Serverless handler — Qwen2.5-VL invoice extraction.

Flow:
  1. Receive base64 PDF / image
  2. PyMuPDF classifier: digital PDF (text layer) or scanned?
  3. Run Qwen2.5-VL vision via Ollama
  4. Confidence gate → needs_review flag

Input:
  {"input": {
      "file": "<base64>",
      "filename": "invoice.pdf",
      "review_threshold": 0.85
  }}

Output:
  {
    "is_digital": bool,
    "text_chars": int,
    "pages": int,
    "classifier_ms": int,
    "total_ms": int,
    "needs_review": bool,
    "review_reasons": [str, ...],
    "review_threshold": float,
    "field_completeness": float,   # 0..1 over critical fields
    "ocr_result": {...}
  }
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time

import fitz  # PyMuPDF
import runpod

from app.ocr.vision_pipeline import vision_pipeline


MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

DIGITAL_TEXT_THRESHOLD = 200
DEFAULT_REVIEW_THRESHOLD = 0.85

CRITICAL_FIELDS = (
    "invoice_number",
    "invoice_date",
    "total_amount",
    "vendor_name",
)


def _mime_from_filename(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return MIME_BY_EXT.get(ext, "application/pdf")


def classify_pdf(file_bytes: bytes) -> tuple[bool, int, int]:
    """Return (is_digital, total_text_chars, page_count)."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return False, 0, 0
    try:
        total_chars = sum(len(page.get_text("text") or "") for page in doc)
        return total_chars >= DIGITAL_TEXT_THRESHOLD, total_chars, doc.page_count
    finally:
        doc.close()


def evaluate_review(ocr_dict: dict, threshold: float) -> tuple[bool, list[str], float]:
    """Decide if the extraction needs human review. Any rule fires → review."""
    reasons: list[str] = []

    overall_conf = float(ocr_dict.get("overall_confidence") or 0.0)
    if overall_conf < threshold:
        reasons.append(f"overall_confidence<{threshold} ({overall_conf:.2f})")

    missing = [f for f in CRITICAL_FIELDS if not ocr_dict.get(f)]
    if missing:
        reasons.append(f"missing_critical_fields:{','.join(missing)}")

    total = float(ocr_dict.get("total_amount") or 0)
    subtotal = float(ocr_dict.get("subtotal") or 0)
    tax = float(ocr_dict.get("tax_amount") or 0)
    if total and subtotal and abs(total - (subtotal + tax)) > max(0.02 * total, 1.0):
        reasons.append("math_mismatch:subtotal+tax!=total")

    line_items = ocr_dict.get("line_items") or []
    if line_items and total:
        li_sum = sum(float(li.get("amount") or 0) for li in line_items)
        if li_sum and abs(li_sum - (subtotal or total)) > max(0.05 * total, 1.0):
            reasons.append("math_mismatch:line_items_sum")

    completeness = sum(1 for f in CRITICAL_FIELDS if ocr_dict.get(f)) / len(CRITICAL_FIELDS)
    if completeness < 1.0:
        reasons.append(f"field_completeness<1.0 ({completeness:.2f})")

    return (len(reasons) > 0, reasons, completeness)


async def _run(job_input: dict) -> dict:
    file_b64 = job_input.get("file")
    if not file_b64:
        return {"error": "missing 'file' (base64 encoded)"}

    filename = job_input.get("filename", "invoice.pdf")
    mime = job_input.get("mime") or _mime_from_filename(filename)
    threshold = float(job_input.get("review_threshold") or DEFAULT_REVIEW_THRESHOLD)
    request_id = job_input.get("request_id")

    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception as exc:
        return {"error": f"invalid base64: {exc}"}

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    t_start = time.time()

    t0 = time.time()
    if mime == "application/pdf":
        is_digital, text_chars, pages = classify_pdf(file_bytes)
    else:
        is_digital, text_chars, pages = False, 0, 1
    classifier_ms = int((time.time() - t0) * 1000)

    debug_result = await vision_pipeline.debug_process(
        file_bytes=file_bytes,
        mime_type=mime,
        invoice_id=filename,
    )
    ocr_dict = debug_result.ocr_result.model_dump(mode="json")

    needs_review, reasons, completeness = evaluate_review(ocr_dict, threshold)
    total_ms = int((time.time() - t_start) * 1000)

    return {
        "request_id": request_id,
        "file_hash": file_hash,
        "filename": filename,
        "is_digital": is_digital,
        "text_chars": text_chars,
        "pages": pages,
        "classifier_ms": classifier_ms,
        "total_ms": total_ms,
        "needs_review": needs_review,
        "review_reasons": reasons,
        "review_threshold": threshold,
        "field_completeness": round(completeness, 2),
        "ocr_result": ocr_dict,
    }


def handler(job):
    return asyncio.run(_run(job["input"]))


runpod.serverless.start({"handler": handler})
