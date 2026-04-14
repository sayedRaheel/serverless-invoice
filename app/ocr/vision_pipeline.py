"""Vision LLM Pipeline — single-model invoice extraction via Qwen2.5-VL.

Completely separate from the multi-stage pipeline. Sends page images
to a local vision model (Ollama) and gets structured JSON back.

Usage:
    from app.ocr.vision_pipeline import vision_pipeline
    result = await vision_pipeline.debug_process(file_bytes, "application/pdf")
"""

from __future__ import annotations

import base64
import json
import time
from typing import Optional

import structlog
import httpx

from .schemas import (
    DebugPipelineResult,
    ExtractionMethod,
    FieldConfidence,
    ConfidenceSource,
    LineItem,
    OCRResult,
    StageInfo,
)

logger = structlog.get_logger(__name__)

# The extraction prompt — tells the vision model exactly what to extract
EXTRACTION_PROMPT = """Extract invoice fields as JSON. Return ONLY the JSON, no markdown.
{
  "invoice_number": "",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "vendor_name": "",
  "vendor_address": "",
  "buyer_name": "",
  "buyer_address": "",
  "subtotal": 0,
  "tax_amount": 0,
  "total_amount": 0,
  "currency": "USD",
  "po_number": "",
  "payment_terms": "",
  "line_items": [{"description":"","quantity":0,"unit_price":0,"amount":0}]
}
Use null for missing fields. Amounts as plain numbers (no symbols/commas)."""


VALIDATION_PROMPT = """Verify this extracted invoice data against the image:
{extracted_json}

Return ONLY JSON:
{{
  "validation_scores": {{"invoice_number":0.0,"invoice_date":0.0,"due_date":0.0,"vendor_name":0.0,"buyer_name":0.0,"total_amount":0.0,"subtotal":0.0,"tax_amount":0.0,"line_items":0.0}},
  "math_check": {{"line_items_sum_matches_subtotal":false,"subtotal_plus_tax_matches_total":false}}
}}
1.0=exact match, 0.5=partial, 0.0=wrong."""


class VisionPipeline:
    """2-stage pipeline: Vision LLM extraction → Vision LLM validation."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
    ):
        self.ollama_base_url = ollama_base_url
        self.model = model

    async def _call_ollama(
        self,
        prompt: str,
        image_b64: str,
        temperature: float = 0.1,
    ) -> str:
        """Send image + prompt to Ollama vision model."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 4096,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]

    def _rasterize_pdf(self, file_bytes: bytes, dpi: int = 150) -> list[tuple[str, int, int]]:
        """PDF → list of (base64_png, width, height) per page.

        150 DPI is 4× faster than 300 DPI for vision LLM inference
        because attention is O(n²) in visual token count.
        """
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            pages.append((b64, pix.width, pix.height))
        doc.close()
        return pages

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            lines = text.split("\n")
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end])
        return json.loads(text)

    def _build_ocr_result(self, data: dict, validation: Optional[dict] = None) -> OCRResult:
        """Convert parsed JSON into OCRResult with field confidence."""
        from datetime import datetime

        def parse_date(val):
            if not val:
                return None
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None

        line_items = []
        for item in data.get("line_items") or []:
            line_items.append(LineItem(
                description=item.get("description", ""),
                quantity=float(item.get("quantity", 0) or 0),
                unit_price=float(item.get("unit_price", 0) or 0),
                amount=float(item.get("amount", 0) or 0),
                confidence=0.9,
            ))

        # Build field confidence — model_score defaults to 0.9 (Qwen2.5-VL is reliable
        # on field extraction), validation_score comes from the 2nd-pass vision check
        val_scores = (validation or {}).get("validation_scores", {})

        field_confidence = {}
        for field_name in [
            "invoice_number", "invoice_date", "due_date", "vendor_name",
            "vendor_address", "buyer_name", "buyer_address", "subtotal",
            "tax_amount", "total_amount", "currency", "po_number", "payment_terms",
        ]:
            value = data.get(field_name)
            if value is None or value == "":
                continue
            m_score = 0.9
            v_score = float(val_scores.get(field_name, 0.8) or 0.8)
            final = 0.7 * m_score + 0.3 * v_score
            field_confidence[field_name] = FieldConfidence(
                model_score=m_score,
                validation_score=v_score,
                final_score=round(final, 3),
                source=ConfidenceSource.LAYOUTLM,
            )

        # Apply corrections from validation
        corrections = (validation or {}).get("field_corrections", {})
        for field_name, corrected_val in corrections.items():
            if corrected_val is not None and field_name in data:
                data[field_name] = corrected_val

        # Overall confidence
        if field_confidence:
            overall = sum(fc.final_score for fc in field_confidence.values()) / len(field_confidence)
        else:
            overall = 0.0

        return OCRResult(
            invoice_number=data.get("invoice_number") or "",
            invoice_date=parse_date(data.get("invoice_date")),
            due_date=parse_date(data.get("due_date")),
            vendor_name=data.get("vendor_name") or "",
            vendor_address=data.get("vendor_address") or "",
            buyer_name=data.get("buyer_name"),
            buyer_address=data.get("buyer_address"),
            subtotal=float(data.get("subtotal") or 0),
            tax_amount=float(data.get("tax_amount") or 0),
            total_amount=float(data.get("total_amount") or 0),
            currency=data.get("currency") or "USD",
            po_number=data.get("po_number"),
            payment_terms=data.get("payment_terms"),
            line_items=line_items,
            overall_confidence=round(overall, 3),
            field_confidence=field_confidence,
            extraction_method=ExtractionMethod.DOCLING_LAYOUTLM,  # we'll add a new enum
        )

    async def debug_process(
        self,
        file_bytes: bytes,
        mime_type: str,
        invoice_id: Optional[str] = None,
    ) -> DebugPipelineResult:
        """Run the 3-stage vision pipeline with full debug output."""
        t0 = time.perf_counter()
        trace = invoice_id or "unknown"
        stages: list[StageInfo] = []

        # ------------------------------------------------------------------
        # Stage 1: Rasterize PDF
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        page_data = self._rasterize_pdf(file_bytes)
        page_images = [p[0] for p in page_data]
        stages.append(StageInfo(
            name="PDF Rasterization",
            duration_ms=_ms(t1),
            status="success",
            metadata={"pages": len(page_data), "dpi": 300},
        ))
        logger.info("vision.rasterize.done", invoice_id=trace, pages=len(page_data))

        # ------------------------------------------------------------------
        # Stage 2: Vision LLM Extraction (Qwen2.5-VL)
        # ------------------------------------------------------------------
        t2 = time.perf_counter()
        # For multi-page, extract from first page (most invoice data is on page 1)
        # TODO: handle multi-page invoices by sending all pages
        raw_response = await self._call_ollama(EXTRACTION_PROMPT, page_images[0])
        try:
            extracted = self._parse_json_response(raw_response)
            extraction_status = "success"
        except (json.JSONDecodeError, Exception) as e:
            logger.error("vision.extraction.parse_failed", error=str(e), raw=raw_response[:500])
            extracted = {}
            extraction_status = "failed"

        stages.append(StageInfo(
            name=f"Vision Extraction ({self.model})",
            duration_ms=_ms(t2),
            status=extraction_status,
            metadata={
                "model": self.model,
                "fields_found": len([v for v in extracted.items() if v[1] is not None and v[0] != "confidence" and v[0] != "line_items"]),
                "line_items_found": len(extracted.get("line_items", [])),
            },
        ))
        logger.info("vision.extraction.done", invoice_id=trace, status=extraction_status, ms=_ms(t2))

        # ------------------------------------------------------------------
        # Stage 3: Vision LLM Validation (second pass)
        # ------------------------------------------------------------------
        validation = None
        if extraction_status == "success" and extracted:
            t3 = time.perf_counter()
            # Send the same image + extracted data for cross-validation
            validation_prompt = VALIDATION_PROMPT.format(
                extracted_json=json.dumps(extracted, indent=2, default=str)
            )
            try:
                val_response = await self._call_ollama(validation_prompt, page_images[0], temperature=0.0)
                validation = self._parse_json_response(val_response)
                val_status = "success"
            except Exception as e:
                logger.error("vision.validation.failed", error=str(e))
                validation = None
                val_status = "failed"

            math_check = (validation or {}).get("math_check", {})
            stages.append(StageInfo(
                name=f"Vision Validation ({self.model})",
                duration_ms=_ms(t3),
                status=val_status,
                metadata={
                    "math_check": math_check,
                    "corrections": len((validation or {}).get("field_corrections", {})),
                },
            ))
            logger.info("vision.validation.done", invoice_id=trace, status=val_status, ms=_ms(t3))
        else:
            stages.append(StageInfo(
                name=f"Vision Validation ({self.model})",
                duration_ms=0.0,
                status="skipped",
                metadata={"reason": "extraction failed"},
            ))

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        ocr_result = self._build_ocr_result(extracted, validation)
        total_ms = _ms(t0)
        ocr_result.processing_time_ms = total_ms
        ocr_result.page_count = len(page_data)
        ocr_result.is_digital_pdf = False  # we don't check in this pipeline

        logger.info(
            "vision.pipeline.complete",
            invoice_id=trace,
            confidence=ocr_result.overall_confidence,
            fields=len(ocr_result.field_confidence),
            line_items=len(ocr_result.line_items),
            total_ms=round(total_ms, 1),
        )

        return DebugPipelineResult(
            ocr_result=ocr_result,
            stages=stages,
            page_images=page_images,
            token_details=[],  # no token-level data in vision pipeline
            layout_regions=[],  # no layout regions
            config={
                "model": self.model,
                "pipeline": "vision_llm",
            },
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# Singleton
vision_pipeline = VisionPipeline()
