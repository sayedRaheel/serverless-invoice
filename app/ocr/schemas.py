"""OCR Pipeline data models — all Pydantic v2."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExtractionMethod(str, Enum):
    DOCLING_PADDLE_LAYOUTLM = "docling+paddleocr+layoutlm"
    DOCLING_LAYOUTLM = "docling+layoutlm"  # digital PDF (no OCR needed)
    GPT4O_FALLBACK = "gpt4o_fallback"
    MOCK = "mock"


class ConfidenceSource(str, Enum):
    LAYOUTLM = "layoutlm"
    GPT4O = "gpt4o_fallback"
    REGEX = "regex"
    DIGITAL = "digital_extract"


# ---------------------------------------------------------------------------
# Low-level: words, pages
# ---------------------------------------------------------------------------

class WordData(BaseModel):
    """Single word from OCR with position and confidence."""
    text: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = Field(ge=0.0, le=1.0)
    page_num: int = 0


class LayoutRegion(BaseModel):
    """A detected region on a page (header, table, text, footer, etc.)."""
    label: str  # "header" | "table" | "text" | "footer" | "key_value"
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0.0, le=1.0)


class PageData(BaseModel):
    """Preprocessed page — image bytes + layout regions + optional OCR words."""
    page_num: int
    image_bytes: Optional[bytes] = None  # rasterized page image (PNG)
    width: int = 0
    height: int = 0
    regions: list[LayoutRegion] = Field(default_factory=list)
    words: list[WordData] = Field(default_factory=list)
    raw_text: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Stage outputs
# ---------------------------------------------------------------------------

class PreprocessResult(BaseModel):
    """Output of Stage 1 — DocumentPreprocessor."""
    pages: list[PageData]
    is_digital_pdf: bool = False
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    page_count: int = 0
    docling_document: Optional[Any] = None  # raw DoclingDocument

    model_config = {"arbitrary_types_allowed": True}


class PageOCR(BaseModel):
    """Output of Stage 2 — TextExtractor (per page)."""
    page_num: int
    words: list[WordData] = Field(default_factory=list)
    raw_text: str = ""


class TokenPrediction(BaseModel):
    """Single token prediction from LayoutLMv3 — for debugging."""
    text: str
    label: str  # BIO tag e.g. "B-INVOICE_NUMBER"
    confidence: float
    bbox: tuple[int, int, int, int]
    page_num: int = 0


class ExtractionResult(BaseModel):
    """Output of Stage 4 — FieldExtractor."""
    fields: dict[str, str] = Field(default_factory=dict)
    field_confidences: dict[str, float] = Field(default_factory=dict)
    token_details: list[TokenPrediction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Line items
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """Single invoice line item."""
    description: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    amount: float = 0.0
    item_code: Optional[str] = None
    tax_rate: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class FieldConfidence(BaseModel):
    """Per-field confidence breakdown."""
    model_score: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_score: float = Field(default=0.5, ge=0.0, le=1.0)  # 0.5 = not checked
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source: ConfidenceSource = ConfidenceSource.LAYOUTLM


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

class OCRResult(BaseModel):
    """Complete OCR extraction result — consumed by downstream agents."""

    # Header fields
    invoice_number: str = ""
    invoice_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    vendor_name: str = ""
    vendor_address: str = ""
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    subtotal: float = 0.0
    tax_amount: float = 0.0
    total_amount: float = 0.0
    currency: str = "USD"
    po_number: Optional[str] = None
    bank_account: Optional[str] = None
    payment_terms: Optional[str] = None

    # Line items
    line_items: list[LineItem] = Field(default_factory=list)

    # Confidence
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    field_confidence: dict[str, FieldConfidence] = Field(default_factory=dict)

    # Metadata
    extraction_method: ExtractionMethod = ExtractionMethod.MOCK
    page_count: int = 0
    processing_time_ms: float = 0.0
    is_digital_pdf: bool = False
    raw_text: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Debug / UI models
# ---------------------------------------------------------------------------

class StageInfo(BaseModel):
    """Info about a single pipeline stage execution."""
    name: str
    duration_ms: float = 0.0
    status: str = "success"  # "success" | "skipped" | "failed"
    metadata: dict = Field(default_factory=dict)


class DebugPipelineResult(BaseModel):
    """Full pipeline result with all intermediates — for the debug UI."""
    ocr_result: OCRResult
    stages: list[StageInfo] = Field(default_factory=list)
    page_images: list[str] = Field(default_factory=list)  # base64 PNG per page
    token_details: list[TokenPrediction] = Field(default_factory=list)
    layout_regions: list[dict] = Field(default_factory=list)  # per-page regions
    config: dict = Field(default_factory=dict)  # thresholds for frontend
