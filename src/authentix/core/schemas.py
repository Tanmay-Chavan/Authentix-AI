"""Core data contracts — strict schemas shared across all modules.

Every pipeline, the fusion engine, and the API must speak through these
contracts.  Changing a field here is a **breaking change** — grep the
entire codebase before touching anything.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════
# ENUMS — Single source of truth for all categorical values
# ═══════════════════════════════════════════════════════════════

class Modality(str, enum.Enum):
    """Supported input modalities."""
    IMAGE = "image"
    VIDEO = "video"
    URL = "url"
    # DOCUMENT = "document"  # Deferred — Phase 2


class RiskVerdict(str, enum.Enum):
    """Final risk classification."""
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    MANIPULATED = "manipulated"
    INCONCLUSIVE = "inconclusive"


class ConfidenceBand(str, enum.Enum):
    """Calibrated confidence band."""
    HIGH = "high"        # ≥ 0.85
    MEDIUM = "medium"    # 0.60 – 0.84
    LOW = "low"          # < 0.60


class EvidenceType(str, enum.Enum):
    """Types of evidence a pipeline can produce."""
    # Image pipeline
    AIGEN_SCORE = "aigen_score"              # AI-generated probability
    CLIP_AIGEN_SCORE = "clip_aigen_score"    # CLIP-based AI-gen score
    GRADCAM_HEATMAP = "gradcam_heatmap"      # Grad-CAM heatmap path
    EXIF_ANOMALY = "exif_anomaly"            # EXIF metadata anomaly flag

    # Video pipeline
    DEEPFAKE_SCORE = "deepfake_score"        # Deepfake probability
    FRAME_SCORES = "frame_scores"            # Per-frame deepfake scores

    # URL pipeline
    PHISHING_SCORE = "phishing_score"        # Phishing probability
    DOMAIN_AGE_DAYS = "domain_age_days"      # Domain registration age
    TLS_VALID = "tls_valid"                  # TLS certificate validity
    LEXICAL_FEATURES = "lexical_features"    # URL lexical feature vector


# ═══════════════════════════════════════════════════════════════
# CORE SCHEMAS
# ═══════════════════════════════════════════════════════════════

class EvidenceItem(BaseModel):
    """A single piece of evidence produced by any pipeline."""
    evidence_type: EvidenceType
    value: float | str | list[float] | dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    source_model: str = Field(
        description="Name + version of the model that produced this evidence, "
                    "e.g. 'efficientnet_b4_genimage_v1'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class AnalysisRequest(BaseModel):
    """Inbound request from the API layer."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    modality: Modality
    # For image/video: base64 or file path (sanitised by security layer)
    # For URL: the raw URL string
    payload_ref: str = Field(
        description="Reference to the uploaded content — file path on disk "
                    "after sanitisation, or the raw URL string."
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    user_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """Outbound result returned to the API layer / stored in DB."""
    request_id: str
    modality: Modality
    verdict: RiskVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation_text: str = ""
    explanation_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated artifacts like heatmaps, "
                    "e.g. {'gradcam': '/tmp/heatmap_abc.png'}"
    )
    processing_time_ms: float = 0.0
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def from_evidence(
        cls,
        request_id: str,
        modality: Modality,
        verdict: RiskVerdict,
        confidence: float,
        evidence: list[EvidenceItem],
        explanation_text: str = "",
        explanation_artifacts: dict[str, str] | None = None,
        processing_time_ms: float = 0.0,
    ) -> AnalysisResult:
        """Factory method with automatic confidence band assignment."""
        if confidence >= 0.85:
            band = ConfidenceBand.HIGH
        elif confidence >= 0.60:
            band = ConfidenceBand.MEDIUM
        else:
            band = ConfidenceBand.LOW

        return cls(
            request_id=request_id,
            modality=modality,
            verdict=verdict,
            confidence=confidence,
            confidence_band=band,
            evidence=evidence,
            explanation_text=explanation_text,
            explanation_artifacts=explanation_artifacts or {},
            processing_time_ms=processing_time_ms,
        )
