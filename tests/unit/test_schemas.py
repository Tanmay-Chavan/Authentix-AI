"""Gate 0 — Tests for core data contracts (schemas)."""

import pytest

from authentix.core.schemas import (
    AnalysisRequest,
    AnalysisResult,
    ConfidenceBand,
    EvidenceItem,
    EvidenceType,
    Modality,
    RiskVerdict,
)


class TestEnums:
    """Verify all enums are well-formed."""

    def test_modality_values(self):
        assert Modality.IMAGE.value == "image"
        assert Modality.VIDEO.value == "video"
        assert Modality.URL.value == "url"

    def test_risk_verdict_values(self):
        assert RiskVerdict.AUTHENTIC.value == "authentic"
        assert RiskVerdict.SUSPICIOUS.value == "suspicious"
        assert RiskVerdict.MANIPULATED.value == "manipulated"
        assert RiskVerdict.INCONCLUSIVE.value == "inconclusive"

    def test_confidence_band_values(self):
        assert ConfidenceBand.HIGH.value == "high"
        assert ConfidenceBand.MEDIUM.value == "medium"
        assert ConfidenceBand.LOW.value == "low"


class TestEvidenceItem:
    """Verify EvidenceItem validation."""

    def test_valid_evidence(self):
        e = EvidenceItem(
            evidence_type=EvidenceType.AIGEN_SCORE,
            value=0.92,
            confidence=0.88,
            source_model="efficientnet_b4_genimage_v1",
        )
        assert e.confidence == 0.88
        assert e.value == 0.92

    def test_confidence_clamped_above_1(self):
        e = EvidenceItem(
            evidence_type=EvidenceType.AIGEN_SCORE,
            value=0.5,
            confidence=1.5,  # Should clamp to 1.0
            source_model="test",
        )
        assert e.confidence == 1.0

    def test_confidence_clamped_below_0(self):
        e = EvidenceItem(
            evidence_type=EvidenceType.AIGEN_SCORE,
            value=0.5,
            confidence=-0.3,  # Should clamp to 0.0
            source_model="test",
        )
        assert e.confidence == 0.0

    def test_evidence_with_dict_value(self):
        e = EvidenceItem(
            evidence_type=EvidenceType.LEXICAL_FEATURES,
            value={"url_length": 45, "num_dots": 3},
            confidence=0.75,
            source_model="url_lexical_v1",
        )
        assert isinstance(e.value, dict)

    def test_evidence_with_list_value(self):
        e = EvidenceItem(
            evidence_type=EvidenceType.FRAME_SCORES,
            value=[0.1, 0.2, 0.8, 0.9],
            confidence=0.65,
            source_model="deepfake_video_v1",
        )
        assert isinstance(e.value, list)


class TestAnalysisRequest:
    """Verify AnalysisRequest creation."""

    def test_auto_request_id(self):
        r = AnalysisRequest(
            modality=Modality.IMAGE,
            payload_ref="/uploads/test.jpg",
        )
        assert len(r.request_id) == 32  # hex UUID

    def test_auto_timestamp(self):
        r = AnalysisRequest(
            modality=Modality.URL,
            payload_ref="https://example.com",
        )
        assert r.submitted_at is not None


class TestAnalysisResult:
    """Verify AnalysisResult creation and confidence banding."""

    def test_from_evidence_high_confidence(self):
        result = AnalysisResult.from_evidence(
            request_id="abc123",
            modality=Modality.IMAGE,
            verdict=RiskVerdict.MANIPULATED,
            confidence=0.95,
            evidence=[],
        )
        assert result.confidence_band == ConfidenceBand.HIGH

    def test_from_evidence_medium_confidence(self):
        result = AnalysisResult.from_evidence(
            request_id="abc123",
            modality=Modality.IMAGE,
            verdict=RiskVerdict.SUSPICIOUS,
            confidence=0.72,
            evidence=[],
        )
        assert result.confidence_band == ConfidenceBand.MEDIUM

    def test_from_evidence_low_confidence(self):
        result = AnalysisResult.from_evidence(
            request_id="abc123",
            modality=Modality.IMAGE,
            verdict=RiskVerdict.INCONCLUSIVE,
            confidence=0.45,
            evidence=[],
        )
        assert result.confidence_band == ConfidenceBand.LOW

    def test_serialisation_roundtrip(self):
        """Verify JSON serialisation / deserialisation works."""
        result = AnalysisResult.from_evidence(
            request_id="test_roundtrip",
            modality=Modality.URL,
            verdict=RiskVerdict.AUTHENTIC,
            confidence=0.91,
            evidence=[
                EvidenceItem(
                    evidence_type=EvidenceType.PHISHING_SCORE,
                    value=0.08,
                    confidence=0.91,
                    source_model="xgboost_url_v1",
                )
            ],
            explanation_text="URL appears legitimate.",
        )
        # Serialize to dict, then back to model
        data = result.model_dump()
        restored = AnalysisResult(**data)
        assert restored.request_id == "test_roundtrip"
        assert restored.verdict == RiskVerdict.AUTHENTIC
        assert len(restored.evidence) == 1
