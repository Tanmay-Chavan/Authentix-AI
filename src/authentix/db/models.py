"""SQLAlchemy ORM models for persistent storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID

from authentix.db.session import Base


class AnalysisRecord(Base):
    """Stores every analysis request and its result."""

    __tablename__ = "analysis_records"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    modality = Column(String(16), nullable=False, index=True)
    payload_ref = Column(Text, nullable=False)
    verdict = Column(String(24), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_band = Column(String(16), nullable=True)
    evidence_json = Column(JSON, nullable=True)
    explanation_text = Column(Text, nullable=True)
    explanation_artifacts = Column(JSON, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    user_id = Column(String(64), nullable=True, index=True)
    submitted_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisRecord request_id={self.request_id} "
            f"modality={self.modality} verdict={self.verdict}>"
        )
