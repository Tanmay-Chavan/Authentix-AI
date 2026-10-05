"""Abstract base classes for all analysis pipelines.

Every modality pipeline (image, video, URL) must subclass
``BasePipeline`` and implement ``analyse()``.  This guarantees a
uniform interface for the fusion engine and API layer.
"""

from __future__ import annotations

import abc
import time
from typing import Any

import structlog

from authentix.core.schemas import (
    AnalysisRequest,
    EvidenceItem,
    Modality,
)

logger = structlog.get_logger(__name__)


class BasePipeline(abc.ABC):
    """Contract every analysis pipeline must satisfy."""

    # Subclasses MUST set this
    modality: Modality

    @abc.abstractmethod
    async def analyse(self, request: AnalysisRequest) -> list[EvidenceItem]:
        """Run the full analysis pipeline and return evidence items.

        Parameters
        ----------
        request : AnalysisRequest
            Validated, sanitised analysis request.

        Returns
        -------
        list[EvidenceItem]
            One or more evidence items produced by this pipeline.
            An empty list signals the pipeline could not produce
            any usable evidence (e.g. corrupt file).
        """
        ...

    async def analyse_timed(
        self, request: AnalysisRequest
    ) -> tuple[list[EvidenceItem], float]:
        """Wrapper that times ``analyse()`` and returns (evidence, ms)."""
        t0 = time.perf_counter()
        evidence = await self.analyse(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "pipeline.analysis_complete",
            pipeline=self.__class__.__name__,
            modality=self.modality.value,
            evidence_count=len(evidence),
            elapsed_ms=round(elapsed_ms, 2),
            request_id=request.request_id,
        )
        return evidence, elapsed_ms

    def health_check(self) -> dict[str, Any]:
        """Return pipeline health status.  Override for custom checks."""
        return {
            "pipeline": self.__class__.__name__,
            "modality": self.modality.value,
            "status": "ok",
        }
