"""Authentix AI — FastAPI Application Entry Point."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from authentix.core.config import settings
from authentix.core.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Modality,
    RiskVerdict,
    ConfidenceBand,
)
from authentix.db.session import close_db, init_db
from authentix.security.guards import (
    UploadSanitiserError,
    SSRFError,
    validate_upload,
    validate_url,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, load models.  Shutdown: close DB."""
    logger.info("app.starting", version=settings.app_version)

    # Ensure directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # Init DB tables (dev mode — use Alembic in prod)
    try:
        await init_db()
        logger.info("db.initialised")
    except Exception as e:
        logger.error("db.init_failed", error=str(e))
        # Don't crash — API can still serve health checks
        # DB will be retried on first request

    yield  # ← App is running

    await close_db()
    logger.info("app.shutdown_complete")


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multi-modal digital content verification platform. "
        "Detects AI-generated images, deepfake videos, and phishing URLs."
    ),
    lifespan=lifespan,
)

# CORS — permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Health & Info endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "service": settings.app_name,
    }


@app.get("/info", tags=["system"])
async def system_info():
    """System information and supported modalities."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "supported_modalities": [m.value for m in Modality],
        "endpoints": {
            "analyse_image": "/api/v1/analyse/image",
            "analyse_url": "/api/v1/analyse/url",
            "analyse_video": "/api/v1/analyse/video",
            "health": "/health",
        },
    }


# ─────────────────────────────────────────────────────────────
# Analysis endpoints (stubs — wired to pipelines in M2+)
# ─────────────────────────────────────────────────────────────

@app.post("/api/v1/analyse/image", tags=["analysis"])
async def analyse_image(file: UploadFile = File(...)):
    """Upload an image for AI-generated content analysis.

    Returns evidence, risk verdict, and explainability artifacts.
    """
    t0 = time.perf_counter()

    # 1. Read and sanitise upload
    try:
        content = await file.read()
        safe_path = validate_upload(
            file_content=content,
            original_filename=file.filename or "unknown.jpg",
            upload_dir=settings.upload_dir / "images",
            is_video=False,
        )
    except UploadSanitiserError as e:
        raise HTTPException(status_code=400, detail=f"Upload rejected: {e}")

    # 2. Create analysis request
    request = AnalysisRequest(
        modality=Modality.IMAGE,
        payload_ref=str(safe_path),
    )

    # 3. TODO (M2): Route to image pipeline → fusion → decision
    #    For now, return a stub response showing the contract works
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = AnalysisResult.from_evidence(
        request_id=request.request_id,
        modality=Modality.IMAGE,
        verdict=RiskVerdict.INCONCLUSIVE,
        confidence=0.0,
        evidence=[],
        explanation_text="Pipeline not yet connected (M1 stub).",
        processing_time_ms=elapsed_ms,
    )

    return result.model_dump()


@app.post("/api/v1/analyse/url", tags=["analysis"])
async def analyse_url(url: str = Form(...)):
    """Submit a URL for phishing / malicious content analysis."""
    t0 = time.perf_counter()

    # 1. SSRF validation
    try:
        validated_url = validate_url(url)
    except SSRFError as e:
        raise HTTPException(status_code=400, detail=f"URL rejected: {e}")

    # 2. Create analysis request
    request = AnalysisRequest(
        modality=Modality.URL,
        payload_ref=validated_url,
    )

    # 3. TODO (M6): Route to URL pipeline → fusion → decision
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = AnalysisResult.from_evidence(
        request_id=request.request_id,
        modality=Modality.URL,
        verdict=RiskVerdict.INCONCLUSIVE,
        confidence=0.0,
        evidence=[],
        explanation_text="Pipeline not yet connected (M1 stub).",
        processing_time_ms=elapsed_ms,
    )

    return result.model_dump()


@app.post("/api/v1/analyse/video", tags=["analysis"])
async def analyse_video(file: UploadFile = File(...)):
    """Upload a video for deepfake detection analysis."""
    t0 = time.perf_counter()

    # 1. Read and sanitise upload
    try:
        content = await file.read()
        safe_path = validate_upload(
            file_content=content,
            original_filename=file.filename or "unknown.mp4",
            upload_dir=settings.upload_dir / "videos",
            is_video=True,
        )
    except UploadSanitiserError as e:
        raise HTTPException(status_code=400, detail=f"Upload rejected: {e}")

    # 2. Create analysis request
    request = AnalysisRequest(
        modality=Modality.VIDEO,
        payload_ref=str(safe_path),
    )

    # 3. TODO (M5): Route to video pipeline → fusion → decision
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = AnalysisResult.from_evidence(
        request_id=request.request_id,
        modality=Modality.VIDEO,
        verdict=RiskVerdict.INCONCLUSIVE,
        confidence=0.0,
        evidence=[],
        explanation_text="Pipeline not yet connected (M1 stub).",
        processing_time_ms=elapsed_ms,
    )

    return result.model_dump()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "authentix.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
