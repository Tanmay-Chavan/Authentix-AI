"""Application configuration — single source of truth for all settings.

Reads from environment variables (.env file) with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_store"
LOGS_DIR = PROJECT_ROOT / "logs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──
    app_name: str = "Authentix AI"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Database ──
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = "geekyguy"
    db_name: str = "authentix"

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync PostgreSQL connection string (for Alembic migrations)."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Security ──
    secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-use-secrets-token-hex-32",
        description="JWT signing key"
    )
    max_upload_size_mb: int = 20
    max_video_upload_size_mb: int = 200

    # ── ML Models ──
    device: str = "auto"  # "auto", "cuda", "cpu"
    batch_size: int = 4
    efficientnet_weights: Optional[str] = None
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"

    # ── Paths ──
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    upload_dir: Path = UPLOAD_DIR


# Singleton instance
settings = Settings()
