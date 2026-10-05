"""Gate 2 and 3 — Integration tests for API endpoints."""

import io
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from authentix.api.app import app
from authentix.core.schemas import Modality, RiskVerdict

@pytest.fixture
def test_image():
    """Create a minimal valid JPEG in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    # Save with minimal EXIF to be realistic
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.mark.asyncio
async def test_health_endpoint():
    """Gate 2: Verify server starts and health check passes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_analyse_image_stub(test_image):
    """Gate 3: Verify image upload sanitisation and stub response."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/analyse/image",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["modality"] == Modality.IMAGE.value
    assert data["verdict"] == RiskVerdict.INCONCLUSIVE.value
    assert "stub" in data["explanation_text"]


@pytest.mark.asyncio
async def test_analyse_url_stub():
    """Verify URL analysis endpoint rejects SSRF and accepts safe URL."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # SSRF blocked
        response = await ac.post("/api/v1/analyse/url", data={"url": "http://127.0.0.1"})
        assert response.status_code == 400

        # Safe URL allowed
        response = await ac.post("/api/v1/analyse/url", data={"url": "https://www.google.com"})
        assert response.status_code == 200
        assert response.json()["modality"] == Modality.URL.value
