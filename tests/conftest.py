"""
Pytest configuration and fixtures for JourneySpotter tests.
Provides common test fixtures and setup.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import AppSettings, LLMSettings, OCRSettings, VideoSettings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    # Create a simple 100x100 RGB image
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def test_settings(temp_dir: Path) -> AppSettings:
    """Create test settings with safe defaults."""
    return AppSettings(
        llm=LLMSettings(api_key="test_key"),
        ocr=OCRSettings(languages=["en"], use_gpu=False),
        video=VideoSettings(max_frames=5),
        temp_dir=temp_dir,
        max_file_size=1024 * 1024  # 1MB for tests
    )


@pytest.fixture
def api_client(test_settings: AppSettings) -> TestClient:
    """Create a test client for the API."""
    app = create_app()
    # Override settings for testing
    app.dependency_overrides[get_app_settings] = lambda: test_settings
    return TestClient(app)


@pytest.fixture
def sample_text() -> str:
    """Sample extracted text for testing."""
    return "Tokyo Station Platform 7 Shinkansen Departure 14:30"
