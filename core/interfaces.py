"""
Core interfaces and abstract base classes for dependency injection.
Defines contracts for all adapters and services.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np

from domain.models import AnalysisRequest, AnalysisResult, Location, OCRResult


class OCRProcessor(ABC):
    """Abstract base class for OCR processing."""

    @abstractmethod
    async def extract_text_from_image(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from a single image."""
        pass

    @abstractmethod
    async def extract_text_from_video(self, video_path: Path, max_frames: int = 10) -> str:
        """Extract text from video frames."""
        pass


class LLMProcessor(ABC):
    """Abstract base class for LLM processing."""

    @abstractmethod
    async def analyze_content(self, extracted_text: str) -> Dict[str, any]:
        """Analyze extracted text for intelligent content analysis."""
        pass


class VideoProcessor(ABC):
    """Abstract base class for video processing."""

    @abstractmethod
    async def extract_frames(self, video_path: Path, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        pass


class AnalysisService(ABC):
    """Abstract base class for analysis service."""

    @abstractmethod
    async def analyze_media(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze media file and return results."""
        pass


class FileStorage(ABC):
    """Abstract base class for file storage operations."""

    @abstractmethod
    async def save_temp_file(self, content: bytes, filename: str) -> Path:
        """Save uploaded file temporarily."""
        pass

    @abstractmethod
    async def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up temporary file."""
        pass
