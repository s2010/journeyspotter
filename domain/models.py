"""
Domain models for JourneySpotter application.
Contains core business entities and value objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from pathlib import Path


class MediaType(str, Enum):
    """Supported media types for analysis."""
    IMAGE = "image"
    VIDEO = "video"


class LocationType(str, Enum):
    """Types of locations that can be detected."""
    TRAIN_STATION = "train_station"
    AIRPORT = "airport"
    BUS_STOP = "bus_stop"
    CITY = "city"
    LANDMARK = "landmark"
    TRANSPORTATION_HUB = "transportation_hub"
    DISTRICT = "district"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates for detected text."""
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid bounding box coordinates")


@dataclass(frozen=True)
class OCRResult:
    """Result from OCR text extraction."""
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None

    def __post_init__(self) -> None:
        """Validate OCR result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.text.strip():
            raise ValueError("Text cannot be empty")


@dataclass(frozen=True)
class Location:
    """A detected location with metadata."""
    name: str
    country: str
    location_type: LocationType
    confidence: float

    def __post_init__(self) -> None:
        """Validate location data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.name.strip():
            raise ValueError("Location name cannot be empty")


@dataclass(frozen=True)
class AnalysisRequest:
    """Request for media analysis."""
    file_path: Path
    media_type: MediaType
    filename: str

    def __post_init__(self) -> None:
        """Validate analysis request."""
        if not self.file_path.exists():
            raise ValueError(f"File does not exist: {self.file_path}")
        if not self.filename.strip():
            raise ValueError("Filename cannot be empty")


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis result for a media file."""
    locations: List[Location]
    summary: str
    extracted_text: str
    confidence: float
    media_type: MediaType
    filename: str
    # Anomaly detection results
    anomaly_scores: Optional[List[float]] = None
    anomaly_threshold: Optional[float] = None
    anomalous_frames: Optional[List[bool]] = None
    anomaly_detected: bool = False

    def __post_init__(self) -> None:
        """Validate analysis result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.filename.strip():
            raise ValueError("Filename cannot be empty")
