"""
Unit tests for domain models.
Tests the core business entities and validation logic.
"""

import pytest
from pathlib import Path

from domain.models import (
    AnalysisRequest, AnalysisResult, BoundingBox, Location, 
    LocationType, MediaType, OCRResult
)


class TestBoundingBox:
    """Test BoundingBox domain model."""

    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_invalid_coordinates(self):
        """Test validation of invalid coordinates."""
        with pytest.raises(ValueError):
            BoundingBox(x=-1, y=20, width=100, height=50)
        
        with pytest.raises(ValueError):
            BoundingBox(x=10, y=20, width=0, height=50)


class TestOCRResult:
    """Test OCRResult domain model."""

    def test_valid_ocr_result(self):
        """Test creating a valid OCR result."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        result = OCRResult(text="Tokyo Station", confidence=0.85, bbox=bbox)
        
        assert result.text == "Tokyo Station"
        assert result.confidence == 0.85
        assert result.bbox == bbox

    def test_invalid_confidence(self):
        """Test validation of confidence values."""
        with pytest.raises(ValueError):
            OCRResult(text="Tokyo", confidence=1.5)
        
        with pytest.raises(ValueError):
            OCRResult(text="Tokyo", confidence=-0.1)

    def test_empty_text(self):
        """Test validation of empty text."""
        with pytest.raises(ValueError):
            OCRResult(text="", confidence=0.8)


class TestLocation:
    """Test Location domain model."""

    def test_valid_location(self):
        """Test creating a valid location."""
        location = Location(
            name="Tokyo Station",
            country="Japan",
            location_type=LocationType.TRAIN_STATION,
            confidence=0.9
        )
        
        assert location.name == "Tokyo Station"
        assert location.country == "Japan"
        assert location.location_type == LocationType.TRAIN_STATION
        assert location.confidence == 0.9

    def test_invalid_confidence(self):
        """Test validation of confidence values."""
        with pytest.raises(ValueError):
            Location(
                name="Tokyo",
                country="Japan",
                location_type=LocationType.CITY,
                confidence=2.0
            )


class TestAnalysisRequest:
    """Test AnalysisRequest domain model."""

    def test_valid_request(self, temp_dir: Path):
        """Test creating a valid analysis request."""
        # Create a temporary file
        test_file = temp_dir / "test.jpg"
        test_file.write_text("test content")
        
        request = AnalysisRequest(
            file_path=test_file,
            media_type=MediaType.IMAGE,
            filename="test.jpg"
        )
        
        assert request.file_path == test_file
        assert request.media_type == MediaType.IMAGE
        assert request.filename == "test.jpg"

    def test_nonexistent_file(self, temp_dir: Path):
        """Test validation of nonexistent file."""
        nonexistent_file = temp_dir / "nonexistent.jpg"
        
        with pytest.raises(ValueError):
            AnalysisRequest(
                file_path=nonexistent_file,
                media_type=MediaType.IMAGE,
                filename="test.jpg"
            )


class TestAnalysisResult:
    """Test AnalysisResult domain model."""

    def test_valid_result(self):
        """Test creating a valid analysis result."""
        locations = [
            Location(
                name="Tokyo",
                country="Japan",
                location_type=LocationType.CITY,
                confidence=0.8
            )
        ]
        
        result = AnalysisResult(
            locations=locations,
            summary="Travel analysis complete",
            extracted_text="Tokyo Station",
            confidence=0.85,
            media_type=MediaType.IMAGE,
            filename="test.jpg"
        )
        
        assert len(result.locations) == 1
        assert result.summary == "Travel analysis complete"
        assert result.extracted_text == "Tokyo Station"
        assert result.confidence == 0.85
        assert result.media_type == MediaType.IMAGE
        assert result.filename == "test.jpg"
