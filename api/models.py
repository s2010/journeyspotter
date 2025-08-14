"""
API request and response models.
Defines the external API contract with proper validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, validator

from domain.models import LocationType, MediaType


class LocationResponse(BaseModel):
    """API response model for location data."""
    
    location: str = Field(..., description="Location name")
    country: str = Field(..., description="Country name")
    type: str = Field(..., description="Location type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    class Config:
        schema_extra = {
            "example": {
                "location": "Tokyo Station",
                "country": "Japan",
                "type": "train_station",
                "confidence": 0.85
            }
        }


class AnalysisResponse(BaseModel):
    """API response model for analysis results."""
    
    locations: List[LocationResponse] = Field(default_factory=list, description="Detected locations")
    summary: str = Field(..., description="Analysis summary")
    extracted_text: str = Field(..., description="Text extracted from media")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    file_type: str = Field(..., description="Type of analyzed file")
    filename: str = Field(..., description="Original filename")

    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {
                        "location": "Tokyo Station",
                        "country": "Japan",
                        "type": "train_station",
                        "confidence": 0.85
                    }
                ],
                "summary": "Travel analysis of Tokyo Station area",
                "extracted_text": "TOKYO STATION Platform 7 Shinkansen",
                "confidence": 0.8,
                "file_type": "image",
                "filename": "travel_photo.jpg"
            }
        }


class HealthResponse(BaseModel):
    """API response model for health check."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: Optional[str] = Field(None, description="Service version")

    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "service": "JourneySpotter API",
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """API response model for errors."""
    
    detail: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(None, description="Error type")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Unsupported file format",
                "error_type": "ValidationError"
            }
        }
