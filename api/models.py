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
    # Anomaly detection fields
    anomaly_scores: Optional[List[float]] = Field(None, description="Anomaly scores per frame (videos only)")
    anomaly_threshold: Optional[float] = Field(None, description="Anomaly detection threshold")
    anomalous_frames: Optional[List[bool]] = Field(None, description="Boolean flags for anomalous frames")
    anomaly_detected: bool = Field(False, description="Whether anomalies were detected")

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
                "summary": "Content analysis of Tokyo Station area",
                "extracted_text": "TOKYO STATION Platform 7 Shinkansen",
                "confidence": 0.8,
                "file_type": "image",
                "filename": "sample_photo.jpg"
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


class AnomalyTrainingResponse(BaseModel):
    """API response model for anomaly training results."""
    
    success: bool = Field(..., description="Training success status")
    message: str = Field(..., description="Training result message")
    model_info: Optional[dict] = Field(None, description="Model information after training")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Anomaly model trained successfully",
                "model_info": {
                    "is_trained": True,
                    "resnet_model": "resnet18",
                    "contamination": 0.1
                }
            }
        }


class AnomalyModelInfoResponse(BaseModel):
    """API response model for anomaly model information."""
    
    is_trained: bool = Field(..., description="Whether model is trained")
    model_path: str = Field(..., description="Path to model file")
    model_exists: bool = Field(..., description="Whether model file exists")
    resnet_model: str = Field(..., description="ResNet model variant")
    contamination: float = Field(..., description="IsolationForest contamination parameter")
    n_estimators: int = Field(..., description="Number of estimators")
    anomaly_threshold: float = Field(..., description="Anomaly detection threshold")
    device: str = Field(..., description="PyTorch device")

    class Config:
        schema_extra = {
            "example": {
                "is_trained": True,
                "model_path": "models/anomaly_model.joblib",
                "model_exists": True,
                "resnet_model": "resnet18",
                "contamination": 0.1,
                "n_estimators": 100,
                "anomaly_threshold": -0.1,
                "device": "cpu"
            }
        }


class ErrorResponse(BaseModel):
    """API response model for errors."""
    
    detail: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(None, description="Error type")

    class Config:
        schema_extra = {
            "example": {
                "detail": "File format not supported",
                "error_type": "ValidationError"
            }
        }
