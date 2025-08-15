"""
FastAPI routes for JourneySpotter API.
Implements the external API endpoints with proper validation and error handling.
"""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.dependencies import get_analysis_service, get_app_settings, get_file_storage
from api.models import AnalysisResponse, ErrorResponse, HealthResponse, LocationResponse, AnomalyTrainingResponse, AnomalyModelInfoResponse
from config.settings import AppSettings
from core.interfaces import AnalysisService, FileStorage
from core.anomaly_service import create_anomaly_service
from domain.models import AnalysisRequest, Location, MediaType

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[AppSettings, Depends(get_app_settings)]
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="JourneySpotter API",
        version=settings.version
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_media(
    file: UploadFile = File(...),
    analysis_service: Annotated[AnalysisService, Depends(get_analysis_service)] = None,
    file_storage: Annotated[FileStorage, Depends(get_file_storage)] = None,
    settings: Annotated[AppSettings, Depends(get_app_settings)] = None
) -> AnalysisResponse:
    """
    Analyze uploaded video/image for intelligent content analysis.
    
    Accepts: video files (.mp4, .avi, .mov, .mkv) or image files (.jpg, .jpeg, .png, .bmp, .tiff)
    Returns: JSON with locations, summary, extracted text, and confidence score
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALL_SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported: {sorted(ALL_SUPPORTED_EXTENSIONS)}"
        )
    
    # Determine media type
    if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
        media_type = MediaType.IMAGE
    else:
        media_type = MediaType.VIDEO
    
    # Read file content
    try:
        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    temp_file_path = None
    try:
        # Save file temporarily
        temp_file_path = await file_storage.save_temp_file(content, file.filename)
        
        # Create analysis request
        request = AnalysisRequest(
            file_path=temp_file_path,
            media_type=media_type,
            filename=file.filename
        )
        
        # Perform analysis
        result = await analysis_service.analyze_media(request)
        
        # Convert domain result to API response
        locations_response = [
            LocationResponse(
                location=loc.name,
                country=loc.country,
                type=loc.location_type.value,
                confidence=loc.confidence
            )
            for loc in result.locations
        ]
        
        return AnalysisResponse(
            locations=locations_response,
            summary=result.summary,
            extracted_text=result.extracted_text,
            confidence=result.confidence,
            file_type=result.media_type.value,
            filename=result.filename,
            anomaly_scores=result.anomaly_scores,
            anomaly_threshold=result.anomaly_threshold,
            anomalous_frames=result.anomalous_frames,
            anomaly_detected=result.anomaly_detected
        )      
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path:
            try:
                await file_storage.cleanup_temp_file(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@router.post("/anomaly/train", response_model=AnomalyTrainingResponse)
async def train_anomaly_model(
    file: UploadFile = File(...),
    file_storage: Annotated[FileStorage, Depends(get_file_storage)] = None,
    settings: Annotated[AppSettings, Depends(get_app_settings)] = None
) -> AnomalyTrainingResponse:
    """
    Train the anomaly detection model on normal video data.
    
    Accepts: video files (.mp4, .avi, .mov, .mkv) containing normal behavior
    Returns: Training result with success status and model information
    """
    temp_file_path = None
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
            )
        
        # Save uploaded file
        temp_file_path = await file_storage.save_file(file)
        logger.info(f"Training file saved: {temp_file_path}")
        
        # Create anomaly service and train model
        anomaly_service = create_anomaly_service(settings.anomaly)
        success = await anomaly_service.train_model(temp_file_path)
        
        if success:
            model_info = await anomaly_service.get_model_info()
            return AnomalyTrainingResponse(
                success=True,
                message="Anomaly model trained successfully",
                model_info=model_info
            )
        else:
            return AnomalyTrainingResponse(
                success=False,
                message="Anomaly model training failed",
                model_info=None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path:
            try:
                await file_storage.cleanup_temp_file(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@router.get("/anomaly/model", response_model=AnomalyModelInfoResponse)
async def get_anomaly_model_info(
    settings: Annotated[AppSettings, Depends(get_app_settings)] = None
) -> AnomalyModelInfoResponse:
    """
    Get information about the current anomaly detection model.
    
    Returns: Model configuration and training status
    """
    try:
        anomaly_service = create_anomaly_service(settings.anomaly)
        model_info = await anomaly_service.get_model_info()
        
        return AnomalyModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# Error handlers are registered in main.py, not on the router
