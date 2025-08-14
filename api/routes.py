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
from api.models import AnalysisResponse, ErrorResponse, HealthResponse, LocationResponse
from config.settings import AppSettings
from core.interfaces import AnalysisService, FileStorage
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
    Analyze uploaded video/image for travel intelligence.
    
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
        
        # Convert to API response format
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
            filename=result.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path:
            try:
                await file_storage.cleanup_temp_file(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


# Error handlers are registered in main.py, not on the router
