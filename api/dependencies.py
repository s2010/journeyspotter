"""
FastAPI dependency injection setup.
Provides singleton instances of services and adapters.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from adapters.llm.processor import create_llm_processor
from adapters.ocr.processor import create_ocr_processor
from adapters.storage.file_storage import create_file_storage
from adapters.video.processor import create_video_processor
from config.settings import AppSettings, get_settings
from core.analysis_service import create_analysis_service
from core.interfaces import AnalysisService, FileStorage, LLMProcessor, OCRProcessor, VideoProcessor


def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return get_settings()


def get_ocr_processor(settings: Annotated[AppSettings, Depends(get_app_settings)]) -> OCRProcessor:
    """Get OCR processor instance."""
    return create_ocr_processor(settings.ocr)


def get_llm_processor(settings: Annotated[AppSettings, Depends(get_app_settings)]) -> LLMProcessor:
    """Get LLM processor instance."""
    return create_llm_processor(settings.llm)


def get_video_processor(settings: Annotated[AppSettings, Depends(get_app_settings)]) -> VideoProcessor:
    """Get video processor instance."""
    return create_video_processor(settings.video)


def get_file_storage(settings: Annotated[AppSettings, Depends(get_app_settings)]) -> FileStorage:
    """Get file storage instance."""
    return create_file_storage(settings)


def get_analysis_service(
    ocr_processor: Annotated[OCRProcessor, Depends(get_ocr_processor)],
    llm_processor: Annotated[LLMProcessor, Depends(get_llm_processor)],
    video_processor: Annotated[VideoProcessor, Depends(get_video_processor)],
    settings: Annotated[AppSettings, Depends(get_app_settings)]
) -> AnalysisService:
    """Get analysis service instance."""
    return create_analysis_service(
        ocr_processor, llm_processor, video_processor, settings
    )
