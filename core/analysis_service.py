"""
Core analysis service implementation.
Orchestrates OCR, LLM, and video processing for media analysis.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config.settings import AppSettings
from core.interfaces import AnalysisService, LLMProcessor, OCRProcessor, VideoProcessor
from domain.models import AnalysisRequest, AnalysisResult, Location, LocationType, MediaType

logger = logging.getLogger(__name__)


class MediaAnalysisService(AnalysisService):
    """Core service for analyzing media files."""

    def __init__(
        self,
        ocr_processor: OCRProcessor,
        llm_processor: LLMProcessor,
        video_processor: VideoProcessor,
        settings: AppSettings
    ) -> None:
        """Initialize analysis service with injected dependencies."""
        self.ocr_processor = ocr_processor
        self.llm_processor = llm_processor
        self.video_processor = video_processor
        self.settings = settings
        logger.info("Media analysis service initialized")

    async def analyze_media(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze media file and return comprehensive results."""
        logger.info(f"Starting analysis for {request.media_type} file: {request.filename}")
        
        try:
            # Extract text based on media type
            if request.media_type == MediaType.IMAGE:
                extracted_text = await self._analyze_image(request.file_path)
            elif request.media_type == MediaType.VIDEO:
                extracted_text = await self._analyze_video(request.file_path)
            else:
                raise ValueError(f"Unsupported media type: {request.media_type}")

            # Analyze extracted text with LLM
            llm_result = await self.llm_processor.analyze_travel_content(extracted_text)
            
            # Convert to domain objects
            locations = self._convert_locations(llm_result.get("locations", []))
            
            result = AnalysisResult(
                locations=locations,
                summary=llm_result.get("summary", "Analysis completed"),
                extracted_text=extracted_text,
                confidence=float(llm_result.get("confidence", 0.0)),
                media_type=request.media_type,
                filename=request.filename
            )
            
            logger.info(f"Analysis completed for {request.filename}: {len(locations)} locations found")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {request.filename}: {e}")
            # Return error result
            return AnalysisResult(
                locations=[],
                summary=f"Analysis failed: {str(e)}",
                extracted_text="",
                confidence=0.0,
                media_type=request.media_type,
                filename=request.filename
            )

    async def _analyze_image(self, image_path: Path) -> str:
        """Extract text from a single image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract text using OCR
            ocr_results = await self.ocr_processor.extract_text_from_image(image)
            
            # Combine all detected text
            extracted_text = " ".join([result.text for result in ocr_results if result.text.strip()])
            
            logger.debug(f"Extracted text from image: {len(extracted_text)} characters")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise

    async def _analyze_video(self, video_path: Path) -> str:
        """Extract text from video frames."""
        try:
            # Extract frames from video
            frames = await self.video_processor.extract_frames(
                video_path, 
                max_frames=self.settings.video.max_frames
            )
            
            if not frames:
                logger.warning(f"No frames extracted from video: {video_path}")
                return ""
            
            # Process each frame with OCR
            all_text = []
            for i, frame in enumerate(frames):
                try:
                    ocr_results = await self.ocr_processor.extract_text_from_image(frame)
                    frame_text = " ".join([result.text for result in ocr_results if result.text.strip()])
                    if frame_text:
                        all_text.append(frame_text)
                        logger.debug(f"Frame {i}: extracted {len(frame_text)} characters")
                except Exception as e:
                    logger.warning(f"Failed to process frame {i}: {e}")
                    continue
            
            # Combine all extracted text
            combined_text = " ".join(all_text)
            logger.debug(f"Extracted text from video: {len(combined_text)} characters from {len(frames)} frames")
            return combined_text
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise

    def _convert_locations(self, locations_data: List[dict]) -> List[Location]:
        """Convert location dictionaries to Location domain objects."""
        locations = []
        
        for loc_data in locations_data:
            try:
                # Parse location type
                location_type_str = loc_data.get("type", "unknown")
                if isinstance(location_type_str, str):
                    try:
                        location_type = LocationType(location_type_str)
                    except ValueError:
                        location_type = LocationType.UNKNOWN
                else:
                    location_type = LocationType.UNKNOWN
                
                location = Location(
                    name=str(loc_data.get("location", "")).strip(),
                    country=str(loc_data.get("country", "")).strip(),
                    location_type=location_type,
                    confidence=float(loc_data.get("confidence", 0.0))
                )
                
                if location.name:  # Only add if name is not empty
                    locations.append(location)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid location data: {loc_data}, error: {e}")
                continue
        
        return locations


def create_analysis_service(
    ocr_processor: OCRProcessor,
    llm_processor: LLMProcessor,
    video_processor: VideoProcessor,
    settings: AppSettings
) -> AnalysisService:
    """Factory function to create analysis service."""
    return MediaAnalysisService(ocr_processor, llm_processor, video_processor, settings)
