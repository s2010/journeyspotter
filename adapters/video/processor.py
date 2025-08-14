"""
Video adapter implementation for frame extraction and processing.
Provides video frame sampling capabilities for OCR analysis.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config.settings import VideoSettings
from core.interfaces import VideoProcessor

logger = logging.getLogger(__name__)


class OpenCVVideoAdapter(VideoProcessor):
    """OpenCV implementation for video processing."""

    def __init__(self, settings: VideoSettings) -> None:
        """Initialize video processor."""
        self.settings = settings
        logger.info("OpenCV video processor initialized")

    async def extract_frames(self, video_path: Path, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                logger.warning(f"Video has no frames: {video_path}")
                return []

            # Determine frame sampling strategy
            max_frames = min(max_frames, frame_count)
            
            if self.settings.frame_sampling == "uniform":
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            else:
                # Default to uniform sampling
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame {frame_idx} from {video_path}")

            cap.release()
            logger.debug(f"Extracted {len(frames)} frames from video: {video_path}")
            return frames

        except Exception as e:
            logger.error(f"Video frame extraction failed: {e}")
            return []


def create_video_processor(settings: VideoSettings) -> VideoProcessor:
    """Factory function to create video processor."""
    return OpenCVVideoAdapter(settings)
