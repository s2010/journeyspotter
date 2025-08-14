"""
OCR adapter implementation using EasyOCR and Tesseract.
Provides text extraction capabilities for images and video frames.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config.settings import OCRSettings
from core.interfaces import OCRProcessor, VideoProcessor
from domain.models import BoundingBox, OCRResult

# Optional imports with fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EasyOCRAdapter(OCRProcessor):
    """EasyOCR implementation of OCR processing."""

    def __init__(self, settings: OCRSettings) -> None:
        """Initialize EasyOCR adapter."""
        self.settings = settings
        self.reader: Optional[easyocr.Reader] = None
        
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR is not available. Install with: pip install easyocr")
        
        self._initialize_reader()

    def _initialize_reader(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            self.reader = easyocr.Reader(
                self.settings.languages,
                gpu=self.settings.use_gpu
            )
            logger.info(f"EasyOCR initialized with languages: {self.settings.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    async def extract_text_from_image(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from image using EasyOCR."""
        if self.reader is None:
            raise RuntimeError("EasyOCR reader not initialized")
        
        if image is None or image.size == 0:
            return []

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text with EasyOCR
            results = self.reader.readtext(processed_image)
            
            ocr_results = []
            for (bbox_coords, text, confidence) in results:
                if text.strip() and confidence > 0.1:  # Filter low confidence results
                    # Convert bbox coordinates
                    bbox = self._convert_bbox(bbox_coords)
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=float(confidence),
                        bbox=bbox
                    ))
            
            logger.debug(f"EasyOCR extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR text extraction failed: {e}")
            return []

    async def extract_text_from_video(self, video_path: Path, max_frames: int = 10) -> str:
        """Extract text from video frames using EasyOCR."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                return ""

            # Sample frames evenly
            frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            
            all_text = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    ocr_results = await self.extract_text_from_image(frame)
                    frame_text = " ".join([result.text for result in ocr_results])
                    if frame_text.strip():
                        all_text.append(frame_text)

            cap.release()
            return " ".join(all_text)
            
        except Exception as e:
            logger.error(f"Video text extraction failed: {e}")
            return ""

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
        
        return enhanced

    def _convert_bbox(self, bbox_coords: List[List[int]]) -> Optional[BoundingBox]:
        """Convert EasyOCR bbox coordinates to BoundingBox."""
        try:
            # EasyOCR returns 4 corner points, convert to x, y, width, height
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            
            x = min(x_coords)
            y = min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y
            
            return BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
        except Exception:
            return None


class TesseractOCRAdapter(OCRProcessor):
    """Tesseract implementation of OCR processing."""

    def __init__(self, settings: OCRSettings) -> None:
        """Initialize Tesseract adapter."""
        self.settings = settings
        
        if not PYTESSERACT_AVAILABLE:
            raise ImportError("Tesseract is not available. Install with: pip install pytesseract")
        
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        
        logger.info("Tesseract OCR initialized")

    async def extract_text_from_image(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text from image using Tesseract."""
        if image is None or image.size == 0:
            return []

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                lang="+".join(self.settings.languages)
            )
            
            ocr_results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                if text and confidence > 0.1:  # Filter low confidence results
                    bbox = BoundingBox(
                        x=int(data['left'][i]),
                        y=int(data['top'][i]),
                        width=int(data['width'][i]),
                        height=int(data['height'][i])
                    )
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox
                    ))
            
            logger.debug(f"Tesseract extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract text extraction failed: {e}")
            return []

    async def extract_text_from_video(self, video_path: Path, max_frames: int = 10) -> str:
        """Extract text from video frames using Tesseract."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                return ""

            # Sample frames evenly
            frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            
            all_text = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    ocr_results = await self.extract_text_from_image(frame)
                    frame_text = " ".join([result.text for result in ocr_results])
                    if frame_text.strip():
                        all_text.append(frame_text)

            cap.release()
            return " ".join(all_text)
            
        except Exception as e:
            logger.error(f"Video text extraction failed: {e}")
            return ""

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold for better text detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh


def create_ocr_processor(settings: OCRSettings, prefer_easyocr: bool = True) -> OCRProcessor:
    """Factory function to create appropriate OCR processor."""
    if prefer_easyocr and EASYOCR_AVAILABLE:
        return EasyOCRAdapter(settings)
    elif PYTESSERACT_AVAILABLE:
        return TesseractOCRAdapter(settings)
    else:
        raise ImportError("No OCR library available. Install either easyocr or pytesseract")
