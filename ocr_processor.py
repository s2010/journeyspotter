#!/usr/bin/env python3
"""
OCR Processor Module
Extract text from video frames and images for travel intelligence
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass

# OCR libraries - will try both for best results
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR detection result"""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)

class OCRProcessor:
    """Text extraction from images and video frames"""
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (e.g., ['en', 'fr', 'de'])
            use_gpu: Whether to use GPU acceleration (for EasyOCR)
        """
        self.languages = languages
        self.use_gpu = use_gpu
        
        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
                logger.info(f"EasyOCR initialized with languages: {languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None
            logger.warning("EasyOCR not available")
        
        # Check Tesseract availability
        if PYTESSERACT_AVAILABLE:
            try:
                # Test Tesseract
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR available")
            except Exception as e:
                logger.warning(f"Tesseract not properly configured: {e}")
        else:
            logger.warning("Pytesseract not available")
    
    def extract_text_from_image(self, image: np.ndarray, method: str = 'auto') -> List[OCRResult]:
        """
        Extract text from a single image
        
        Args:
            image: Input image as numpy array (BGR format)
            method: OCR method ('easyocr', 'tesseract', 'auto')
            
        Returns:
            List of OCRResult objects
        """
        if image is None or image.size == 0:
            return []
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        results = []
        
        if method == 'auto':
            # Try EasyOCR first, then Tesseract as fallback
            if self.easyocr_reader:
                results = self._extract_with_easyocr(processed_image)
            
            if not results and PYTESSERACT_AVAILABLE:
                results = self._extract_with_tesseract(processed_image)
                
        elif method == 'easyocr' and self.easyocr_reader:
            results = self._extract_with_easyocr(processed_image)
            
        elif method == 'tesseract' and PYTESSERACT_AVAILABLE:
            results = self._extract_with_tesseract(processed_image)
        
        # Filter and clean results
        results = self._filter_results(results)
        
        return results
    
    def extract_text_from_video(self, video_path: str, sample_rate: int = 30) -> Dict[int, List[OCRResult]]:
        """
        Extract text from video frames at specified intervals
        
        Args:
            video_path: Path to video file
            sample_rate: Extract text every N frames
            
        Returns:
            Dictionary mapping frame numbers to OCR results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        results = {}
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every sample_rate frames
            if frame_count % sample_rate == 0:
                ocr_results = self.extract_text_from_image(frame)
                if ocr_results:  # Only store frames with detected text
                    results[frame_count] = ocr_results
                    logger.debug(f"Frame {frame_count}: Found {len(ocr_results)} text regions")
            
            frame_count += 1
            
            # Progress logging
            if frame_count % (sample_rate * 10) == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        logger.info(f"Video processing complete. Found text in {len(results)} frames.")
        
        return results
    
    def combine_text_results(self, ocr_results: Dict[int, List[OCRResult]]) -> str:
        """
        Combine all OCR results into a single text string
        
        Args:
            ocr_results: Dictionary of frame number to OCR results
            
        Returns:
            Combined text string
        """
        all_text = []
        seen_text = set()  # Avoid duplicates
        
        for frame_num in sorted(ocr_results.keys()):
            frame_results = ocr_results[frame_num]
            
            for result in frame_results:
                # Clean and normalize text
                text = result.text.strip()
                if len(text) < 2:  # Skip very short text
                    continue
                
                # Simple deduplication
                text_normalized = re.sub(r'\s+', ' ', text.lower())
                if text_normalized not in seen_text:
                    all_text.append(text)
                    seen_text.add(text_normalized)
        
        return ' | '.join(all_text)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding for better text contrast
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Upscale for better OCR (smaller text)
        height, width = cleaned.shape
        if height < 400 or width < 400:
            scale_factor = max(400 / height, 400 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            ocr_results = []
            
            for detection in results:
                bbox, text, confidence = detection
                
                # Convert bbox to (x, y, width, height)
                if bbox and len(bbox) >= 2:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    bbox_rect = (x, y, width, height)
                else:
                    bbox_rect = None
                
                ocr_results.append(OCRResult(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=bbox_rect
                ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using Tesseract"""
        try:
            # Get detailed output with bounding boxes
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Uniform block of text
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Filter out low confidence and empty results
                if conf > 30 and len(text) > 1:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=conf / 100.0,  # Convert to 0-1 scale
                        bbox=(x, y, w, h)
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return []
    
    def _filter_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """
        Filter and clean OCR results
        
        Args:
            results: Raw OCR results
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            text = result.text.strip()
            
            # Skip very short text
            if len(text) < 2:
                continue
            
            # Skip text that's mostly numbers or symbols (likely noise)
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.3:
                continue
            
            # Skip very low confidence
            if result.confidence < 0.5:
                continue
            
            # Clean up common OCR artifacts
            text = re.sub(r'[^\w\s\-\.,;:()\[\]{}]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) >= 2:
                filtered.append(OCRResult(
                    text=text,
                    confidence=result.confidence,
                    bbox=result.bbox
                ))
        
        return filtered
    
    def visualize_text_detection(self, image: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """
        Draw bounding boxes around detected text
        
        Args:
            image: Input image
            results: OCR results with bounding boxes
            
        Returns:
            Image with text regions highlighted
        """
        output = image.copy()
        
        for result in results:
            if result.bbox:
                x, y, w, h = result.bbox
                # Draw bounding box
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text label
                label = f"{result.text} ({result.confidence:.2f})"
                cv2.putText(output, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output

def demo_ocr_processor():
    """Demo function to test OCR functionality"""
    try:
        # Create OCR processor
        ocr = OCRProcessor()
        
        # Test with a simple text image
        # Create a sample image with text
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "DEPARTURE: TOKYO 14:30", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "ARRIVAL: KYOTO 17:45", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "JR SHINKANSEN", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        print("Testing OCR on sample image...")
        results = ocr.extract_text_from_image(img)
        
        print(f"Found {len(results)} text regions:")
        for i, result in enumerate(results, 1):
            print(f"{i}. '{result.text}' (confidence: {result.confidence:.2f})")
        
        # Combine all text
        combined_text = ' '.join([r.text for r in results])
        print(f"\nCombined text: {combined_text}")
        
        # Show visualization
        vis_img = ocr.visualize_text_detection(img, results)
        cv2.imshow("OCR Demo", vis_img)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure you have easyocr or pytesseract installed:")
        print("pip install easyocr")
        print("# OR")
        print("pip install pytesseract")

if __name__ == "__main__":
    demo_ocr_processor() 