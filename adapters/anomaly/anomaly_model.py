"""
Two-stage anomaly detection model combining ResNet-18 and IsolationForest.
Implements the complete anomaly detection pipeline with training and inference.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import asyncio

import cv2
import numpy as np
import joblib

from config.settings import AnomalySettings
from domain.ports import AnomalyModel, FrameEncoder, AnomalyScorer
from .frame_encoder import create_frame_encoder
from .anomaly_scorer import create_anomaly_scorer

logger = logging.getLogger(__name__)


class TwoStageAnomalyModel(AnomalyModel):
    """Two-stage anomaly detection model using ResNet-18 + IsolationForest."""

    def __init__(self, settings: AnomalySettings) -> None:
        """Initialize two-stage anomaly model."""
        self.settings = settings
        self.frame_encoder = create_frame_encoder(settings)
        self.anomaly_scorer = create_anomaly_scorer(settings)
        self._is_trained = False
        
        # Ensure model directory exists
        model_path = Path(settings.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Two-stage anomaly model initialized")

    async def train(self, normal_video_path: Path) -> None:
        """
        Train the anomaly detection model on normal video data.
        
        Args:
            normal_video_path: Path to video file containing normal behavior
        """
        if not normal_video_path.exists():
            raise FileNotFoundError(f"Training video not found: {normal_video_path}")
        
        logger.info(f"Starting anomaly model training on: {normal_video_path}")
        
        try:
            # Extract frames from normal video
            frames = await self._extract_frames_for_training(normal_video_path)
            
            if not frames:
                raise ValueError("No frames extracted from training video")
            
            logger.info(f"Extracted {len(frames)} frames for training")
            
            # Encode frames to embeddings using ResNet-18
            embeddings = await self.frame_encoder.encode_frames(frames)
            
            if embeddings.size == 0:
                raise ValueError("No embeddings generated from training frames")
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Train IsolationForest on embeddings
            await self.anomaly_scorer.fit(embeddings)
            
            # Save the trained model
            await self.save_model(Path(self.settings.model_path))
            
            self._is_trained = True
            logger.info("Anomaly model training completed successfully")
            
        except Exception as e:
            logger.error(f"Anomaly model training failed: {e}")
            raise

    async def score(self, video_path: Path) -> Tuple[List[float], float]:
        """
        Score video frames for anomalies.
        
        Args:
            video_path: Path to video file to analyze
            
        Returns:
            Tuple of (anomaly_scores, threshold) where:
            - anomaly_scores: List of anomaly scores per frame
            - threshold: Anomaly detection threshold
        """
        if not await self.is_trained():
            raise ValueError("Model must be trained before scoring")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Scoring video for anomalies: {video_path}")
        
        try:
            # Extract frames from video
            frames = await self._extract_frames_for_inference(video_path)
            
            if not frames:
                logger.warning("No frames extracted from video")
                return [], self.settings.anomaly_threshold
            
            logger.debug(f"Extracted {len(frames)} frames for scoring")
            
            # Encode frames to embeddings
            embeddings = await self.frame_encoder.encode_frames(frames)
            
            if embeddings.size == 0:
                logger.warning("No embeddings generated from frames")
                return [], self.settings.anomaly_threshold
            
            # Score embeddings for anomalies
            scores = await self.anomaly_scorer.score_samples(embeddings)
            
            # Convert to list for JSON serialization
            anomaly_scores = scores.tolist()
            
            logger.info(f"Generated {len(anomaly_scores)} anomaly scores")
            logger.debug(f"Score range: [{min(anomaly_scores):.4f}, {max(anomaly_scores):.4f}]")
            
            return anomaly_scores, self.settings.anomaly_threshold
            
        except Exception as e:
            logger.error(f"Anomaly scoring failed: {e}")
            raise

    async def is_trained(self) -> bool:
        """Check if the model has been trained and is ready for inference."""
        if self._is_trained:
            return True
        
        # Check if model file exists and try to load it
        model_path = Path(self.settings.model_path)
        if model_path.exists():
            try:
                await self.load_model(model_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                return False
        
        return False

    async def save_model(self, model_path: Path) -> None:
        """Save the trained model to disk."""
        try:
            # Ensure directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the IsolationForest model
            if hasattr(self.anomaly_scorer, 'model') and self.anomaly_scorer.model is not None:
                joblib.dump(self.anomaly_scorer.model, model_path)
                logger.info(f"Anomaly model saved to: {model_path}")
            else:
                raise ValueError("No trained model to save")
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    async def load_model(self, model_path: Path) -> None:
        """Load a trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load the IsolationForest model
            loaded_model = joblib.load(model_path)
            
            # Set the loaded model in the scorer
            self.anomaly_scorer.model = loaded_model
            self.anomaly_scorer._is_fitted = True
            
            self._is_trained = True
            logger.info(f"Anomaly model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def _extract_frames_for_training(self, video_path: Path) -> List[np.ndarray]:
        """Extract frames from video for training at specified frame rate."""
        return await self._extract_frames(video_path, self.settings.frame_rate, max_frames=None)

    async def _extract_frames_for_inference(self, video_path: Path) -> List[np.ndarray]:
        """Extract frames from video for inference with frame limit."""
        return await self._extract_frames(
            video_path, 
            self.settings.frame_rate, 
            max_frames=self.settings.max_inference_frames
        )

    async def _extract_frames(self, video_path: Path, fps: float, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video at specified frame rate.
        
        Args:
            video_path: Path to video file
            fps: Frame extraction rate (frames per second)
            max_frames: Maximum number of frames to extract (None for no limit)
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if video_fps <= 0:
                logger.warning("Could not determine video FPS, using default 30")
                video_fps = 30.0
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / fps))
            
            logger.debug(f"Video: {video_fps} fps, extracting every {frame_interval} frames")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames is not None and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.debug(f"Extracted {len(frames)} frames from {total_frames} total frames")
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
        
        return frames


def create_anomaly_model(settings: AnomalySettings) -> AnomalyModel:
    """Factory function to create anomaly model."""
    return TwoStageAnomalyModel(settings)
