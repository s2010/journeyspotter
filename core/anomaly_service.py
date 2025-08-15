"""
Core anomaly detection service.
Provides high-level anomaly detection functionality for the application.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from config.settings import AnomalySettings
from domain.ports import AnomalyModel
from adapters.anomaly import TwoStageAnomalyModel

logger = logging.getLogger(__name__)


class AnomalyService:
    """Service for anomaly detection operations."""

    def __init__(self, settings: AnomalySettings) -> None:
        """Initialize anomaly service."""
        self.settings = settings
        self.model: AnomalyModel = TwoStageAnomalyModel(settings)
        
        logger.info("Anomaly service initialized")

    async def train_model(self, normal_video_path: Path) -> bool:
        """
        Train the anomaly detection model on normal video data.
        
        Args:
            normal_video_path: Path to video file containing normal behavior
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            logger.info(f"Starting anomaly model training with: {normal_video_path}")
            
            await self.model.train(normal_video_path)
            
            logger.info("Anomaly model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Anomaly model training failed: {e}")
            return False

    async def detect_anomalies(self, video_path: Path) -> Optional[Tuple[List[float], float, List[bool]]]:
        """
        Detect anomalies in video frames.
        
        Args:
            video_path: Path to video file to analyze
            
        Returns:
            Tuple of (anomaly_scores, threshold, is_anomalous) or None if detection fails
            - anomaly_scores: List of anomaly scores per frame
            - threshold: Anomaly detection threshold
            - is_anomalous: List of boolean flags indicating anomalous frames
        """
        try:
            # Check if model is trained
            if not await self.model.is_trained():
                logger.warning("Anomaly model is not trained, cannot detect anomalies")
                return None
            
            logger.info(f"Detecting anomalies in: {video_path}")
            
            # Get anomaly scores
            anomaly_scores, threshold = await self.model.score(video_path)
            
            if not anomaly_scores:
                logger.warning("No anomaly scores generated")
                return None
            
            # Determine which frames are anomalous based on threshold
            is_anomalous = [score < threshold for score in anomaly_scores]
            
            anomaly_count = sum(is_anomalous)
            total_frames = len(anomaly_scores)
            anomaly_percentage = (anomaly_count / total_frames) * 100 if total_frames > 0 else 0
            
            logger.info(f"Anomaly detection completed: {anomaly_count}/{total_frames} frames ({anomaly_percentage:.1f}%) flagged as anomalous")
            
            return anomaly_scores, threshold, is_anomalous
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return None

    async def is_model_trained(self) -> bool:
        """Check if the anomaly model is trained and ready."""
        try:
            return await self.model.is_trained()
        except Exception as e:
            logger.error(f"Failed to check model training status: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about the current anomaly model."""
        try:
            is_trained = await self.is_model_trained()
            model_path = Path(self.settings.model_path)
            
            return {
                "is_trained": is_trained,
                "model_path": str(model_path),
                "model_exists": model_path.exists(),
                "resnet_model": self.settings.resnet_model,
                "contamination": self.settings.contamination,
                "n_estimators": self.settings.n_estimators,
                "anomaly_threshold": self.settings.anomaly_threshold,
                "device": self.settings.device
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}


def create_anomaly_service(settings: AnomalySettings) -> AnomalyService:
    """Factory function to create anomaly service."""
    return AnomalyService(settings)
