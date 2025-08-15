"""
Domain ports and interfaces for anomaly detection.
Defines contracts for anomaly detection components following SOLID principles.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np


class AnomalyModel(ABC):
    """Abstract base class for anomaly detection models."""

    @abstractmethod
    async def train(self, normal_video_path: Path) -> None:
        """
        Train the anomaly detection model on normal video data.
        
        Args:
            normal_video_path: Path to video file containing normal behavior
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def is_trained(self) -> bool:
        """Check if the model has been trained and is ready for inference."""
        pass

    @abstractmethod
    async def save_model(self, model_path: Path) -> None:
        """Save the trained model to disk."""
        pass

    @abstractmethod
    async def load_model(self, model_path: Path) -> None:
        """Load a trained model from disk."""
        pass


class FrameEncoder(ABC):
    """Abstract base class for frame encoding (e.g., ResNet-18)."""

    @abstractmethod
    async def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Encode video frames into feature embeddings.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Feature embeddings as numpy array of shape (n_frames, n_features)
        """
        pass

    @abstractmethod
    async def encode_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame into feature embedding.
        
        Args:
            frame: Single video frame as numpy array
            
        Returns:
            Feature embedding as numpy array of shape (n_features,)
        """
        pass


class AnomalyScorer(ABC):
    """Abstract base class for anomaly scoring (e.g., IsolationForest)."""

    @abstractmethod
    async def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the anomaly scorer on normal embeddings.
        
        Args:
            embeddings: Feature embeddings from normal data
        """
        pass

    @abstractmethod
    async def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Score embeddings for anomalies.
        
        Args:
            embeddings: Feature embeddings to score
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        pass

    @abstractmethod
    async def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute anomaly decision function.
        
        Args:
            embeddings: Feature embeddings to evaluate
            
        Returns:
            Decision function values (negative = anomalous)
        """
        pass
