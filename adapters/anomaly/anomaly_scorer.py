"""
IsolationForest anomaly scorer for anomaly detection.
Provides anomaly scoring using scikit-learn's IsolationForest.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from config.settings import AnomalySettings
from domain.ports import AnomalyScorer

logger = logging.getLogger(__name__)


class IsolationForestScorer(AnomalyScorer):
    """IsolationForest-based anomaly scorer."""

    def __init__(self, settings: AnomalySettings) -> None:
        """Initialize IsolationForest scorer."""
        self.settings = settings
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        
        logger.info(f"IsolationForest scorer initialized with contamination={settings.contamination}")

    async def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the IsolationForest on normal embeddings.
        
        Args:
            embeddings: Feature embeddings from normal data (n_samples, n_features)
        """
        if embeddings.size == 0:
            raise ValueError("Cannot fit on empty embeddings")
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
        
        logger.info(f"Fitting IsolationForest on {embeddings.shape[0]} samples with {embeddings.shape[1]} features")
        
        # Create and fit IsolationForest
        self.model = IsolationForest(
            contamination=self.settings.contamination,
            n_estimators=self.settings.n_estimators,
            random_state=self.settings.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(embeddings)
        self._is_fitted = True
        
        logger.info("IsolationForest fitting completed")

    async def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Score embeddings for anomalies using score_samples method.
        
        Args:
            embeddings: Feature embeddings to score (n_samples, n_features)
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before scoring")
        
        if embeddings.size == 0:
            return np.array([])
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
        
        logger.debug(f"Scoring {embeddings.shape[0]} samples")
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(embeddings)
        
        logger.debug(f"Anomaly scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return scores

    async def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute anomaly decision function.
        
        Args:
            embeddings: Feature embeddings to evaluate (n_samples, n_features)
            
        Returns:
            Decision function values (negative = anomalous)
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before computing decision function")
        
        if embeddings.size == 0:
            return np.array([])
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
        
        logger.debug(f"Computing decision function for {embeddings.shape[0]} samples")
        
        # Get decision function values (negative = anomalous)
        decision_scores = self.model.decision_function(embeddings)
        
        logger.debug(f"Decision scores range: [{decision_scores.min():.4f}, {decision_scores.max():.4f}]")
        
        return decision_scores

    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted and self.model is not None


def create_anomaly_scorer(settings: AnomalySettings) -> AnomalyScorer:
    """Factory function to create anomaly scorer."""
    return IsolationForestScorer(settings)
