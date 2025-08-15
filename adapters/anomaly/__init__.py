"""
Anomaly detection adapters.
Provides concrete implementations for anomaly detection components.
"""

from .frame_encoder import ResNetFrameEncoder
from .anomaly_scorer import IsolationForestScorer
from .anomaly_model import TwoStageAnomalyModel

__all__ = [
    "ResNetFrameEncoder",
    "IsolationForestScorer", 
    "TwoStageAnomalyModel"
]
