"""
Unit tests for anomaly detection components.
Tests the ResNet-18 encoder, IsolationForest scorer, and two-stage model.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from sklearn.ensemble import IsolationForest

from config.settings import AnomalySettings
from adapters.anomaly.frame_encoder import ResNetFrameEncoder
from adapters.anomaly.anomaly_scorer import IsolationForestScorer
from adapters.anomaly.anomaly_model import TwoStageAnomalyModel
from core.anomaly_service import AnomalyService


@pytest.fixture
def anomaly_settings():
    """Create test anomaly settings."""
    return AnomalySettings(
        model_path="test_model.joblib",
        resnet_model="resnet18",
        use_pretrained=False,  # Faster for tests
        frame_rate=1.0,
        contamination=0.1,
        n_estimators=10,  # Smaller for faster tests
        random_state=42,
        anomaly_threshold=-0.1,
        max_inference_frames=5,
        device="cpu"
    )


@pytest.fixture
def sample_frames():
    """Create sample video frames for testing."""
    # Create 5 sample RGB frames (64x64)
    frames = []
    for i in range(5):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.random.randn(10, 512).astype(np.float32)


class TestResNetFrameEncoder:
    """Test ResNet-18 frame encoder."""

    def test_init(self, anomaly_settings):
        """Test encoder initialization."""
        encoder = ResNetFrameEncoder(anomaly_settings)
        
        assert encoder.settings == anomaly_settings
        assert encoder.device.type == "cpu"
        assert encoder.model is not None
        assert encoder.transform is not None

    @pytest.mark.asyncio
    async def test_encode_single_frame(self, anomaly_settings, sample_frames):
        """Test encoding a single frame."""
        encoder = ResNetFrameEncoder(anomaly_settings)
        
        embedding = await encoder.encode_single_frame(sample_frames[0])
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == 512  # ResNet-18 feature size

    @pytest.mark.asyncio
    async def test_encode_frames(self, anomaly_settings, sample_frames):
        """Test encoding multiple frames."""
        encoder = ResNetFrameEncoder(anomaly_settings)
        
        embeddings = await encoder.encode_frames(sample_frames)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (5, 512)

    @pytest.mark.asyncio
    async def test_encode_empty_frames(self, anomaly_settings):
        """Test encoding empty frame list."""
        encoder = ResNetFrameEncoder(anomaly_settings)
        
        embeddings = await encoder.encode_frames([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0

    @pytest.mark.asyncio
    async def test_encode_invalid_frame(self, anomaly_settings):
        """Test encoding invalid frame raises error."""
        encoder = ResNetFrameEncoder(anomaly_settings)
        
        # Invalid frame shape
        invalid_frame = np.random.randint(0, 255, (64,), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            await encoder.encode_single_frame(invalid_frame)


class TestIsolationForestScorer:
    """Test IsolationForest anomaly scorer."""

    def test_init(self, anomaly_settings):
        """Test scorer initialization."""
        scorer = IsolationForestScorer(anomaly_settings)
        
        assert scorer.settings == anomaly_settings
        assert scorer.model is None
        assert not scorer._is_fitted

    @pytest.mark.asyncio
    async def test_fit(self, anomaly_settings, sample_embeddings):
        """Test fitting the scorer."""
        scorer = IsolationForestScorer(anomaly_settings)
        
        await scorer.fit(sample_embeddings)
        
        assert scorer.model is not None
        assert scorer._is_fitted
        assert isinstance(scorer.model, IsolationForest)

    @pytest.mark.asyncio
    async def test_fit_empty_embeddings(self, anomaly_settings):
        """Test fitting with empty embeddings raises error."""
        scorer = IsolationForestScorer(anomaly_settings)
        
        with pytest.raises(ValueError):
            await scorer.fit(np.array([]))

    @pytest.mark.asyncio
    async def test_score_samples(self, anomaly_settings, sample_embeddings):
        """Test scoring samples."""
        scorer = IsolationForestScorer(anomaly_settings)
        await scorer.fit(sample_embeddings)
        
        scores = await scorer.score_samples(sample_embeddings[:5])
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    @pytest.mark.asyncio
    async def test_score_unfitted_model(self, anomaly_settings, sample_embeddings):
        """Test scoring with unfitted model raises error."""
        scorer = IsolationForestScorer(anomaly_settings)
        
        with pytest.raises(ValueError):
            await scorer.score_samples(sample_embeddings)

    @pytest.mark.asyncio
    async def test_decision_function(self, anomaly_settings, sample_embeddings):
        """Test decision function."""
        scorer = IsolationForestScorer(anomaly_settings)
        await scorer.fit(sample_embeddings)
        
        decisions = await scorer.decision_function(sample_embeddings[:5])
        
        assert isinstance(decisions, np.ndarray)
        assert decisions.shape == (5,)
        assert np.all(np.isfinite(decisions))


class TestTwoStageAnomalyModel:
    """Test two-stage anomaly model."""

    def test_init(self, anomaly_settings):
        """Test model initialization."""
        model = TwoStageAnomalyModel(anomaly_settings)
        
        assert model.settings == anomaly_settings
        assert model.frame_encoder is not None
        assert model.anomaly_scorer is not None
        assert not model._is_trained

    @pytest.mark.asyncio
    async def test_is_trained_false(self, anomaly_settings):
        """Test is_trained returns False for new model."""
        model = TwoStageAnomalyModel(anomaly_settings)
        
        is_trained = await model.is_trained()
        
        assert not is_trained

    @pytest.mark.asyncio
    async def test_train_nonexistent_video(self, anomaly_settings):
        """Test training with nonexistent video raises error."""
        model = TwoStageAnomalyModel(anomaly_settings)
        
        with pytest.raises(FileNotFoundError):
            await model.train(Path("nonexistent.mp4"))

    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_extract_frames(self, mock_cv2, anomaly_settings):
        """Test frame extraction from video."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # CAP_PROP_FPS
            7: 150    # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        # Mock frame reading
        sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, sample_frame)] * 5 + [(False, None)]
        
        mock_cv2.return_value = mock_cap
        
        model = TwoStageAnomalyModel(anomaly_settings)
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
            frames = await model._extract_frames(Path(tmp_file.name), fps=1.0, max_frames=5)
            
            assert len(frames) == 5
            assert all(isinstance(frame, np.ndarray) for frame in frames)
            assert all(frame.shape == (480, 640, 3) for frame in frames)

    @pytest.mark.asyncio
    async def test_save_load_model(self, anomaly_settings, sample_embeddings):
        """Test model save and load functionality."""
        model = TwoStageAnomalyModel(anomaly_settings)
        
        # Fit the scorer first
        await model.anomaly_scorer.fit(sample_embeddings)
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = Path(tmp_file.name)
        
        try:
            # Save model
            await model.save_model(model_path)
            assert model_path.exists()
            
            # Create new model and load
            new_model = TwoStageAnomalyModel(anomaly_settings)
            await new_model.load_model(model_path)
            
            assert new_model._is_trained
            assert new_model.anomaly_scorer._is_fitted
            
        finally:
            # Cleanup
            if model_path.exists():
                model_path.unlink()


class TestAnomalyService:
    """Test anomaly service."""

    def test_init(self, anomaly_settings):
        """Test service initialization."""
        service = AnomalyService(anomaly_settings)
        
        assert service.settings == anomaly_settings
        assert service.model is not None

    @pytest.mark.asyncio
    async def test_is_model_trained_false(self, anomaly_settings):
        """Test is_model_trained returns False for new service."""
        service = AnomalyService(anomaly_settings)
        
        is_trained = await service.is_model_trained()
        
        assert not is_trained

    @pytest.mark.asyncio
    async def test_get_model_info(self, anomaly_settings):
        """Test getting model information."""
        service = AnomalyService(anomaly_settings)
        
        model_info = await service.get_model_info()
        
        assert isinstance(model_info, dict)
        assert "is_trained" in model_info
        assert "model_path" in model_info
        assert "resnet_model" in model_info
        assert "contamination" in model_info

    @pytest.mark.asyncio
    async def test_detect_anomalies_untrained_model(self, anomaly_settings):
        """Test anomaly detection with untrained model returns None."""
        service = AnomalyService(anomaly_settings)
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
            result = await service.detect_anomalies(Path(tmp_file.name))
            
            assert result is None

    @pytest.mark.asyncio
    async def test_train_model_nonexistent_video(self, anomaly_settings):
        """Test training with nonexistent video returns False."""
        service = AnomalyService(anomaly_settings)
        
        success = await service.train_model(Path("nonexistent.mp4"))
        
        assert not success


@pytest.mark.integration
class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection system."""

    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_full_pipeline(self, mock_cv2, anomaly_settings):
        """Test complete anomaly detection pipeline."""
        # Mock video capture for training
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # CAP_PROP_FPS
            7: 60     # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        # Mock frame reading - return different frames
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frames.append((True, frame))
        frames.append((False, None))
        
        mock_cap.read.side_effect = frames
        mock_cv2.return_value = mock_cap
        
        service = AnomalyService(anomaly_settings)
        
        # Create temporary video files
        with tempfile.NamedTemporaryFile(suffix='.mp4') as train_file, \
             tempfile.NamedTemporaryFile(suffix='.mp4') as test_file:
            
            # Train model
            success = await service.train_model(Path(train_file.name))
            assert success
            
            # Check model is trained
            is_trained = await service.is_model_trained()
            assert is_trained
            
            # Test anomaly detection
            result = await service.detect_anomalies(Path(test_file.name))
            assert result is not None
            
            anomaly_scores, threshold, is_anomalous = result
            assert isinstance(anomaly_scores, list)
            assert isinstance(threshold, float)
            assert isinstance(is_anomalous, list)
            assert len(anomaly_scores) == len(is_anomalous)


if __name__ == "__main__":
    pytest.main([__file__])
