"""
ResNet-18 frame encoder for anomaly detection.
Provides feature extraction from video frames using pretrained ResNet-18.
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config.settings import AnomalySettings
from domain.ports import FrameEncoder

logger = logging.getLogger(__name__)


class ResNetFrameEncoder(FrameEncoder):
    """ResNet-18 based frame encoder for feature extraction."""

    def __init__(self, settings: AnomalySettings) -> None:
        """Initialize ResNet frame encoder."""
        self.settings = settings
        self.device = torch.device(settings.device)
        
        # Load pretrained ResNet-18 and remove classifier
        if settings.resnet_model == "resnet18":
            self.model = models.resnet18(pretrained=settings.use_pretrained)
        elif settings.resnet_model == "resnet34":
            self.model = models.resnet34(pretrained=settings.use_pretrained)
        elif settings.resnet_model == "resnet50":
            self.model = models.resnet50(pretrained=settings.use_pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {settings.resnet_model}")
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"ResNet frame encoder initialized with {settings.resnet_model} on {settings.device}")

    async def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Encode multiple video frames into feature embeddings.
        
        Args:
            frames: List of video frames as numpy arrays (H, W, C)
            
        Returns:
            Feature embeddings as numpy array of shape (n_frames, n_features)
        """
        if not frames:
            return np.array([])
        
        embeddings = []
        
        with torch.no_grad():
            for frame in frames:
                try:
                    embedding = await self.encode_single_frame(frame)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to encode frame: {e}")
                    continue
        
        if not embeddings:
            logger.error("No frames could be encoded")
            return np.array([])
        
        embeddings_array = np.vstack(embeddings)
        logger.debug(f"Encoded {len(frames)} frames to embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array

    async def encode_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame into feature embedding.
        
        Args:
            frame: Single video frame as numpy array (H, W, C)
            
        Returns:
            Feature embedding as numpy array of shape (n_features,)
        """
        try:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Handle different channel orders
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB format
                pil_image = Image.fromarray(frame, mode='RGB')
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Grayscale with channel dimension
                pil_image = Image.fromarray(frame.squeeze(), mode='L').convert('RGB')
            elif len(frame.shape) == 2:
                # Grayscale without channel dimension
                pil_image = Image.fromarray(frame, mode='L').convert('RGB')
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            
            # Apply preprocessing transforms
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features using ResNet
            with torch.no_grad():
                features = self.model(tensor)
                # Flatten the features (remove spatial dimensions)
                features = features.view(features.size(0), -1)
                # Convert to numpy
                embedding = features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode single frame: {e}")
            raise


def create_frame_encoder(settings: AnomalySettings) -> FrameEncoder:
    """Factory function to create frame encoder."""
    return ResNetFrameEncoder(settings)
