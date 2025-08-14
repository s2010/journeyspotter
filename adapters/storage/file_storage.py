"""
File storage adapter implementation for temporary file handling.
Provides secure file storage and cleanup capabilities.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from config.settings import AppSettings
from core.interfaces import FileStorage

logger = logging.getLogger(__name__)


class LocalFileStorage(FileStorage):
    """Local file system implementation for file storage."""

    def __init__(self, settings: AppSettings) -> None:
        """Initialize file storage."""
        self.settings = settings
        self.temp_dir = settings.temp_dir
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"File storage initialized with temp dir: {self.temp_dir}")

    async def save_temp_file(self, content: bytes, filename: str) -> Path:
        """Save uploaded file temporarily."""
        if not filename.strip():
            raise ValueError("Filename cannot be empty")
        
        if len(content) > self.settings.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed: {len(content)} > {self.settings.max_file_size}")

        try:
            # Create secure temporary file
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=self.temp_dir
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            
            logger.debug(f"Saved temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {e}")
            raise

    async def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


def create_file_storage(settings: AppSettings) -> FileStorage:
    """Factory function to create file storage."""
    return LocalFileStorage(settings)
