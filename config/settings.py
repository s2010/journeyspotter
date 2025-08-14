"""
Configuration management using Pydantic Settings.
Centralizes all application configuration with environment variable support.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class OCRSettings(BaseSettings):
    """OCR-specific configuration."""
    
    languages: List[str] = Field(default=["en"], description="OCR languages to support")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration for EasyOCR")
    tesseract_cmd: Optional[str] = Field(default=None, description="Path to tesseract executable")
    
    class Config:
        env_prefix = "OCR_"


class LLMSettings(BaseSettings):
    """LLM-specific configuration."""
    
    api_key: str = Field(..., description="Groq API key")
    model: str = Field(default="llama3-8b-8192", description="Groq model to use")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens for LLM response")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("max_tokens")
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v
    
    class Config:
        env_prefix = "GROQ_"


class VideoSettings(BaseSettings):
    """Video processing configuration."""
    
    max_frames: int = Field(default=10, description="Maximum frames to extract from video")
    frame_sampling: str = Field(default="uniform", description="Frame sampling strategy")
    
    @validator("max_frames")
    def validate_max_frames(cls, v: int) -> int:
        """Validate max_frames is positive."""
        if v <= 0:
            raise ValueError("max_frames must be positive")
        return v
    
    class Config:
        env_prefix = "VIDEO_"


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    class Config:
        env_prefix = "API_"


class UISettings(BaseSettings):
    """Streamlit UI configuration."""
    
    api_base_url: str = Field(default="http://localhost:8000", description="API base URL")
    port: int = Field(default=8501, description="Streamlit port")
    
    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    class Config:
        env_prefix = "UI_"


class AppSettings(BaseSettings):
    """Main application settings."""
    
    # Sub-configurations
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    api: APISettings = Field(default_factory=APISettings)
    ui: UISettings = Field(default_factory=UISettings)
    
    # General settings
    app_name: str = Field(default="JourneySpotter", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # File handling
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Maximum file size in bytes (50MB)")
    temp_dir: Path = Field(default=Path("/tmp"), description="Temporary directory for file processing")
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("max_file_size")
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max file size is positive."""
        if v <= 0:
            raise ValueError("max_file_size must be positive")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> AppSettings:
    """Get application settings instance."""
    return AppSettings()
