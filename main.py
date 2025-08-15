"""
Main application entry point for JourneySpotter.
Provides a unified interface to run different components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from config.settings import get_settings


def setup_logging(log_level: str) -> None:
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str) -> None:
    """JourneySpotter - Video/image analysis using OCR + intelligent processing."""
    setup_logging(log_level)


@cli.command()
@click.option("--host", default=None, help="API host")
@click.option("--port", default=None, type=int, help="API port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api(host: Optional[str], port: Optional[int], reload: bool) -> None:
    """Start the FastAPI server."""
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload or settings.api.debug
    )


@cli.command()
@click.option("--port", default=None, type=int, help="Streamlit port")
def ui(port: Optional[int]) -> None:
    """Start the Streamlit UI."""
    import subprocess
    
    settings = get_settings()
    ui_port = port or settings.ui.port
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", str(ui_port),
        "--server.address", "0.0.0.0"
    ]
    
    subprocess.run(cmd)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file")
def analyze(file_path: Path, output: Optional[Path]) -> None:
    """Analyze a single file via CLI."""
    import asyncio
    import json
    
    from adapters.llm.processor import create_llm_processor
    from adapters.ocr.processor import create_ocr_processor
    from adapters.video.processor import create_video_processor
    from core.analysis_service import create_analysis_service
    from domain.models import AnalysisRequest, MediaType
    
    async def run_analysis() -> None:
        settings = get_settings()
        
        # Create services
        ocr_processor = create_ocr_processor(settings.ocr)
        llm_processor = create_llm_processor(settings.llm)
        video_processor = create_video_processor(settings.video)
        analysis_service = create_analysis_service(
            ocr_processor, llm_processor, video_processor, settings
        )
        
        # Determine media type
        file_ext = file_path.suffix.lower()
        if file_ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            media_type = MediaType.IMAGE
        elif file_ext in {".mp4", ".avi", ".mov", ".mkv"}:
            media_type = MediaType.VIDEO
        else:
            click.echo(f"Unsupported file type: {file_ext}")
            return
        
        # Create request
        request = AnalysisRequest(
            file_path=file_path,
            media_type=media_type,
            filename=file_path.name
        )
        
        # Perform analysis
        click.echo(f"Analyzing {file_path}...")
        result = await analysis_service.analyze_media(request)
        
        # Prepare output
        output_data = {
            "filename": result.filename,
            "media_type": result.media_type.value,
            "summary": result.summary,
            "confidence": result.confidence,
            "extracted_text": result.extracted_text,
            "locations": [
                {
                    "name": loc.name,
                    "country": loc.country,
                    "type": loc.location_type.value,
                    "confidence": loc.confidence
                }
                for loc in result.locations
            ]
        }
        
        # Output results
        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2))
    
    asyncio.run(run_analysis())


if __name__ == "__main__":
    cli()
