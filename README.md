# Anomaly detector - Video Analysis Tool

Extract information from videos/images using OCR + Llama 3.1 8B model.

## Quick Start

```bash
docker build -t journeyspotter .
GROQ_API_KEY=your_groq_api_key_here docker run -p 8000:8000 -e GROQ_API_KEY journeyspotter
```

## API Usage

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

## Demo UI

```bash
docker-compose up
# Visit: http://localhost:8501 (UI) and http://localhost:8000 (API)
```

## Features

- FastAPI backend with OCR (EasyOCR/Tesseract) + LLM analysis
- Streamlit UI with file upload and sample media
- Llama 3.1 8B model

MIT License