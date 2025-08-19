# Multi-stage build for JourneySpotter production deployment
FROM python:3.11-slim

# Install system dependencies for OCR and image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy modular application code
COPY api/ api/
COPY ui/ ui/
COPY core/ core/
COPY adapters/ adapters/
COPY domain/ domain/
COPY config/ config/
COPY .streamlit/ .streamlit/
COPY main.py .

# Copy existing samples directory
COPY samples/ samples/

# Create models directory for anomaly detection
RUN mkdir -p models/

# Create a startup script for different components
RUN echo '#!/bin/bash\n\
case "$1" in\n\
    "api")\n\
        python main.py api --host 0.0.0.0 --port 8000\n\
        ;;\n\
    "ui")\n\
        python main.py ui --port 8501\n\
        ;;\n\
    "streamlit")\n\
        streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0\n\
        ;;\n\
    *)\n\
        python main.py api --host 0.0.0.0 --port 8000\n\
        ;;\n\
esac' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GROQ_API_KEY=""

# Expose ports (FastAPI on 8000, Streamlit on 8501)
EXPOSE 8000 8501

# Default to FastAPI backend using new modular structure
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"] 