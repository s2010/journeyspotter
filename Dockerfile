# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --default-timeout=1000 --retries=5 -r requirements.txt

# Copy application code
COPY journey_spotter.py .
COPY travel_intelligence.py .
COPY ocr_processor.py .
COPY demo_ai_travel.py .
COPY README.md .

# Create directories for data
RUN mkdir -p /app/data /app/models /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Add environment variable for OpenAI API key (to be set at runtime)
ENV OPENAI_API_KEY=""

# Expose port for any future web interface
EXPOSE 8000

# Default command
CMD ["python", "journey_spotter.py", "--help"] 