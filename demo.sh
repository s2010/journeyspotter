#!/bin/bash

echo "Public Transport Journey Spotter - Docker Demo"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "SUCCESS: Docker is running"

# Build the image if it doesn't exist
if [[ "$(docker images -q journey-spotter 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t journey-spotter .
else
    echo "SUCCESS: Docker image already exists"
fi

echo ""
echo "Available commands:"
echo "1. Show help"
echo "2. Record normal behavior (requires camera)"
echo "3. Train model (requires normal.mp4)"
echo "4. Detect anomalies (requires trained model and input video)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Showing help..."
        docker run --rm journey-spotter
        ;;
    2)
        echo "Recording normal behavior..."
        echo "Note: This requires camera access and may not work in all Docker environments"
        docker run --rm -it \
            --device=/dev/video0:/dev/video0 2>/dev/null || \
            docker run --rm -it \
            -v $(pwd)/data:/app/data \
            journey-spotter python journey_spotter.py --record
        ;;
    3)
        echo "Training model..."
        if [ ! -f "data/normal.mp4" ]; then
            echo "ERROR: normal.mp4 not found in data/ directory"
            echo "Please record normal behavior first or place a normal.mp4 file in the data/ directory"
            exit 1
        fi
        docker run --rm -it \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/models:/app/models \
            journey-spotter python journey_spotter.py --train
        ;;
    4)
        echo "Detecting anomalies..."
        if [ ! -f "models/journey_model.pth" ]; then
            echo "ERROR: Trained model not found"
            echo "Please train the model first"
            exit 1
        fi
        
        echo "Available video files in videos/ directory:"
        ls -la videos/ 2>/dev/null || echo "No videos directory found"
        
        read -p "Enter video filename (or full path): " video_file
        
        if [ ! -f "videos/$video_file" ] && [ ! -f "$video_file" ]; then
            echo "ERROR: Video file not found"
            exit 1
        fi
        
        # Determine the correct path
        if [ -f "videos/$video_file" ]; then
            video_path="/app/videos/$video_file"
        else
            # Copy external file to videos directory
            cp "$video_file" "videos/"
            video_path="/app/videos/$(basename $video_file)"
        fi
        
        docker run --rm -it \
            -v $(pwd)/videos:/app/videos \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/output:/app/output \
            journey-spotter python journey_spotter.py \
            --input "$video_path" \
            --model /app/models/journey_model.pth \
            --output-csv /app/output/anomalies.csv
        
        echo "SUCCESS: Results saved to output/anomalies.csv"
        ;;
    *)
        echo "ERROR: Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Demo completed!"
echo "Check the following directories for results:"
echo "  - data/     : Training videos"
echo "  - models/   : Trained models"
echo "  - output/   : Detection results"
echo "  - videos/   : Input videos" 