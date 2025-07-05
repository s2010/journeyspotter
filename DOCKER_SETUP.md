# Docker Setup Guide

Complete containerization for easy deployment across environments.

## Quick Start

```bash
# Build and run
docker build -t journey-spotter .
docker-compose up

# Headless mode (servers)
docker-compose --profile headless up journey-spotter-headless
```

## Key Commands

**Basic Usage:**
```bash
# Help
docker run --rm journey-spotter

# Train model
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  journey-spotter python journey_spotter.py --train

# Detect anomalies
docker run --rm -v $(pwd)/videos:/app/videos -v $(pwd)/output:/app/output \
  journey-spotter python journey_spotter.py --input /app/videos/video.mp4 --detect-anomalies
```

## Volume Mounts

- `./data:/app/data` - Training data
- `./models:/app/models` - Model files
- `./output:/app/output` - Results
- `./videos:/app/videos` - Input videos

## Services

- **journey-spotter** - Full functionality with camera/display
- **journey-spotter-headless** - Server deployment (no camera/display)

## Troubleshooting

**Permissions:**
```bash
sudo chown -R $USER:$USER data models output videos
```

**Camera Access (Linux):**
```bash
sudo usermod -a -G video $USER
```

**Display (Linux):**
```bash
xhost +local:docker
```

## Files

- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration
- `.dockerignore` - Build exclusions

Cross-platform compatible with Docker Desktop on Windows/macOS/Linux. 