# Public Transport Journey Spotter

**Self-learning anomaly detection for monitoring videos from trains and buses**

Public Transport Journey Spotter is an intelligent video surveillance tool that uses cutting-edge self-supervised learning to detect unusual behavior in public transport environments. The system learns what "normal" looks like from your own video data and automatically identifies anomalies like crowd surges, dropped objects, loitering, or other suspicious activities without requiring manual labeling.

![Demo](demo.gif)

## Features

- **Self-learning anomaly detection** - No manual labels needed, learns from your normal video data
- **Weak + Self-supervised learning** - Uses iterative training approach for improved accuracy
- **Real-time processing** - Works with live camera feeds or uploaded video files
- **Visual alerts** - Overlays red alert boxes on anomalous frames
- **Comprehensive logging** - Saves all anomalies with timestamps to CSV for audit and review
- **Easy CLI interface** - Simple command-line tools for recording, training, and detection
- **GPU acceleration** - Automatically uses CUDA when available for faster processing
- **Docker support** - Containerized deployment for easy setup and portability

## Installation

### Option 1: Docker (Recommended)

The easiest way to run the Journey Spotter is using Docker. This eliminates dependency issues and works across all platforms.

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system
- Webcam (for recording normal behavior)

#### Quick Start with Docker

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/journeyspotter.git
cd journeyspotter
```

2. **Build the Docker image**
```bash
docker build -t journey-spotter .
```

3. **Run with Docker Compose (Recommended)**
```bash
# For systems with camera and display
docker-compose up journey-spotter

# For headless systems (no camera/display)
docker-compose --profile headless up journey-spotter-headless
```

4. **Or run directly with Docker**
```bash
# Show help
docker run --rm journey-spotter

# Record normal behavior (requires camera access)
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd)/data:/app/data \
  journey-spotter python journey_spotter.py --record

# Train the model
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  journey-spotter python journey_spotter.py --train

# Detect anomalies in video
docker run --rm -it \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  journey-spotter python journey_spotter.py \
  --input /app/videos/your_video.mp4 \
  --model /app/models/journey_model.pth \
  --output-csv /app/output/anomalies.csv
```

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- Webcam (for recording normal behavior)
- CUDA-compatible GPU (optional, for faster training)

#### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/journeyspotter.git
cd journeyspotter
```

2. **Install Python requirements**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python journey_spotter.py --help
```

## How to Use

### Docker Usage

#### Step 1: Record Normal Behavior
```bash
# Using Docker Compose
docker-compose run --rm journey-spotter python journey_spotter.py --record

# Or using Docker directly
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd)/data:/app/data \
  journey-spotter python journey_spotter.py --record
```

#### Step 2: Train the Anomaly Detector
```bash
# Using Docker Compose
docker-compose run --rm journey-spotter python journey_spotter.py --train

# Or using Docker directly
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  journey-spotter python journey_spotter.py --train
```

#### Step 3: Detect Anomalies
```bash
# Place your video file in the videos/ directory first
cp your_video.mp4 videos/

# Using Docker Compose
docker-compose run --rm journey-spotter python journey_spotter.py \
  --input /app/videos/your_video.mp4 \
  --model /app/models/journey_model.pth \
  --output-csv /app/output/anomalies.csv

# Or using Docker directly
docker run --rm -it \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  journey-spotter python journey_spotter.py \
  --input /app/videos/your_video.mp4 \
  --model /app/models/journey_model.pth \
  --output-csv /app/output/anomalies.csv
```

### Local Usage

#### Step 1: Record Normal Behavior
First, capture 10-15 seconds of normal activity in your transport environment:

```bash
python journey_spotter.py --record
```

This will:
- Open your webcam
- Record 12 seconds of normal behavior (adjustable with `--duration`)
- Save the video as `normal.mp4`
- Display a live preview during recording

#### Step 2: Train the Anomaly Detector
Train the self-learning model on your normal behavior video:

```bash
python journey_spotter.py --train
```

This will:
- Extract frames from `normal.mp4`
- Train the weak classifier using motion-based pseudo-labels
- Train the self-supervised classifier for refined detection
- Save the trained model as `journey_model.pth`

#### Step 3: Detect Anomalies
Run anomaly detection on new video footage:

```bash
python journey_spotter.py --input bus_video.mp4 --model journey_model.pth --output-csv anomalies.csv
```

This will:
- Process each frame of the input video
- Overlay red alert boxes on anomalous frames
- Log all detections with timestamps to CSV
- Display real-time results with anomaly scores
- Print summary statistics at the end

### Advanced Usage

**Custom recording duration:**
```bash
# Local
python journey_spotter.py --record --duration 20

# Docker
docker-compose run --rm journey-spotter python journey_spotter.py --record --duration 20
```

**Use different model file:**
```bash
# Local
python journey_spotter.py --train --model my_custom_model.pth

# Docker
docker-compose run --rm journey-spotter python journey_spotter.py --train --model /app/models/my_custom_model.pth
```

**Process live camera feed:**
```bash
# Local
python journey_spotter.py --input 0 --model journey_model.pth

# Docker (requires camera access)
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  journey-spotter python journey_spotter.py \
  --input 0 \
  --model /app/models/journey_model.pth
```

## Docker Architecture

The Docker setup includes:

- **Main service**: Full functionality with camera and display support
- **Headless service**: For server deployments without camera/display
- **Volume mounts**: Persistent storage for models, data, and output
- **Device access**: Camera access for recording and live detection

### Directory Structure
```
journeyspotter/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── journey_spotter.py       # Main application
├── requirements.txt         # Python dependencies
├── data/                   # Training data (mounted volume)
├── models/                 # Trained models (mounted volume)
├── output/                 # Detection results (mounted volume)
└── videos/                 # Input videos (mounted volume)
```

## Understanding the Output

### Console Output
```
Total anomalies detected: 23

Top 5 anomaly timestamps:
1. Time: 45.67s, Frame: 1370, Score: 0.8934
2. Time: 23.12s, Frame: 694, Score: 0.8721
3. Time: 67.89s, Frame: 2037, Score: 0.8456
4. Time: 12.34s, Frame: 370, Score: 0.8123
5. Time: 89.45s, Frame: 2683, Score: 0.7892
```

### CSV Output Format
The `anomalies.csv` file contains detailed frame-by-frame analysis:

| timestamp | frame_number | anomaly_score |
|-----------|--------------|---------------|
| 0.00      | 0            | 0.1234        |
| 0.03      | 1            | 0.1456        |
| 0.07      | 2            | 0.8934        |
| ...       | ...          | ...           |

**Anomaly Score Interpretation:**
- 0.0 - 0.5: Normal behavior
- 0.5 - 0.7: Suspicious activity
- 0.7 - 1.0: High probability anomaly

## Technical Details

### Architecture
The system uses a two-stage learning approach:

1. **Weak Classifier**: Initial training using motion-based pseudo-labels
2. **Self-Supervised Classifier**: Refined training using weak classifier predictions

### Neural Network
- **Backbone**: ResNet-18 with ImageNet pre-training
- **Input**: 224x224 RGB frames
- **Output**: Binary classification (Normal/Anomaly) with confidence scores

### Training Process
1. Extract frames from normal behavior video
2. Generate pseudo-labels using motion analysis
3. Train weak classifier on pseudo-labeled data
4. Use weak classifier predictions to train self-supervised model
5. Save combined model for inference

## Future Development Ideas

### 1. Voice Alert System
Add audio notifications when anomalies are detected:
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Anomaly detected in sector 3")
```

### 2. Real-time Dashboard
Create a web dashboard to visualize anomaly heatmaps and statistics:
- Live video feed with overlay
- Anomaly frequency charts
- Historical trend analysis
- Multi-camera support

### 3. Behavior-Specific Models
Fine-tune models for specific types of incidents:
- **Theft Detection**: Focus on rapid object movement
- **Medical Emergency**: Detect people falling or crowd gathering
- **Vandalism**: Identify destructive behavior patterns
- **Overcrowding**: Monitor passenger density levels

### 4. Integration Features
- **CCTV Integration**: Connect to existing security camera systems
- **Alert Systems**: Send notifications via email, SMS, or Slack
- **Database Logging**: Store incidents in PostgreSQL or MongoDB
- **Mobile App**: Remote monitoring and alert management

## Architecture Overview

```
Input Video → Frame Extraction → Preprocessing → Neural Network → Anomaly Detection → Alert System
                                                      ↓
Normal Video → Training Pipeline → Weak Classifier → Self-Supervised Model → Saved Model
```

## Troubleshooting

### Docker Issues

**Camera not accessible:**
```bash
# On Linux, ensure your user is in the video group
sudo usermod -a -G video $USER

# On macOS, camera access in Docker requires additional setup
# Consider using the local installation for camera recording
```

**Permission issues:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER data models output videos
```

**Display issues (Linux):**
```bash
# Allow X11 forwarding
xhost +local:docker
```

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Dataset contributions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Based on the research paper and implementation:
**"Human Activity Analysis: Iterative Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events"** by DegardinBruno/human-self-learning-anomaly

Original repository: https://github.com/DegardinBruno/human-self-learning-anomaly

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check our Wiki for detailed guides
- **Community**: Join our Discord for discussions and help

---

**Important Note**: This tool is designed to assist human operators and should not be used as the sole method for security monitoring. Always have trained personnel review flagged incidents. 