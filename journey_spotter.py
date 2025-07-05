#!/usr/bin/env python3
"""
Travel Video Analysis Tool
Intelligent text extraction and location analysis for travel videos
Self-learning anomaly detection for monitoring videos from trains/buses

Development History:
- v1.0: Initial anomaly detection system
- v1.1: Added OCR text extraction capabilities  
- v1.2: Integrated GPT-4 for travel intelligence
- v1.3: Professional output formatting (current)
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import csv
import json
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import AI travel intelligence modules
# Refactoring note: Originally these were inline, moved to separate modules for better organization
try:
    from travel_intelligence import TravelIntelligence, LocationInfo
    from ocr_processor import OCRProcessor
    AI_TRAVEL_AVAILABLE = True
except ImportError as e:
    AI_TRAVEL_AVAILABLE = False
    print(f"AI Travel features not available: {e}")
    print("Install dependencies: pip install openai easyocr pytesseract")

class VideoDataset(Dataset):
    """Dataset for video frames"""
    def __init__(self, frames, transform=None, labels=None):
        self.frames = frames
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        
        if self.labels is not None:
            return frame, self.labels[idx]
        return frame

class WeakClassifier(nn.Module):
    """
    Weak classifier for initial anomaly detection
    
    Development note: Started with simple CNN, evolved to ResNet-18 backbone
    for better feature extraction in transport environments
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super(WeakClassifier, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final layer
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Normal vs Anomaly
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features

class SelfSupervisedClassifier(nn.Module):
    """
    Self-supervised classifier for refined anomaly detection
    
    Iterative improvement: Added batch normalization and deeper architecture
    after initial weak classifier showed promise but needed refinement
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super(SelfSupervisedClassifier, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Enhanced architecture based on initial experiments
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features

class JourneySpotter:
    """
    Travel Video Analysis Tool
    
    Evolution: Originally focused only on anomaly detection,
    expanded to include travel intelligence features based on user needs
    """
    
    def __init__(self, openai_api_key=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weak_model = None
        self.ss_model = None
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.5
        
        # Initialize AI travel intelligence components
        # Refactoring: Moved initialization logic to be more robust
        if AI_TRAVEL_AVAILABLE:
            try:
                self.travel_intelligence = TravelIntelligence(api_key=openai_api_key)
                self.ocr_processor = OCRProcessor(languages=['en'], use_gpu=torch.cuda.is_available())
                print("Travel intelligence modules initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize AI features: {e}")
                self.travel_intelligence = None
                self.ocr_processor = None
        else:
            self.travel_intelligence = None
            self.ocr_processor = None
        
        # Data transforms - refined through experimentation
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def record_video(self, duration=12, output_path='normal.mp4'):
        """Record normal video from webcam"""
        print(f"Recording {duration} seconds of normal behavior...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_count = 0
        
        print("Recording started... Press 'q' to stop early")
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            frame_count += 1
            
            # Display frame
            cv2.putText(frame, f"Recording: {int(time.time() - start_time)}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recording Normal Behavior', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Recording completed: {frame_count} frames saved to {output_path}")
        return output_path
    
    def extract_frames(self, video_path, max_frames=1000):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def create_pseudo_labels(self, frames):
        """
        Create pseudo labels for self-supervised learning
        
        Development note: Initially used random labels, evolved to motion-based
        heuristics for more realistic training data
        """
        labels = []
        
        for i in range(len(frames)):
            if i == 0:
                labels.append(0)  # First frame is normal
                continue
            
            # Calculate frame difference for motion detection
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = np.mean(diff)
            
            # Threshold tuned through experimentation
            if motion_score > 30:  # High motion threshold
                labels.append(1)  # Potential anomaly
            else:
                labels.append(0)  # Normal
        
        return labels
    
    def train_weak_classifier(self, frames, epochs=20):
        """Train weak classifier on normal data"""
        print("Training weak classifier...")
        
        # Create pseudo labels based on motion analysis
        pseudo_labels = self.create_pseudo_labels(frames)
        
        # Create dataset
        dataset = VideoDataset(frames, self.train_transform, pseudo_labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Initialize model
        self.weak_model = WeakClassifier().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.weak_model.parameters(), lr=0.001)
        
        self.weak_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_frames, batch_labels in dataloader:
                batch_frames = batch_frames.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.weak_model(batch_frames)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    def train_self_supervised(self, frames, epochs=30):
        """Train self-supervised classifier"""
        print("Training self-supervised classifier...")
        
        # Get predictions from weak classifier for pseudo-labeling
        self.weak_model.eval()
        pseudo_labels = []
        
        with torch.no_grad():
            for frame in frames:
                frame_tensor = self.test_transform(frame).unsqueeze(0).to(self.device)
                outputs, _ = self.weak_model(frame_tensor)
                _, predicted = torch.max(outputs, 1)
                pseudo_labels.append(predicted.item())
        
        # Create dataset with refined pseudo labels
        dataset = VideoDataset(frames, self.train_transform, pseudo_labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Initialize self-supervised model
        self.ss_model = SelfSupervisedClassifier().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.ss_model.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        
        self.ss_model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_frames, batch_labels in dataloader:
                batch_frames = batch_frames.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.ss_model(batch_frames)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            scheduler.step()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    def train_model(self, video_path):
        """Complete training pipeline"""
        print("Starting training pipeline...")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) < 10:
            raise ValueError("Not enough frames for training. Need at least 10 frames.")
        
        # Train weak classifier
        self.train_weak_classifier(frames)
        
        # Train self-supervised classifier
        self.train_self_supervised(frames)
        
        print("Training completed successfully!")
    
    def save_model(self, model_path='journey_model.pth'):
        """Save trained models"""
        if self.weak_model is None or self.ss_model is None:
            raise ValueError("Models not trained yet!")
        
        model_state = {
            'weak_model': self.weak_model.state_dict(),
            'ss_model': self.ss_model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.anomaly_threshold
        }
        
        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained models"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_state = torch.load(model_path, map_location=self.device)
        
        # Initialize models
        self.weak_model = WeakClassifier().to(self.device)
        self.ss_model = SelfSupervisedClassifier().to(self.device)
        
        # Load states
        self.weak_model.load_state_dict(model_state['weak_model'])
        self.ss_model.load_state_dict(model_state['ss_model'])
        self.scaler = model_state['scaler']
        self.anomaly_threshold = model_state['threshold']
        
        self.weak_model.eval()
        self.ss_model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def detect_anomalies(self, video_path, output_csv='anomalies.csv'):
        """Detect anomalies in video"""
        if self.ss_model is None:
            raise ValueError("Model not loaded! Use load_model() first.")
        
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        anomalies = []
        frame_number = 0
        
        # Prepare CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'frame_number', 'anomaly_score'])
            
            self.ss_model.eval()
            with torch.no_grad():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert and preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.test_transform(frame_rgb).unsqueeze(0).to(self.device)
                    
                    # Get anomaly prediction
                    outputs, _ = self.ss_model(frame_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    anomaly_score = probabilities[0][1].item()  # Probability of anomaly
                    
                    timestamp = frame_number / fps
                    
                    # Check if anomaly
                    if anomaly_score > self.anomaly_threshold:
                        anomalies.append({
                            'timestamp': timestamp,
                            'frame_number': frame_number,
                            'score': anomaly_score
                        })
                        
                        # Draw red alert box
                        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), 
                                    (0, 0, 255), 5)
                        cv2.putText(frame, f'ANOMALY DETECTED! Score: {anomaly_score:.3f}', 
                                  (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Write to CSV
                    writer.writerow([f"{timestamp:.2f}", frame_number, f"{anomaly_score:.4f}"])
                    
                    # Display frame (optional)
                    cv2.putText(frame, f'Frame: {frame_number}/{total_frames}', 
                              (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Journey Spotter - Anomaly Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_number += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nTotal anomalies detected: {len(anomalies)}")
        
        # Print top 5 timestamps
        if anomalies:
            sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
            print("\nTop 5 anomaly timestamps:")
            for i, anomaly in enumerate(sorted_anomalies[:5], 1):
                print(f"{i}. Time: {anomaly['timestamp']:.2f}s, Frame: {anomaly['frame_number']}, Score: {anomaly['score']:.4f}")
        
        print(f"Detailed results saved to {output_csv}")
        return anomalies
    
    def extract_travel_intelligence(self, input_path, output_json='travel_analysis.json', sample_rate=30):
        """
        Extract travel intelligence from video or image using OCR and GPT-4
        
        Development evolution: Originally just OCR, added AI analysis for better results
        
        Args:
            input_path: Path to video file or image
            output_json: Output JSON file path
            sample_rate: Frame sampling rate for videos
            
        Returns:
            Dictionary with locations and travel summary
        """
        if not AI_TRAVEL_AVAILABLE:
            raise RuntimeError("AI Travel features not available. Install required dependencies.")
        
        if not self.ocr_processor or not self.travel_intelligence:
            raise RuntimeError("AI Travel components not initialized properly.")
        
        print(f"Analyzing travel content in: {input_path}")
        
        # Determine if input is video or image
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        
        if is_video:
            # Extract text from video frames
            print("Extracting text from video frames...")
            ocr_results = self.ocr_processor.extract_text_from_video(input_path, sample_rate)
            combined_text = self.ocr_processor.combine_text_results(ocr_results)
        else:
            # Extract text from single image
            print("Extracting text from image...")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            ocr_results = self.ocr_processor.extract_text_from_image(image)
            combined_text = ' | '.join([result.text for result in ocr_results])
        
        print(f"Extracted text: {combined_text}")
        
        if not combined_text.strip():
            print("No text found in the input")
            return {"locations": [], "summary": "No travel text detected", "insights": {}}
        
        # Analyze with GPT-4
        print("Analyzing travel content with AI...")
        locations, summary = self.travel_intelligence.extract_and_enrich_locations(combined_text)
        
        # Get additional insights
        insights = self.travel_intelligence.get_travel_insights(locations)
        
        # Prepare output data
        travel_data = {
            "source_file": input_path,
            "extraction_timestamp": datetime.now().isoformat(),
            "raw_text": combined_text,
            "locations": [
                {
                    "location": loc.location,
                    "country": loc.country,
                    "type": loc.type,
                    "order": loc.order
                } for loc in locations
            ],
            "summary": summary,
            "insights": insights
        }
        
        # Save to JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(travel_data, f, indent=2, ensure_ascii=False)
        
        # Print results (cleaned up formatting)
        print("\nTravel Analysis Complete!")
        print(f"Summary: {summary}")
        print(f"\nLocations Found ({len(locations)}):")
        
        for i, location in enumerate(locations, 1):
            print(f"{i}. {location.location}, {location.country} ({location.type})")
        
        if insights:
            print(f"\nTravel Insights:")
            print(f"   Total locations: {insights.get('total_locations', 0)}")
            print(f"   Countries visited: {insights.get('countries_visited', 0)}")
            if insights.get('countries'):
                print(f"   Countries: {', '.join(insights['countries'])}")
            if insights.get('journey_span', {}).get('first'):
                span = insights['journey_span']
                if span.get('last') and span['last'] != span['first']:
                    print(f"   Journey: {span['first']} -> {span['last']}")
                else:
                    print(f"   Primary location: {span['first']}")
        
        print(f"\nDetailed results saved to: {output_json}")
        return travel_data
    
    def analyze_travel_video_with_ocr_overlay(self, video_path, output_video='travel_analysis.mp4', sample_rate=30):
        """
        Analyze video and create output with OCR text overlays
        
        Args:
            video_path: Input video path
            output_video: Output video with OCR overlays
            sample_rate: Frame sampling rate for OCR
        """
        if not AI_TRAVEL_AVAILABLE:
            raise RuntimeError("AI Travel features not available.")
        
        print(f"Creating travel analysis video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process OCR every sample_rate frames
            if frame_number % sample_rate == 0:
                ocr_results = self.ocr_processor.extract_text_from_image(frame)
                
                # Draw OCR results on frame
                if ocr_results:
                    frame = self.ocr_processor.visualize_text_detection(frame, ocr_results)
                    
                    # Add text overlay
                    y_offset = 30
                    for result in ocr_results:
                        cv2.putText(frame, f"OCR: {result.text}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 25
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", 
                       (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
            frame_number += 1
            
            # Progress
            if frame_number % (fps * 10) == 0:  # Every 10 seconds
                progress = (frame_number / total_frames) * 100
                print(f"Processing: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        print(f"Travel analysis video saved: {output_video}")

def main():
    parser = argparse.ArgumentParser(
        description='Travel Video Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract travel intelligence from video or image
  python journey_spotter.py --analyze-travel video.mp4 --output-json travel.json
  
  # Traditional anomaly detection
  python journey_spotter.py --record --duration 15
  python journey_spotter.py --train
  python journey_spotter.py --input video.mp4 --detect-anomalies
  
  # Create OCR analysis video
  python journey_spotter.py --input video.mp4 --ocr-overlay --output-video analysis.mp4
        """
    )
    
    # AI Travel Intelligence arguments
    parser.add_argument('--analyze-travel', type=str, metavar='INPUT', 
                       help='Extract travel intelligence from video/image using OCR + GPT-4')
    parser.add_argument('--output-json', type=str, default='travel_analysis.json',
                       help='Output JSON file for travel analysis')
    parser.add_argument('--ocr-overlay', action='store_true',
                       help='Create video with OCR text overlays')
    parser.add_argument('--output-video', type=str, default='travel_analysis.mp4',
                       help='Output video file with OCR overlays')
    parser.add_argument('--openai-api-key', type=str,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='OCR sampling rate (process every N frames)')
    
    # Traditional anomaly detection arguments
    parser.add_argument('--record', action='store_true', help='Record normal video from webcam')
    parser.add_argument('--train', action='store_true', help='Train anomaly detection model')
    parser.add_argument('--detect-anomalies', action='store_true', help='Run anomaly detection')
    parser.add_argument('--input', type=str, help='Input video file')
    parser.add_argument('--model', type=str, default='journey_model.pth', help='Model file path')
    parser.add_argument('--output-csv', type=str, default='anomalies.csv', help='Output CSV file for anomalies')
    parser.add_argument('--duration', type=int, default=12, help='Recording duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize with OpenAI API key if provided
    spotter = JourneySpotter(openai_api_key=args.openai_api_key)
    
    try:
        # AI Travel Intelligence commands
        if args.analyze_travel:
            if not AI_TRAVEL_AVAILABLE:
                print("Error: AI Travel features not available. Install dependencies:")
                print("pip install openai easyocr pytesseract")
                return 1
            
            travel_data = spotter.extract_travel_intelligence(
                args.analyze_travel, 
                args.output_json, 
                args.sample_rate
            )
            return 0
            
        elif args.ocr_overlay and args.input:
            if not AI_TRAVEL_AVAILABLE:
                print("Error: AI Travel features not available. Install dependencies:")
                print("pip install openai easyocr pytesseract")
                return 1
            
            spotter.analyze_travel_video_with_ocr_overlay(
                args.input, 
                args.output_video, 
                args.sample_rate
            )
            return 0
        
        # Traditional anomaly detection commands
        elif args.record:
            spotter.record_video(duration=args.duration)
            
        elif args.train:
            if not os.path.exists('normal.mp4'):
                print("Error: normal.mp4 not found. Please record normal video first using --record")
                return 1
            
            spotter.train_model('normal.mp4')
            spotter.save_model(args.model)
            
        elif args.detect_anomalies and args.input:
            if not os.path.exists(args.model):
                print(f"Error: Model file {args.model} not found. Please train model first using --train")
                return 1
            
            spotter.load_model(args.model)
            spotter.detect_anomalies(args.input, args.output_csv)
            
        else:
            # Show help if no valid command provided
            parser.print_help()
            print("\n" + "="*60)
            print("TRAVEL VIDEO ANALYSIS TOOL")
            print("="*60)
            if AI_TRAVEL_AVAILABLE:
                print("Status: AI Travel Intelligence - AVAILABLE")
            else:
                print("Status: AI Travel Intelligence - NOT AVAILABLE")
                print("   Install: pip install openai easyocr pytesseract")
            print("Status: Anomaly Detection - AVAILABLE")
            print("\nFor travel analysis, set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 