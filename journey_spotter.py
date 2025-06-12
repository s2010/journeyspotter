#!/usr/bin/env python3
"""
Public Transport Journey Spotter
Self-learning anomaly detection for monitoring videos from trains/buses
Based on DegardinBruno/human-self-learning-anomaly
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
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

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
    """Weak classifier for initial anomaly detection"""
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
    """Self-supervised classifier for refined anomaly detection"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super(SelfSupervisedClassifier, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
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
    """Main class for Public Transport Journey Spotter"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weak_model = None
        self.ss_model = None
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.5
        
        # Data transforms
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
        """Create pseudo labels for self-supervised learning"""
        # Simple heuristic: frames with high motion/change are potential anomalies
        labels = []
        
        for i in range(len(frames)):
            if i == 0:
                labels.append(0)  # First frame is normal
                continue
            
            # Calculate frame difference
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = np.mean(diff)
            
            # Threshold for anomaly detection (tunable)
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

def main():
    parser = argparse.ArgumentParser(description='Public Transport Journey Spotter')
    parser.add_argument('--record', action='store_true', help='Record normal video from webcam')
    parser.add_argument('--train', action='store_true', help='Train anomaly detection model')
    parser.add_argument('--input', type=str, help='Input video file for anomaly detection')
    parser.add_argument('--model', type=str, default='journey_model.pth', help='Model file path')
    parser.add_argument('--output-csv', type=str, default='anomalies.csv', help='Output CSV file for anomalies')
    parser.add_argument('--duration', type=int, default=12, help='Recording duration in seconds')
    
    args = parser.parse_args()
    
    spotter = JourneySpotter()
    
    try:
        if args.record:
            spotter.record_video(duration=args.duration)
            
        elif args.train:
            if not os.path.exists('normal.mp4'):
                print("Error: normal.mp4 not found. Please record normal video first using --record")
                return
            
            spotter.train_model('normal.mp4')
            spotter.save_model(args.model)
            
        elif args.input:
            if not os.path.exists(args.model):
                print(f"Error: Model file {args.model} not found. Please train model first using --train")
                return
            
            spotter.load_model(args.model)
            spotter.detect_anomalies(args.input, args.output_csv)
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 