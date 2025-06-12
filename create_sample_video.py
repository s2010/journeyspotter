#!/usr/bin/env python3
"""
Create a sample normal behavior video for testing the Journey Spotter
"""
import cv2
import numpy as np
import os

def create_sample_video():
    """Create a simple test video with normal behavior patterns"""
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 12  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = '/app/data/normal.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f'Creating sample normal behavior video...')
    print(f'Output: {output_path}')
    print(f'Duration: {duration} seconds')
    print(f'Resolution: {width}x{height}')
    print(f'FPS: {fps}')
    
    for i in range(total_frames):
        # Create a simple scene with moving objects
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add some moving elements to simulate normal behavior
        # Moving circle (person walking)
        x1 = int(100 + 50 * np.sin(i * 0.1))
        y1 = int(200 + 30 * np.cos(i * 0.05))
        cv2.circle(frame, (x1, y1), 20, (0, 255, 0), -1)
        
        # Moving rectangle (vehicle)
        x2 = int(400 + 80 * np.cos(i * 0.08))
        y2 = int(300 + 40 * np.sin(i * 0.12))
        cv2.rectangle(frame, (x2-15, y2-15), (x2+15, y2+15), (255, 0, 0), -1)
        
        # Add some static elements (infrastructure)
        cv2.rectangle(frame, (50, 50), (100, 100), (128, 128, 128), -1)
        cv2.rectangle(frame, (500, 400), (600, 450), (128, 128, 128), -1)
        
        # Add frame number for debugging
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        # Progress indicator
        if i % 60 == 0:
            print(f'Progress: {i}/{total_frames} frames ({i/total_frames*100:.1f}%)')
    
    out.release()
    print(f'Sample video created successfully!')
    
    # Verify the file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f'File size: {file_size/1024:.1f} KB')
    else:
        print('ERROR: Video file was not created!')

if __name__ == '__main__':
    create_sample_video() 