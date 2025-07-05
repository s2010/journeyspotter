#!/usr/bin/env python3
"""
Create a test image with travel text for analysis demo
"""

import cv2
import numpy as np
import os

def create_travel_test_image():
    """Create a test image with travel destination text"""
    
    # Create a white background image
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add a blue header background
    cv2.rectangle(img, (0, 0), (width, 80), (100, 50, 0), -1)
    
    # Add travel text content
    texts = [
        ("DEPARTURE BOARD", (250, 50), 1.2, (255, 255, 255), 3),
        ("FLIGHT BA123 LONDON HEATHROW LHR", (50, 150), 0.8, (0, 0, 0), 2),
        ("DESTINATION: PARIS CHARLES DE GAULLE CDG", (50, 200), 0.8, (0, 0, 0), 2),
        ("DEPARTURE: 14:30  ARRIVAL: 16:45", (50, 250), 0.8, (0, 0, 0), 2),
        ("GATE: A24  TERMINAL: 5", (50, 300), 0.8, (0, 0, 0), 2),
        ("", (0, 0), 0, (0, 0, 0), 0),  # Spacer
        ("CONNECTING FLIGHT:", (50, 380), 0.7, (100, 100, 100), 2),
        ("PARIS CDG → ROME FIUMICINO FCO", (50, 420), 0.8, (0, 0, 0), 2),
        ("ALITALIA AZ123  DEP: 18:15", (50, 460), 0.8, (0, 0, 0), 2),
        ("FINAL DESTINATION: ROMA", (50, 510), 0.8, (0, 0, 0), 2),
    ]
    
    # Add each text to the image
    for text, pos, scale, color, thickness in texts:
        if text:  # Skip empty spacer
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
    # Add some decorative elements
    cv2.rectangle(img, (30, 120), (770, 340), (200, 200, 200), 2)
    cv2.rectangle(img, (30, 360), (770, 540), (200, 200, 200), 2)
    
    # Save the image
    output_path = '/app/videos/travel_test.jpg'
    os.makedirs('/app/videos', exist_ok=True)
    cv2.imwrite(output_path, img)
    
    print(f"Created travel test image: {output_path}")
    print("Content: London -> Paris -> Rome flight itinerary")
    print("Ready for testing travel intelligence analysis")
    
    return output_path

if __name__ == "__main__":
    create_travel_test_image() 