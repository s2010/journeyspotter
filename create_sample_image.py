#!/usr/bin/env python3
"""
Create a sample travel image with text for demo purposes
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_travel_image():
    """Create a sample travel image with location text"""
    
    # Create a new image with a travel-themed background
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw background elements
    draw.rectangle([50, 50, width-50, height-50], fill='white', outline='navy', width=3)
    
    # Add travel-related text elements
    texts = [
        ("TOKYO STATION", 100, 120, font_large, 'navy'),
        ("Platform 7 - Shinkansen", 120, 180, font_medium, 'darkblue'),
        ("Next: Kyoto 京都", 120, 220, font_medium, 'darkgreen'),
        ("JR Central", 120, 280, font_small, 'gray'),
        ("Departure: 14:30", 120, 320, font_small, 'red'),
        ("東京駅", 120, 360, font_medium, 'navy'),
        ("Welcome to Japan", 120, 420, font_medium, 'purple'),
        ("Exit → 出口", 120, 460, font_small, 'black')
    ]
    
    for text, x, y, font, color in texts:
        draw.text((x, y), text, fill=color, font=font)
    
    # Add some decorative elements
    draw.ellipse([600, 100, 700, 200], fill='yellow', outline='orange', width=2)
    draw.text((620, 140), "JR", fill='red', font=font_large)
    
    # Save the image
    output_path = "samples/sample_travel_image.jpg"
    img.save(output_path, "JPEG", quality=95)
    print(f"Sample travel image created: {output_path}")

if __name__ == "__main__":
    create_sample_travel_image()
