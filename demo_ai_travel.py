#!/usr/bin/env python3
"""
Demo: Travel Intelligence System
Showcase the travel text analysis functionality
"""

import json
from travel_intelligence import TravelIntelligence

def demo_text_analysis():
    """Demo the travel intelligence with sample travel texts"""
    
    print("Travel Intelligence Demo")
    print("=" * 50)
    
    # Sample OCR texts that might come from travel videos/images
    sample_texts = [
        "DEPARTURE: TOKYO 14:30 - ARRIVAL: KYOTO 17:45 JR SHINKANSEN",
        "London Heathrow LHR → Paris CDG → Rome FCO Final Destination", 
        "Bus Route: NYC Times Square - Philadelphia - Washington DC",
        "SWISS INTERNATIONAL Basel-Mulhouse → Zurich → Geneva Lake",
        "Mediterranean Cruise: Barcelona Spain, Nice France, Palma Mallorca, Rome Italy",
        "Flight AA123 Boston Logan BOS to San Francisco SFO Gate 24A",
        "Train Ticket München Hauptbahnhof to Wien Westbahnhof via Salzburg"
    ]
    
    try:
        # Initialize travel intelligence
        ti = TravelIntelligence()
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\nSample {i}: Travel Text Analysis")
            print(f"Input: {text}")
            print("-" * 60)
            
            # Extract and enrich locations
            locations, summary = ti.extract_and_enrich_locations(text)
            
            print(f"Analysis Summary: {summary}")
            
            if locations:
                print(f"\nLocations Detected ({len(locations)}):")
                for j, loc in enumerate(locations, 1):
                    print(f"   {j}. {loc.location}, {loc.country}")
                    print(f"      Type: {loc.type} | Order: {loc.order}")
                
                # Get travel insights
                insights = ti.get_travel_insights(locations)
                print(f"\nInsights:")
                print(f"   Countries: {insights.get('countries_visited', 0)}")
                print(f"   Location types: {', '.join(insights.get('location_types', []))}")
                
                # Show JSON output
                json_output = ti.format_locations_json(locations)
                print(f"\nJSON Output:")
                print(json_output)
            else:
                print("No locations detected")
            
            print("\n" + "="*70)
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTo run this demo, you need:")
        print("1. OpenAI API key: export OPENAI_API_KEY='your-key'") 
        print("2. Install dependencies: pip install openai")

def demo_integration_example():
    """Show how to integrate the travel intelligence into other applications"""
    
    print("\nIntegration Example")
    print("=" * 30)
    
    sample_code = '''
# Example: Integrate travel analysis into your app

from travel_intelligence import TravelIntelligence
from ocr_processor import OCRProcessor

# Initialize components
ti = TravelIntelligence()
ocr = OCRProcessor()

# Extract text from video
ocr_results = ocr.extract_text_from_video("vacation.mp4")
text = ocr.combine_text_results(ocr_results)

# Get location analysis
locations, summary = ti.extract_and_enrich_locations(text)

# Use the structured data
for location in locations:
    print(f"Visiting {location.location} in {location.country}")
    print(f"Type: {location.type}, Order: {location.order}")

print(f"Trip summary: {summary}")
'''
    
    print(sample_code)

if __name__ == "__main__":
    demo_text_analysis()
    demo_integration_example()
    
    print("\nNext Steps:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Try with real travel videos: python journey_spotter.py --analyze-travel video.mp4")
    print("3. Create OCR overlay videos: python journey_spotter.py --input video.mp4 --ocr-overlay")
    print("4. Use the structured JSON output in your applications!") 