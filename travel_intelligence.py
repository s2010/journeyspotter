#!/usr/bin/env python3
"""
Travel Intelligence Module
Location extraction and enrichment using language models
"""

import json
import time
import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LocationInfo:
    """Structured location information"""
    location: str
    country: str
    type: str
    order: int
    confidence: float = 0.0

class TravelIntelligence:
    """Travel text analysis system for location extraction"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize Travel Intelligence module
        
        Development note: Originally used rule-based parsing, 
        evolved to use language models for better accuracy
        
        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: Language model to use (default: gpt-4)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def extract_and_enrich_locations(self, ocr_text: str) -> Tuple[List[LocationInfo], str]:
        """
        Extract locations from OCR text and enrich with metadata
        
        Development evolution: Started with simple regex matching,
        added language model analysis for better context understanding
        
        Args:
            ocr_text: Raw text extracted from OCR
            
        Returns:
            Tuple of (list of LocationInfo objects, trip summary)
        """
        if not ocr_text or not ocr_text.strip():
            return [], "No text provided for analysis."
            
        try:
            # Clean the OCR text first
            cleaned_text = self._clean_ocr_text(ocr_text)
            
            # Extract and enrich locations using language model
            locations_data, summary = self._analyze_with_language_model(cleaned_text)
            
            # Convert to LocationInfo objects
            locations = []
            for i, loc_data in enumerate(locations_data):
                location_info = LocationInfo(
                    location=loc_data.get('location', ''),
                    country=loc_data.get('country', ''),
                    type=loc_data.get('type', ''),
                    order=loc_data.get('order', i + 1),
                    confidence=loc_data.get('confidence', 0.8)
                )
                locations.append(location_info)
            
            return locations, summary
            
        except Exception as e:
            logger.error(f"Error in location extraction: {e}")
            return [], f"Error analyzing travel data: {str(e)}"
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean and normalize OCR text for better analysis
        
        Iterative improvement: Added more sophisticated text cleaning
        based on common OCR artifacts observed in travel documents
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-\.,;:()\[\]{}]', '', text)
        
        # Normalize common transportation keywords
        text = re.sub(r'\b(arr|arrival|dep|departure)\b', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
        text = re.sub(r'\b(to|from|via)\b', lambda m: m.group(1).lower(), text, flags=re.IGNORECASE)
        
        return text
    
    def _analyze_with_language_model(self, text: str) -> Tuple[List[Dict], str]:
        """
        Use language model to analyze text and extract location information
        
        Development note: Refined prompt engineering through multiple iterations
        to get consistent, structured output from the language model
        
        Args:
            text: Cleaned OCR text
            
        Returns:
            Tuple of (locations list, summary string)
        """
        system_prompt = """You are a travel text analysis system that processes text from transportation displays, tickets, and signs to extract location information.

Your task is to:
1. Identify valid location names (cities, countries, landmarks, airports, stations)
2. Clean and standardize location names
3. Enrich each location with metadata
4. Order locations by travel sequence when possible
5. Generate a natural trip summary

Return a JSON object with this exact structure:
{
  "locations": [
    {
      "location": "standardized location name",
      "country": "country name",
      "type": "location type (e.g., capital city, coastal town, airport, historic city, mountain resort)",
      "order": 1,
      "confidence": 0.9
    }
  ],
  "summary": "A concise, natural language trip description"
}

Guidelines:
- Only include real, identifiable places
- Standardize names (e.g., "NYC" → "New York City")
- Infer travel order from context clues (departure/arrival times, sequence)
- Types: capital city, historic city, coastal town, mountain resort, airport, train station, landmark, etc.
- Confidence: 0.7-1.0 based on clarity and certainty
- Summary: 1-2 sentences describing the journey or locations visited"""

        user_prompt = f"""Analyze this text from a travel/transportation context and extract location information:

TEXT: {text}

Please identify locations, clean the names, and provide enriched metadata as specified."""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=1000,
                    timeout=30
                )
                
                # Parse the response
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    raise ValueError("No valid JSON found in response")
                
                result = json.loads(json_match.group())
                
                # Validate the structure
                if 'locations' not in result or 'summary' not in result:
                    raise ValueError("Invalid response structure")
                
                locations = result['locations']
                summary = result['summary']
                
                # Validate locations data
                for loc in locations:
                    if not all(key in loc for key in ['location', 'country', 'type', 'order']):
                        raise ValueError("Missing required location fields")
                
                logger.info(f"Successfully extracted {len(locations)} locations")
                return locations, summary
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return [], "Error: Could not parse language model response"
                    
            except openai.RateLimitError:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return [], "Error: API rate limit exceeded"
                    
            except openai.APITimeoutError:
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return [], "Error: API timeout"
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return [], f"Error: {str(e)}"
                    
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        return [], "Error: Failed to analyze text after multiple attempts"
    
    def format_locations_json(self, locations: List[LocationInfo]) -> str:
        """
        Format locations as clean JSON string
        
        Args:
            locations: List of LocationInfo objects
            
        Returns:
            Pretty-formatted JSON string
        """
        data = []
        for loc in locations:
            data.append({
                "location": loc.location,
                "country": loc.country,
                "type": loc.type,
                "order": loc.order
            })
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def get_travel_insights(self, locations: List[LocationInfo]) -> Dict[str, any]:
        """
        Generate additional travel insights from location data
        
        Development note: Added statistical analysis features
        based on user feedback for better trip understanding
        
        Args:
            locations: List of LocationInfo objects
            
        Returns:
            Dictionary with travel insights
        """
        if not locations:
            return {"total_locations": 0, "countries": [], "types": []}
        
        countries = list(set(loc.country for loc in locations))
        types = list(set(loc.type for loc in locations))
        
        insights = {
            "total_locations": len(locations),
            "countries_visited": len(countries),
            "countries": countries,
            "location_types": types,
            "journey_span": {
                "first": locations[0].location if locations else None,
                "last": locations[-1].location if len(locations) > 1 else None
            }
        }
        
        return insights

def demo_travel_intelligence():
    """Demo function to test the travel intelligence module"""
    # Sample OCR text that might come from travel documents
    sample_texts = [
        "DEPARTURE: TOKYO 14:30 - ARRIVAL: KYOTO 17:45 JR SHINKANSEN",
        "London Heathrow LHR → Paris CDG → Rome FCO Final Destination",
        "Bus Route: NYC Times Square - Philadelphia - Washington DC",
        "SWISS INTERNATIONAL Basel-Mulhouse → Zurich → Geneva Lake",
        "Cruise: Barcelona Spain, Nice France, Palma Mallorca, Rome Italy"
    ]
    
    # Note: This requires a valid OpenAI API key
    try:
        ti = TravelIntelligence()
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Input: {text}")
            
            locations, summary = ti.extract_and_enrich_locations(text)
            
            print(f"Summary: {summary}")
            print(f"Locations JSON:")
            print(ti.format_locations_json(locations))
            
            insights = ti.get_travel_insights(locations)
            print(f"Insights: {insights}")
            
    except Exception as e:
        print(f"Demo error (likely missing API key): {e}")
        print("To test this module, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")

if __name__ == "__main__":
    demo_travel_intelligence() 