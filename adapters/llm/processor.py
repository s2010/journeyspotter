"""
LLM adapter implementation using Groq API.
Provides intelligent content analysis capabilities.
"""

import json
import logging
from typing import Any, Dict, List

from groq import Groq

from config.settings import LLMSettings
from core.interfaces import LLMProcessor
from domain.models import Location, LocationType

logger = logging.getLogger(__name__)


class GroqLLMAdapter(LLMProcessor):
    """Groq LLM implementation for intelligent content analysis."""

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize Groq LLM adapter."""
        self.settings = settings
        self.client = Groq(api_key=settings.api_key)
        logger.info(f"Groq LLM initialized with model: {settings.model}")

    async def analyze_content(self, extracted_text: str) -> Dict[str, Any]:
        """Analyze extracted text for intelligent content analysis using Groq LLM."""
        if not extracted_text.strip():
            return self._create_empty_response(extracted_text)

        try:
            prompt = self._create_analysis_prompt(extracted_text)
            
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent content analyzer that extracts location information from text. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse and validate response
            try:
                result = json.loads(result_text)
                return self._validate_and_normalize_response(result, extracted_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {result_text}")
                return self._create_fallback_response(extracted_text)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._create_error_response(extracted_text, str(e))

    def _create_analysis_prompt(self, extracted_text: str) -> str:
        """Create analysis prompt for the LLM."""
        return f"""
Analyze the following text extracted from a video/image and provide intelligent content analysis:

Text: "{extracted_text}"

Please extract and return a JSON response with:
1. "locations": Array of location objects with "location", "country", "type" fields
2. "summary": Brief content summary
3. "extracted_text": The original text
4. "confidence": Overall confidence score (0.0-1.0)

Focus on identifying:
- Place names, cities, countries
- Transportation hubs (airports, train stations, bus stops)
- Tourist attractions, landmarks
- Street names, addresses
- Signage and text content

Valid location types: train_station, airport, bus_stop, city, landmark, transportation_hub, district, unknown

Return only valid JSON, no additional text.
"""

    def _validate_and_normalize_response(self, result: Dict[str, Any], extracted_text: str) -> Dict[str, Any]:
        """Validate and normalize LLM response."""
        # Ensure required fields exist
        normalized = {
            "locations": [],
            "summary": result.get("summary", "Content analysis completed"),
            "extracted_text": extracted_text,
            "confidence": float(result.get("confidence", 0.5))
        }
        
        # Validate confidence
        if not 0.0 <= normalized["confidence"] <= 1.0:
            normalized["confidence"] = 0.5
        
        # Process locations
        locations = result.get("locations", [])
        if isinstance(locations, list):
            for loc in locations:
                if isinstance(loc, dict):
                    try:
                        # Normalize location type
                        location_type = self._normalize_location_type(loc.get("type", "unknown"))
                        
                        location = Location(
                            name=str(loc.get("location", "")).strip(),
                            country=str(loc.get("country", "")).strip(),
                            location_type=location_type,
                            confidence=float(loc.get("confidence", 0.5))
                        )
                        
                        # Convert back to dict for API response
                        normalized["locations"].append({
                            "location": location.name,
                            "country": location.country,
                            "type": location.location_type.value,
                            "confidence": location.confidence
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid location data: {loc}, error: {e}")
                        continue
        
        return normalized

    def _normalize_location_type(self, location_type: str) -> LocationType:
        """Normalize location type string to LocationType enum."""
        type_mapping = {
            "train_station": LocationType.TRAIN_STATION,
            "train station": LocationType.TRAIN_STATION,
            "railway station": LocationType.TRAIN_STATION,
            "airport": LocationType.AIRPORT,
            "bus_stop": LocationType.BUS_STOP,
            "bus stop": LocationType.BUS_STOP,
            "bus station": LocationType.BUS_STOP,
            "city": LocationType.CITY,
            "landmark": LocationType.LANDMARK,
            "transportation_hub": LocationType.TRANSPORTATION_HUB,
            "transportation hub": LocationType.TRANSPORTATION_HUB,
            "district": LocationType.DISTRICT,
            "unknown": LocationType.UNKNOWN
        }
        
        normalized_type = location_type.lower().strip()
        return type_mapping.get(normalized_type, LocationType.UNKNOWN)

    def _create_empty_response(self, extracted_text: str) -> Dict[str, Any]:
        """Create response for empty extracted text."""
        return {
            "locations": [],
            "summary": "No text detected in the provided media",
            "extracted_text": extracted_text,
            "confidence": 0.0
        }

    def _create_fallback_response(self, extracted_text: str) -> Dict[str, Any]:
        """Create fallback response when LLM response parsing fails."""
        return {
            "locations": [],
            "summary": f"Analysis completed. Extracted text: {extracted_text[:200]}...",
            "extracted_text": extracted_text,
            "confidence": 0.5
        }

    def _create_error_response(self, extracted_text: str, error: str) -> Dict[str, Any]:
        """Create error response when LLM analysis fails."""
        return {
            "locations": [],
            "summary": f"Analysis failed: {error}",
            "extracted_text": extracted_text,
            "confidence": 0.0,
            "error": error
        }


def create_llm_processor(settings: LLMSettings) -> LLMProcessor:
    """Factory function to create LLM processor."""
    return GroqLLMAdapter(settings)
