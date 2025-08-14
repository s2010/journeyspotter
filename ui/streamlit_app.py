"""
Streamlit UI for JourneySpotter demo.
Provides a user-friendly interface for travel video/image analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import streamlit as st

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="JourneySpotter Demo",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class UIService:
    """Service class for UI operations."""
    
    def __init__(self) -> None:
        """Initialize UI service."""
        self.settings = get_settings()
        self.api_base_url = self.settings.ui.api_base_url
    
    def call_analyze_api(self, file_content: bytes, filename: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Call the /analyze API endpoint."""
        try:
            files = {"file": (filename, file_content, "application/octet-stream")}
            response = requests.post(
                f"{self.api_base_url}/analyze", 
                files=files, 
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.ConnectionError:
            return None, "Could not connect to API. Make sure the backend is running."
        except Exception as e:
            return None, f"Request failed: {str(e)}"
    
    def check_api_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def display_analysis_results(results: Dict) -> None:
    """Display analysis results in a formatted way."""
    if not results:
        return
    
    # Main results
    st.subheader("🎯 Analysis Results")
    
    # Summary
    if "summary" in results:
        st.info(f"**Summary:** {results['summary']}")
    
    # Confidence score
    if "confidence" in results:
        confidence = results["confidence"]
        st.metric("Confidence Score", f"{confidence:.2f}")
    
    # Locations found
    if "locations" in results and results["locations"]:
        st.subheader("📍 Locations Detected")
        
        for i, location in enumerate(results["locations"], 1):
            with st.expander(f"Location {i}: {location.get('location', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Location:** {location.get('location', 'N/A')}")
                with col2:
                    st.write(f"**Country:** {location.get('country', 'N/A')}")
                with col3:
                    st.write(f"**Type:** {location.get('type', 'N/A')}")
    else:
        st.warning("No locations detected in the media.")
    
    # Extracted text
    if "extracted_text" in results and results["extracted_text"]:
        with st.expander("📝 Extracted Text (OCR)"):
            st.text_area("Raw OCR Output", results["extracted_text"], height=100, disabled=True)
    
    # Raw JSON (for debugging)
    with st.expander("🔧 Raw API Response"):
        st.json(results)


def main() -> None:
    """Main Streamlit application."""
    ui_service = UIService()
    
    # Header
    st.title("🗺️ JourneySpotter Demo")
    st.markdown("**Video/image analysis using OCR + Groq LLM**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo analyzes travel videos and images to:
        - Extract text using OCR (EasyOCR/Tesseract)
        - Identify locations and travel information
        - Provide intelligence analysis
        """)
        
        st.header("📁 Supported Formats")
        st.markdown("""
        **Images:** .jpg, .jpeg, .png, .bmp, .tiff
        **Videos:** .mp4, .avi, .mov, .mkv
        """)
        
        # API Health Check
        st.header("🏥 API Status")
        if ui_service.check_api_health():
            st.success("✅ API is running")
        else:
            st.error("❌ Cannot connect to API")
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload File", "🎬 Try Sample"])
    
    with tab1:
        st.header("Upload Your Media")
        
        uploaded_file = st.file_uploader(
            "Choose a video or image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv'],
            help="Upload a travel video or image for analysis"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Analyze button
            if st.button("🔍 Analyze Media", type="primary"):
                with st.spinner("Analyzing media... This may take a few moments."):
                    # Read file content
                    file_content = uploaded_file.read()
                    
                    # Call API
                    results, error = ui_service.call_analyze_api(file_content, uploaded_file.name)
                    
                    if error:
                        st.error(f"❌ Analysis failed: {error}")
                    else:
                        display_analysis_results(results)
    
    with tab2:
        st.header("Try Sample Media")
        st.markdown("Use our provided sample files to test the system:")
        
        # Check for sample files
        samples_dir = Path("samples")
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*"))
            
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose a sample file:",
                    options=[f.name for f in sample_files],
                    help="Select a sample file to analyze"
                )
                
                if st.button("🎯 Analyze Sample", type="primary"):
                    sample_path = samples_dir / selected_sample
                    
                    with st.spinner(f"Analyzing {selected_sample}..."):
                        try:
                            with open(sample_path, "rb") as f:
                                file_content = f.read()
                            
                            results, error = ui_service.call_analyze_api(file_content, selected_sample)
                            
                            if error:
                                st.error(f"❌ Analysis failed: {error}")
                            else:
                                display_analysis_results(results)
                                
                        except Exception as e:
                            st.error(f"❌ Failed to read sample file: {str(e)}")
            else:
                st.warning("No sample files found in the samples/ directory.")
        else:
            st.warning("Samples directory not found. Sample files will be available after setup.")
            
            # Show demo results instead
            st.subheader("📋 Demo Results Preview")
            demo_results = {
                "locations": [
                    {"location": "Tokyo Station", "country": "Japan", "type": "train_station"},
                    {"location": "Shibuya", "country": "Japan", "type": "district"}
                ],
                "summary": "Journey through Tokyo featuring major transportation hubs and districts",
                "extracted_text": "Tokyo Station Platform 1 Shibuya Crossing JR Yamanote Line",
                "confidence": 0.85,
                "file_type": "image"
            }
            display_analysis_results(demo_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Powered by <strong>Groq Llama 3.1 8B</strong> | 
        🔍 OCR by <strong>EasyOCR/Tesseract</strong> | 
        ⚡ Built with <strong>FastAPI + Streamlit</strong></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
