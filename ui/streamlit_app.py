"""
Streamlit UI for JourneySpotter demo.
Provides a user-friendly interface for video/image analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Anomaly Detector Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit UI elements (Deploy button and main menu)
st.markdown("""
<style>
    /* Hide Deploy button */
    .stAppDeployButton {
        display: none !important;
    }
    
    /* Hide Main Menu (three dots) */
    .stMainMenu {
        display: none !important;
    }
    
    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Optional: Hide footer */
    .stAppViewContainer > .main > div > div > div > div > section > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


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


def display_anomaly_results(results: Dict) -> None:
    """Display anomaly detection results with visualization."""
    if not results.get("anomaly_scores"):
        return
    
    st.subheader("🚨 Anomaly Detection Results")
    
    anomaly_scores = results["anomaly_scores"]
    anomaly_threshold = results.get("anomaly_threshold", -0.1)
    anomalous_frames = results.get("anomalous_frames", [])
    anomaly_detected = results.get("anomaly_detected", False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Frames", len(anomaly_scores))
    with col2:
        anomalous_count = sum(anomalous_frames) if anomalous_frames else 0
        st.metric("Anomalous Frames", anomalous_count)
    with col3:
        anomaly_percentage = (anomalous_count / len(anomaly_scores)) * 100 if anomaly_scores else 0
        st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")
    with col4:
        status = "🚨 DETECTED" if anomaly_detected else "✅ NORMAL"
        st.metric("Status", status)
    
    # Anomaly score visualization
    if anomaly_scores:
        fig = go.Figure()
        
        # Add anomaly scores line
        fig.add_trace(go.Scatter(
            x=list(range(len(anomaly_scores))),
            y=anomaly_scores,
            mode='lines+markers',
            name='Anomaly Score',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add threshold line
        fig.add_hline(
            y=anomaly_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({anomaly_threshold})"
        )
        
        # Highlight anomalous frames
        if anomalous_frames:
            anomalous_indices = [i for i, is_anomalous in enumerate(anomalous_frames) if is_anomalous]
            if anomalous_indices:
                fig.add_trace(go.Scatter(
                    x=anomalous_indices,
                    y=[anomaly_scores[i] for i in anomalous_indices],
                    mode='markers',
                    name='Anomalous Frames',
                    marker=dict(color='red', size=8, symbol='x')
                ))
        
        fig.update_layout(
            title="Anomaly Scores Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Anomaly Score",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Frame-by-frame details
        with st.expander("📋 Frame-by-Frame Details"):
            for i, (score, is_anomalous) in enumerate(zip(anomaly_scores, anomalous_frames)):
                status_icon = "🚨" if is_anomalous else "✅"
                st.write(f"Frame {i+1}: {status_icon} Score: {score:.4f}")


def display_analysis_results(results: Dict) -> None:
    """Display analysis results in a formatted way."""
    st.subheader("📊 Analysis Results")
    
    # Display summary
    if results.get("summary"):
        st.markdown("**Summary:**")
        st.write(results["summary"])
    
    # Display extracted text
    if results.get("extracted_text"):
        st.markdown("**Extracted Text:**")
        st.text_area("", results["extracted_text"], height=100, disabled=True)
    
    # Display locations
    if results.get("locations"):
        st.markdown("**Detected Locations:**")
        for i, location in enumerate(results["locations"], 1):
            with st.expander(f"Location {i}: {location['location']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Country:** {location['country']}")
                    st.write(f"**Type:** {location['type']}")
                with col2:
                    st.write(f"**Confidence:** {location['confidence']:.2%}")
    else:
        st.info("No locations detected in this media.")
    
    # Display confidence score
    if "confidence" in results:
        st.markdown("**Overall Confidence:**")
        st.progress(results["confidence"])
        st.write(f"{results['confidence']:.2%}")
    
    # Display file info
    if results.get("filename"):
        st.markdown("**File Information:**")
        st.write(f"**Filename:** {results['filename']}")
        st.write(f"**Type:** {results.get('file_type', 'Unknown')}")
    
    # Display anomaly detection results
    display_anomaly_results(results)


def main() -> None:
    """Main Streamlit application."""
    ui_service = UIService()
    
    # Header
    st.title("Anomaly Detector Demo")
    st.markdown("**Video/image analysis with intelligent text processing**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo analyzes videos and images to:
        - Extract text using OCR (EasyOCR/Tesseract)
        - Identify locations and content information
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
    tab1, tab2 = st.tabs(["🎬 Try Sample", "📤 Upload File"])
    
    with tab1:
        st.header("Try Sample Media")
        
        # Featured Traffic Demo
        st.subheader("🚗 Traffic Analysis Demo")
        st.markdown("""
        **Featured Demo:** Real traffic video analysis with intelligent text processing
        - 7-second traffic intersection footage
        - Real-time text extraction from signs, license plates, and road markings
        - Advanced pattern recognition and content analysis
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🎯 Analyze Traffic Demo", type="primary", use_container_width=True):
                traffic_sample = "traffic_demo_01.mp4"
                sample_path = Path("samples") / traffic_sample
                
                if sample_path.exists():
                    with st.spinner("🔍 Analyzing traffic video with intelligent processing..."):
                        try:
                            with open(sample_path, "rb") as f:
                                file_content = f.read()
                            
                            results, error = ui_service.call_analyze_api(file_content, traffic_sample)
                            
                            if error:
                                st.error(f"❌ Analysis failed: {error}")
                            else:
                                st.success("✅ Intelligent analysis completed!")
                                display_analysis_results(results)
                                
                        except Exception as e:
                            st.error(f"❌ Failed to read traffic demo file: {str(e)}")
                else:
                    st.info("No other sample files found.")
        # Other Sample Files
        st.subheader("📁 Other Sample Files")
        samples_dir = Path("samples")
        if samples_dir.exists():
            sample_files = [f for f in samples_dir.glob("*") if f.is_file() and f.name != "traffic_demo_01.mp4"]
            
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose another sample file:",
                    options=[f.name for f in sample_files],
                    help="Select a sample file to analyze"
                )
                
                if st.button("🔍 Analyze Other Sample", type="secondary"):
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
                st.info("Traffic demo is the primary sample. Other samples will appear here when available.")
        else:
            st.warning("Samples directory not found. Sample files will be available after setup.")
    
    with tab2:
        st.header("Upload Your Media")
        
        uploaded_file = st.file_uploader(
            "Choose a video or image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video or image for analysis"
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
         Built with ❤️ and lots of matcha
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
