import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file 
import google.generativeai as genai   
import time
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration with custom theme
st.set_page_config(
    page_title="Video Summarizer Agent",
    page_icon="📽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTitle {
        color: #1E88E5;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .stHeader {
        color: #424242;
        text-align: center;
        margin-bottom: 3rem !important;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        padding: 1rem;
        font-size: 1rem;
        height: 120px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .results-container {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Application header
st.markdown('<h1 class="stTitle">Video Summarizer Agent 📽️</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="stHeader">Powered by Gemini AI</h2>', unsafe_allow_html=True)

def initialize_agent():
    return Agent(
        name="Video Analysis Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

multimodal_Agent = initialize_agent()

# File upload section with custom styling
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
video_file = st.file_uploader(
    "Drop your video file here",
    type=['mp4', 'mov', 'avi'],
    help="Supported formats: MP4, MOV, AVI"
)
st.markdown('</div>', unsafe_allow_html=True)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # Display video with custom container
    st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
    st.video(video_path, format="video/mp4", start_time=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Query input section
    user_query = st.text_area(
        "What would you like to know about this video?",
        placeholder="Ask any question about the video content. Our AI will analyze and provide detailed insights.",
        help="Be specific in your question to get the most relevant analysis."
    )

    # Analysis button with loading animation
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze_button = st.button("🔍 Analyze Video", use_container_width=True)

    if analyze_button:
        if not user_query:
            st.warning("⚠️ Please enter a question or topic to analyze the video.")
        else:
            try:
                with st.spinner("🎬 Processing your video..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Query: {user_query}

                    Please provide a comprehensive analysis addressing the query,
                    including relevant insights and observations from the video.
                    Format your response in a clear, well-structured manner.
                    """

                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                # Display results in a styled container
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("📊 Analysis Results")
                st.markdown(response.content)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as error:
                st.error(f"❌ Analysis Error: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("👆 Start by uploading a video file above")