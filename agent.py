import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import os
from pathlib import Path
import PyPDF2
import io
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Phi PDF Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styles
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTitle {
        color: #2E7D32;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
    }
    .stHeader {
        color: #424242;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .upload-box {
        border: 2px dashed #2E7D32;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .chat-container {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .stButton button {
        background-color: #2E7D32;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1B5E20;
    }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="stTitle">Phi PDF Voice Assistant 🎙️</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="stHeader">I’ll Speak My Answers to You!</h2>', unsafe_allow_html=True)

def initialize_agent():
    """Initialize the Phi Agent with Gemini and DuckDuckGo"""
    return Agent(
        name="PDF Voice Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
        description="A friendly research assistant that speaks its answers"
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def text_to_speech(text):
    """Convert text to speech and return the audio file path"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Voice generation failed: {str(e)}")
        return None

# Initialize session state
if 'pdfs' not in st.session_state:
    st.session_state.pdfs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = initialize_agent()

# PDF upload section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload your research PDFs",
    type=['pdf'],
    accept_multiple_files=True,
    help="Upload one or more PDF files to chat about their contents"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded PDFs
if uploaded_files:
    for pdf_file in uploaded_files:
        if pdf_file.name not in st.session_state.pdfs:
            with st.spinner(f"Reading {pdf_file.name}..."):
                pdf_text = extract_text_from_pdf(pdf_file)
                st.session_state.pdfs[pdf_file.name] = pdf_text
                st.success(f"Got it! Added {pdf_file.name} to my research pile!")

# Display uploaded PDFs
if st.session_state.pdfs:
    st.write("### My Research Stack 📚")
    for pdf_name in st.session_state.pdfs.keys():
        st.write(f"- {pdf_name}")

# Chat interface
user_question = st.text_area(
    "What would you like me to research?",
    placeholder="Ask me anything, and I’ll answer in voice!",
    height=100
)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    ask_button = st.button("🎤 Speak to Me!", use_container_width=True)

if ask_button and user_question:
    with st.spinner("Thinking and preparing to speak..."):
        try:
            # Prepare context from PDFs if available
            if st.session_state.pdfs:
                combined_context = "\n\n===\n\n".join(
                    f"Document: {name}\n{content}" 
                    for name, content in st.session_state.pdfs.items()
                )
                prompt = f"""
                Hello! I'm your friendly voice assistant.
                Here's the context from the uploaded PDFs:
                {combined_context}

                User Question: {user_question}

                Provide a conversational, friendly response based on the PDF content.
                Reference specific documents when possible. Keep it concise for voice output!
                """
            else:
                prompt = f"""
                Hello! I'm your friendly voice assistant.
                User Question: {user_question}

                Since no PDFs are uploaded, use your general knowledge and search tools if needed.
                Provide a conversational, friendly response. Keep it concise for voice output!
                """

            # Get response from Phi Agent
            response = st.session_state.agent.run(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            # Convert to speech
            audio_file = text_to_speech(answer)
            if audio_file:
                # Display text and play audio
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "audio": audio_file
                })
        except Exception as e:
            st.error(f"Oops! Something went wrong: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("Our Conversation")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You asked:** {chat['question']}")
        st.markdown(f"**I said:** {chat['answer']}")
        if 'audio' in chat and chat['audio']:
            st.audio(chat['audio'], format="audio/mp3")
            # Clean up temporary file after use (optional)
            # os.unlink(chat['audio'])
        st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat history
    if st.button("🧹 Start Fresh"):
        # Clean up audio files
        for chat in st.session_state.chat_history:
            if 'audio' in chat and os.path.exists(chat['audio']):
                os.unlink(chat['audio'])
        st.session_state.chat_history = []
        st.experimental_rerun()

if not st.session_state.pdfs and not user_question:
    st.info("Hey there! Upload some PDFs or ask me a question, and I’ll speak my answer!")
