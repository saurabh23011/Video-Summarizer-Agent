import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import time
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import PyPDF2
import io

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
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
    .chat-container {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
    }
    .pdf-list {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="stTitle">PDF Chat Assistant 📚</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="stHeader">Chat with Multiple PDFs</h2>', unsafe_allow_html=True)

def initialize_agent():
    return Agent(
        name="PDF Analysis Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Initialize session state for storing PDFs and chat history
if 'pdfs' not in st.session_state:
    st.session_state.pdfs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the agent
multimodal_Agent = initialize_agent()

# PDF upload section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload your PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="You can upload multiple PDF files"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded PDFs
if uploaded_files:
    for pdf_file in uploaded_files:
        if pdf_file.name not in st.session_state.pdfs:
            with st.spinner(f"Processing {pdf_file.name}..."):
                pdf_text = extract_text_from_pdf(pdf_file)
                st.session_state.pdfs[pdf_file.name] = pdf_text
                st.success(f"Successfully processed {pdf_file.name}")

# Display uploaded PDFs
if st.session_state.pdfs:
    st.markdown("### 📑 Uploaded PDFs")
    st.markdown('<div class="pdf-list">', unsafe_allow_html=True)
    for pdf_name in st.session_state.pdfs.keys():
        st.write(f"- {pdf_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    user_question = st.text_area(
        "Ask a question about your PDFs",
        placeholder="Ask any question about the content of your uploaded PDFs...",
        help="Be specific in your question to get the most relevant information from the PDFs"
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        chat_button = st.button("💬 Ask Question", use_container_width=True)

    if chat_button and user_question:
        try:
            with st.spinner("🤔 Analyzing PDFs and preparing response..."):
                # Combine all PDF contents for context
                combined_context = "\n\n===\n\n".join(
                    f"Document: {name}\n{content}" 
                    for name, content in st.session_state.pdfs.items()
                )
                
                analysis_prompt = f"""
                Context from PDFs:
                {combined_context}

                User Question: {user_question}

                Please provide a comprehensive answer based on the content of the PDFs.
                Include specific references to the source documents where appropriate.
                Format your response in a clear, well-structured manner.
                """

                response = multimodal_Agent.run(analysis_prompt)
                
                # Add to chat history
                st.session_state.chat_history.append({"question": user_question, "answer": response.content})

            # Display chat history
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.subheader("💭 Chat History")
            for chat in st.session_state.chat_history:
                st.markdown("**Question:**")
                st.markdown(chat["question"])
                st.markdown("**Answer:**")
                st.markdown(chat["answer"])
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as error:
            st.error(f"❌ Analysis Error: {error}")

else:
    st.info("👆 Start by uploading one or more PDF files above")

# Add clear chat history button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()