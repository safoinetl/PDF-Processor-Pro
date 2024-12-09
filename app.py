import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import os
import concurrent.futures
import unicodedata
import logging
from typing import List, Callable, Dict, Any
import tempfile
import time
from datetime import datetime
import requests
import json
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add caching for better performance
@lru_cache(maxsize=1000)
def cache_chunk_summary(chunk: str) -> str:
    return chunk

class CustomPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font("Arial", size=12)
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Document Summary', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, ln=True, align='L')
        self.ln(5)
        self.set_font("Arial", size=12)

    def chapter_body(self, text):
        self.set_font("Arial", size=12)
        text = self.sanitize_text(text)
        self.multi_cell(0, 10, text)
        self.ln()

    def sanitize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', str(text))
        return ''.join(c for c in text if ord(c) < 128)

class MistralClient:
    def __init__(self, base_url="http://"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"
        self._session = requests.Session()

    def check_connection(self) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def generate_response(self, system_prompt: str, user_content: str, stream: bool = True) -> str:
        try:
            payload = {
                "model": "mistral",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "stream": stream
            }
            
            if stream:
                return self._stream_response(payload)
            else:
                response = self._session.post(self.chat_endpoint, json=payload)
                response.raise_for_status()
                return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling model API: {str(e)}")
            raise

    def _stream_response(self, payload: dict) -> str:
        response = self._session.post(self.chat_endpoint, json=payload, stream=True)
        response.raise_for_status()
        
        full_response = ""
        response_container = st.empty()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    full_response += content
                    response_container.markdown(full_response)
        
        return full_response

class PDFProcessor:
    def __init__(self):
        self.mistral_client = MistralClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def extract_text(pdf_file) -> str:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def process_chunks_parallel(self, chunks: List[str]) -> List[str]:
        futures = []
        for chunk in chunks:
            if len(chunk) < 500:
                futures.append(self.executor.submit(lambda x: x, chunk))
            else:
                futures.append(self.executor.submit(self.summarize_chunk, chunk))
        
        return [future.result() for future in concurrent.futures.as_completed(futures)]

    def summarize_chunk(self, chunk: str) -> str:
        cached_summary = cache_chunk_summary(chunk)
        if cached_summary != chunk:
            return cached_summary
            
        try:
            system_prompt = """Create a concise, focused summary capturing only essential information."""
            summary = self.mistral_client.generate_response(system_prompt, chunk, stream=False)
            return cache_chunk_summary(summary.strip())
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return chunk

    def create_summary_pdf(self, summaries: List[str], metadata: Dict[str, Any] = None) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"summary_{timestamp}.pdf"
            
            pdf = CustomPDF()
            pdf.add_page()
            
            if metadata:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Document Information", ln=True)
                pdf.set_font("Arial", size=12)
                for key, value in metadata.items():
                    pdf.cell(0, 10, f"{key}: {value}", ln=True)
                pdf.ln(10)
            
            pdf.chapter_title("Executive Summary")
            for i, summary in enumerate(summaries, 1):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Section {i}", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.chapter_body(summary)
                pdf.ln(5)
            
            pdf.output(output_filename)
            return output_filename
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise

def initialize_streamlit():
    st.set_page_config(
        page_title="PDF Processor Pro",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .stTextInput > div > div > input {
                padding: 15px;
                border-radius: 10px;
                background-color: #f8f9fa;
            }
            .stButton > button {
                padding: 12px 30px;
                border-radius: 10px;
                background-color: #007bff;
                color: white;
                font-weight: 500;
            }
            .stButton > button:hover {
                background-color: #0056b3;
            }
            .chat-message {
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                animation: fadeIn 0.5s;
            }
            .user-message {
                background-color: #f0f2f6;
            }
            .assistant-message {
                background-color: #e8f4ea;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .stExpander {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                margin: 10px 0;
            }
            /* Hide the default submit button */
            .stButton {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

def display_chat():
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.get('chat_history', []):
            message_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f"""
                <div class="chat-message {message_class}">
                    <div style="color: #666666; font-size: 0.8em; margin-bottom: 5px;">
                        {message["timestamp"]}
                    </div>
                    <div>
                        <strong>{'You' if message["role"] == "user" else 'Assistant'}:</strong> {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def create_qa_system(text: str) -> Callable:
    processor = st.session_state.processor
    chunks = processor.text_splitter.split_text(text)
    
    def qa_function(question: str) -> str:
        context = "\n".join(chunks[:3])
        system_prompt = """Answer questions based on the provided context. Be concise and accurate."""
        user_content = f"Context: {context}\n\nQuestion: {question}"
        return processor.mistral_client.generate_response(system_prompt, user_content)
    
    return qa_function

def handle_qa(question: str, qa_system: Callable):
    if not question.strip():
        return
        
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Add user question
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": timestamp
    })
    
    # Get streaming response
    response = qa_system(question)
    
    # Add assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": timestamp
    })

def main():
    initialize_streamlit()
    
    st.title("📚 PDF Processor Pro")
    st.write("Upload your PDF for AI-powered summarization and interactive Q&A")

    if 'processor' not in st.session_state:
        st.session_state.processor = PDFProcessor()
        
    if not st.session_state.processor.mistral_client.check_connection():
        st.error("❌ Cannot connect to Ollama service. Please ensure it's running.")
        return

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key='pdf_uploader')
    
    if uploaded_file:
        if not hasattr(st.session_state, 'processed_text') or not st.session_state.processed_text:
            with st.spinner("Processing PDF..."):
                text = st.session_state.processor.extract_text(uploaded_file)
                chunks = st.session_state.processor.text_splitter.split_text(text)
                
                progress_bar = st.progress(0)
                summaries = []
                
                # Process chunks with parallel processing
                chunk_groups = [chunks[i:i + 4] for i in range(0, len(chunks), 4)]
                for i, group in enumerate(chunk_groups):
                    group_summaries = st.session_state.processor.process_chunks_parallel(group)
                    summaries.extend(group_summaries)
                    progress_bar.progress((i + 1) / len(chunk_groups))
                
                st.session_state.summaries = summaries
                
                # Create and save summary PDF
                metadata = {
                    "Original File": uploaded_file.name,
                    "Processed Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Number of Sections": len(summaries),
                    "Processing Engine": "Mistral"
                }
                
                summary_pdf_path = st.session_state.processor.create_summary_pdf(summaries, metadata)
                st.session_state.summary_pdf_path = summary_pdf_path
                st.session_state.qa_system = create_qa_system(text)
                st.session_state.processed_text = True
                
                st.success("✅ Processing complete!")

        # Display summary section
        st.markdown("---")
        st.subheader("📑 Document Summary")
        
        # Summary display and download section
        col1, col2 = st.columns([3, 1])
        with col1:
            if hasattr(st.session_state, 'summaries'):
                for i, summary in enumerate(st.session_state.summaries, 1):
                    with st.expander(f"Section {i} Summary"):
                        st.write(summary)
        
        with col2:
            if hasattr(st.session_state, 'summary_pdf_path'):
                with open(st.session_state.summary_pdf_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Full Summary PDF",
                        data=file,
                        file_name=st.session_state.summary_pdf_path,
                        mime="application/pdf",
                        help="Download a PDF containing all section summaries"
                    )

        # Q&A Section
        st.markdown("---")
        st.subheader("❓ Ask Questions About Your Document")
        
        # Chat interface
        display_chat()
        
        # Create a form for the chat input
        with st.form(key="chat_form", clear_on_submit=True):
            question = st.text_input("Type your question:", key="question_input")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and question:
                handle_qa(question, st.session_state.qa_system)
                st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        logger.error(f"Application Error: {str(e)}")
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()