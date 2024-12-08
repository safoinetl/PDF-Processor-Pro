import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import pdfplumber
import os
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import unicodedata
import logging
from typing import List, Callable, Dict, Any
import tempfile
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CustomPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font("Arial", size=12)
        
    def header(self):
        """Add custom header to each page"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Document Summary', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        """Add custom footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def sanitize_text(self, text: str) -> str:
        """Sanitize text for PDF output"""
        text = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in text if ord(c) < 128)

    def chapter_title(self, title):
        """Add a chapter title with consistent formatting"""
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, ln=True, align='L')
        self.ln(5)
        self.set_font("Arial", size=12)

    def chapter_body(self, text):
        """Add chapter body text with proper formatting"""
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln()

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with required models and configurations"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len
            )
            self.embeddings = HuggingFaceEmbeddings()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @staticmethod
    def extract_text(pdf_file) -> str:
        """Extract text from PDF file with improved error handling"""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def create_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks with error handling"""
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise

    def summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk of text with improved handling and longer outputs"""
        try:
            if len(chunk) < 500:
                return chunk
            
            inputs = self.tokenizer.encode(
                "summarize: " + chunk,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            outputs = self.model.generate(
                inputs,
                max_length=300,  # Increased for longer summaries
                min_length=100,  # Increased minimum length
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,  # Added temperature for more natural text
                no_repeat_ngram_size=3  # Prevent repetition
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return chunk

    def summarize_chunks(self, chunks: List[str], progress_callback: Callable = None) -> List[str]:
        """Summarize chunks in parallel with progress tracking"""
        try:
            summaries = []
            total_chunks = len(chunks)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {executor.submit(self.summarize_chunk, chunk): i 
                                 for i, chunk in enumerate(chunks)}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        summary = future.result()
                        summaries.append(summary)
                        if progress_callback:
                            progress_callback((chunk_index + 1) / total_chunks)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
            
            return [s for s in summaries if s.strip()]
        except Exception as e:
            logger.error(f"Error in parallel summarization: {str(e)}")
            raise

    def create_summary_pdf(self, summaries: List[str], metadata: Dict[str, Any] = None) -> str:
        """Create PDF with summaries using built-in fonts and return the filename"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"summary_{timestamp}.pdf"
            
            pdf = CustomPDF()
            pdf.add_page()
            
            # Add metadata if provided
            if metadata:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Document Information", ln=True)
                pdf.set_font("Arial", size=12)
                for key, value in metadata.items():
                    pdf.cell(0, 10, f"{key}: {value}", ln=True)
                pdf.ln(10)
            
            pdf.chapter_title("Executive Summary")
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(10)
            
            for i, summary in enumerate(summaries, 1):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Section {i}", ln=True)
                pdf.set_font("Arial", size=12)
                sanitized_summary = pdf.sanitize_text(summary)
                pdf.chapter_body(sanitized_summary)
                pdf.ln(5)
            
            pdf.output(output_filename)
            return output_filename
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise

    def create_qa_system(self, text: str) -> Callable:
        """Create question-answering system with improved response quality"""
        try:
            chunks = self.create_chunks(text)
            vectorstore = FAISS.from_texts(chunks, self.embeddings)
            qa_chain = load_qa_chain(
                HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token="azerazeazer_api_token",
                    model_kwargs={
                        "temperature": 0.7,
                        "max_length": 1024,  # Increased for longer responses
                        "min_length": 100,  # Set minimum length
                        "num_beams": 4,
                        "no_repeat_ngram_size": 3
                    }
                ),
                chain_type="map_reduce",  # Changed to map_reduce for better handling of long contexts
                verbose=True
            )
            
            def qa_function(question: str) -> str:
                try:
                    # Get relevant documents with increased context
                    docs = vectorstore.similarity_search(question, k=5)  # Increased k for more context
                    
                    # Prepare context from documents
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Generate response
                    response = qa_chain.run(
                        input_documents=docs,
                        question=question
                    )
                    
                    # Post-process and validate response
                    if len(response.strip()) < 50:  # Increased minimum response length
                        return ("I apologize, but I need more context to provide a detailed answer. "
                               "Could you please rephrase your question or provide more specifics?")
                    
                    # Add confidence statement if response seems uncertain
                    if any(word in response.lower() for word in ['maybe', 'might', 'possibly']):
                        response += ("\n\nNote: This response is based on my current understanding of the document. "
                                   "Please verify critical information in the original text.")
                    
                    return response.strip()
                    
                except Exception as e:
                    logger.error(f"Error in QA system: {str(e)}")
                    return ("I apologize, but I encountered an error processing your question. "
                           "Please try rephrasing it or ask another question.")
            
            return qa_function
        except Exception as e:
            logger.error(f"Error creating QA system: {str(e)}")
            raise

def initialize_chat_history():
    """Initialize session state variables for chat functionality"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "processed_text" not in st.session_state:
        st.session_state.processed_text = False
    if "summaries" not in st.session_state:
        st.session_state.summaries = []
    if "summary_pdf_path" not in st.session_state:
        st.session_state.summary_pdf_path = None

def add_message(role: str, content: str):
    """Add a message to the chat history with improved formatting"""
    content = content.strip()
    if not content:
        return
        
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def display_chat():
    """Display the chat history with improved formatting and styling"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <div style='color: #666666; font-size: 0.8em; margin-bottom: 5px;'>
                        {message["timestamp"]}
                    </div>
                    <div>
                        <strong>You:</strong> {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color: #e8f4ea; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <div style='color: #666666; font-size: 0.8em; margin-bottom: 5px;'>
                        {message["timestamp"]}
                    </div>
                    <div>
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="PDF Processor Pro", layout="wide")
    
    initialize_chat_history()
    
    st.title("📚 PDF Processor Pro")
    st.write("Upload your PDF for summarization and interactive Q&A")
    
    # Add CSS for better styling
    st.markdown("""
        <style>
            .stTextInput > div > div > input {
                padding: 15px;
                border-radius: 10px;
            }
            .stButton > button {
                padding: 10px 25px;
                border-radius: 10px;
            }
            div[data-testid="stMarkdownContainer"] > div {
                margin: 10px 0;
            }
            .upload-section {
                padding: 20px;
                border-radius: 10px;
                border: 2px dashed #cccccc;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # File upload section
    with st.container():
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            # Only process the document if it hasn't been processed yet
            if not st.session_state.processed_text:
                processor = PDFProcessor()
                
                with st.spinner("📄 Extracting text from PDF..."):
                    text = processor.extract_text(uploaded_file)
                    if not text.strip():
                        st.error("❌ No text could be extracted from the PDF. Please ensure it's not a scanned document.")
                        return

                st.success(f"✅ Successfully extracted {len(text):,} characters.")
                
                with st.spinner("🔄 Processing document..."):
                    chunks = processor.create_chunks(text)
                    st.info(f"📊 Created {len(chunks)} text chunks for processing.")
                    
                    progress_bar = st.progress(0)
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    summaries = processor.summarize_chunks(chunks, update_progress)
                    st.session_state.summaries = summaries
                    
                    st.success(f"✅ Successfully summarized {len(summaries)} sections.")
                    
                    # Create and save summary PDF with metadata
                    metadata = {
                        "Original File": uploaded_file.name,
                        "Processed Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Number of Sections": len(summaries)
                    }
                    
                   # Continuing from the previous code...
                    
                    summary_pdf_path = processor.create_summary_pdf(summaries, metadata)
                    st.session_state.summary_pdf_path = summary_pdf_path
                    
                    # Create Q&A system and store in session state
                    st.session_state.qa_system = processor.create_qa_system(text)
                    st.session_state.processed_text = True

            # Display summary section
            st.markdown("---")
            st.subheader("📑 Document Summary")
            
            # Summary display and download section
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.session_state.summaries:
                    for i, summary in enumerate(st.session_state.summaries, 1):
                        with st.expander(f"Section {i} Summary"):
                            st.write(summary)
            
            with col2:
                if st.session_state.summary_pdf_path:
                    with open(st.session_state.summary_pdf_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Full Summary PDF",
                            data=file,
                            file_name=st.session_state.summary_pdf_path,
                            mime="application/pdf",
                            help="Download a PDF containing all section summaries"
                        )
                    
                    # Add a button to generate a shorter version
                    if st.button("📄 Generate Concise Summary"):
                        with st.spinner("Generating concise summary..."):
                            # Create a shorter version of the summaries
                            concise_summaries = [s[:200] + "..." for s in st.session_state.summaries]
                            concise_pdf_path = processor.create_summary_pdf(
                                concise_summaries,
                                {**metadata, "Version": "Concise"}
                            )
                            
                            with open(concise_pdf_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Concise Summary",
                                    data=file,
                                    file_name=f"concise_{concise_pdf_path}",
                                    mime="application/pdf",
                                    help="Download a shorter version of the summary"
                                )

            # Chat interface
            st.markdown("---")
            st.subheader("💬 Chat with your Document")
            
            # Display existing chat history
            display_chat()
            
            # Question input with improved layout
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_input(
                    "Ask a detailed question about your document:",
                    key="question_input",
                    placeholder="Type your question here... (e.g., 'What are the main topics discussed in this document?')"
                )
            with col2:
                send_button = st.button("Send 📤", key="send_button")
            
            if send_button and question:
                # Add user question to chat history
                add_message("user", question)
                
                # Get answer using cached qa_system
                with st.spinner("🤔 Analyzing document and generating response..."):
                    answer = st.session_state.qa_system(question)
                    if answer:
                        add_message("assistant", answer)
                
                # Clear the input
                st.session_state.question_input = ""
                st.rerun()
            
            # Chat management buttons
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    if st.session_state.chat_history:
                        st.session_state.chat_history = []
                        st.rerun()
            
            # Add session information
            with st.expander("ℹ️ Session Information"):
                st.write(f"Documents Processed: {1 if st.session_state.processed_text else 0}")
                st.write(f"Number of Summaries: {len(st.session_state.summaries)}")
                st.write(f"Chat Messages: {len(st.session_state.chat_history)}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logging.error(f"Application error: {str(e)}")
            
            # Provide recovery options
            if st.button("🔄 Reset Application"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()