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
from typing import List, Callable
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get model name and API key from environment variables
HF_MODEL_NAME = os.getenv('HF_MODEL_NAME')
HF_API_KEY = os.getenv('HF_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CustomPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_font("Arial", size=12)
        
    def sanitize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if ord(c) < 128)
        return text

    def chapter_title(self, title):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, ln=True, align='C')
        self.ln(10)
        self.set_font("Arial", size=12)

    def chapter_body(self, text):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln()

class PDFProcessor:
    def __init__(self):
        try:
            # Use the model name from the environment variable
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            self.embeddings = HuggingFaceEmbeddings()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @staticmethod
    def extract_text(pdf_file) -> str:
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
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise

    def summarize_chunk(self, chunk: str) -> str:
        try:
            if len(chunk) < 500:
                return chunk
            
            inputs = self.tokenizer.encode(
                "summarize: " + chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            outputs = self.model.generate(
                inputs,
                max_length=300,
                min_length=100,
                length_penalty=1.5,
                num_beams=6,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return chunk

    def create_qa_system(self, text: str) -> Callable:
        try:
            chunks = self.create_chunks(text)
            vectorstore = FAISS.from_texts(chunks, self.embeddings)
            qa_chain = load_qa_chain(HuggingFaceHub(
                repo_id=HF_MODEL_NAME,  # Use model name from the .env file
                huggingfacehub_api_token=HF_API_KEY
            ))
            
            def qa_function(question: str) -> str:
                try:
                    return qa_chain.run(
                        input_documents=vectorstore.similarity_search(question, k=5),
                        question=question
                    )
                except Exception as e:
                    logger.error(f"Error in QA system: {str(e)}")
                    return "Sorry, I couldn't process your question. Please try rephrasing it."
            
            return qa_function
        except Exception as e:
            logger.error(f"Error creating QA system: {str(e)}")
            raise

def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "processed_text" not in st.session_state:
        st.session_state.processed_text = False

def add_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("Assistant: " + message["content"])

def main():
    st.set_page_config(page_title="PDF Processor Pro", layout="wide")
    initialize_chat_history()
    st.title("PDF Processor Pro")
    st.write("Upload your PDF for summarization and Q&A")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            if not st.session_state.processed_text:
                processor = PDFProcessor()
                with st.spinner("Extracting text from PDF..."):
                    text = processor.extract_text(uploaded_file)
                    if not text.strip():
                        st.error("No text could be extracted from the PDF. Please ensure it's not a scanned document.")
                        return
                st.info(f"Successfully extracted {len(text)} characters.")
                st.session_state.qa_system = processor.create_qa_system(text)
                st.session_state.processed_text = True
            
            st.subheader("Chat with your document")
            display_chat()
            question = st.text_input("Ask a question:")
            if st.button("Send") and question:
                add_message("user", question)
                with st.spinner("Thinking..."):
                    answer = st.session_state.qa_system(question)
                    add_message("assistant", answer)
                st.rerun()
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
