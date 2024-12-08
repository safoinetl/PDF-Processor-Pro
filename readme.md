# PDF Processor Pro

## Overview

**PDF Processor Pro** is a web-based application that allows users to upload PDF documents and interact with them through summarization and question answering (Q&A). The app extracts text from the uploaded PDF and processes it for two main features:

1. **Text Summarization**: The app splits the text into manageable chunks, summarizes the content, and displays it in an easily digestible format.
2. **Question Answering**: The app allows users to ask questions about the content of the document, and it returns relevant answers based on the document's contents.

This app is built using **Streamlit**, **HuggingFace** models, **LangChain**, and **FAISS** for document processing, and **PyPDF2**, **pdfplumber**, and **FPDF** for handling PDF documents.

### Features

- **Upload a PDF**: Upload a PDF document to interact with its content.
- **Summarization**: Get summaries of the content extracted from the PDF.
- **Q&A**: Ask specific questions, and the app will search through the document to provide relevant answers.
- **Text Processing**: The application uses advanced models to handle text processing and chunking for efficient summarization and question answering.

## Requirements

### Python Version

- Python 3.7+

### Dependencies

The following Python libraries are required to run the application:

- **streamlit**: For building the web app interface.
- **PyPDF2**: To handle PDF parsing.
- **fpdf**: To create PDFs, if needed.
- **langchain**: For text chunking and question-answering functionality.
- **langchain_community**: For utilizing HuggingFace embeddings and vector stores.
- **pdfplumber**: For extracting text from PDF files.
- **transformers**: For handling pre-trained models from HuggingFace.
- **dotenv**: For loading environment variables securely.

### Install Dependencies

To install the necessary dependencies, create a virtual environment and run the following:

```bash
# Create and activate a virtual environment (if not already done)
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```
