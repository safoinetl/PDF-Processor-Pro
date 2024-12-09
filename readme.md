# 📚 PDF Processor Pro

PDF Processor Pro is a Streamlit-powered application that leverages AI for summarizing PDF documents and enabling interactive Q&A sessions. It uses the Mistral API for generating summaries and answering questions based on the document's content.

---

## Features
- **PDF Text Extraction**: Extracts text from uploaded PDF files.
- **AI-Powered Summarization**: Summarizes large documents into concise sections.
- **Interactive Q&A System**: Enables users to ask questions based on the document's content.
- **Summary PDF Generation**: Saves summarized content in a PDF format.
- **Parallel Processing**: Efficiently processes large documents using multithreading.
- **User-Friendly Interface**: Intuitive UI with expandable summaries and a chat-like Q&A system.

---

## How It Works
1. **Upload a PDF**: Upload your document via the file uploader.
2. **Summarize**: The application splits the text into chunks, summarizes each using the Mistral API, and generates a summary PDF.
3. **Q&A Interaction**: Users can ask questions about the document, and the application provides accurate responses based on the content.

---

## Installation

### Prerequisites
- Python 3.8+
- Pip

### Clone the Repository
```bash
git clone https://github.com/your-username/pdf-processor-pro.git
cd pdf-processor-pro
pip install -r requirements.txt
```
### run
```bash
streamlit run app.py
```
