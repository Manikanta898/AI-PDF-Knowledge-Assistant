# AI PDF Knowledge Assistant

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions about their content through an interactive Streamlit interface.

The application uses **Google Gemini** to generate answers and **FAISS** to retrieve relevant information from uploaded documents.

## Features

- Upload PDF documents through a Streamlit interface.
- Extract text from PDFs while preserving page-level metadata.
- Split extracted text into smaller chunks.
- Generate semantic embeddings using Google's Gemini embedding model.
- Store and retrieve document embeddings using FAISS.
- Ask questions about uploaded PDF documents.
- Generate contextual answers using Gemini.
- Display source references, including PDF filenames and page numbers.
- Save the FAISS vector store locally for reuse.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| LangChain | Document processing and LLM integration |
| Google Gemini | Embeddings and answer generation |
| FAISS | Vector storage and similarity search |
| python-dotenv | Environment variable management |

## Project Architecture

```text
User uploads PDF
       |
       v
   PDF Loader
       |
       v
  Text Extraction
       |
       v
  Text Splitter
       |
       v
 Embedding Model
       |
       v
   FAISS Vector Store
       |
       v
 User asks a question
       |
       v
 Query Embedding
       |
       v
 Similarity Search
       |
       v
 Relevant Document Chunks
       |
       v
 Google Gemini LLM
       |
       v
 Contextual Answer + Sources
```

## Project Structure

```text
AI-PDF-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
├── data/
│
├── uploaded_files/
│
├── vector_db/
│
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── chatbot.py
│
└── assets/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-PDF-Knowledge-Assistant.git
cd AI-PDF-Knowledge-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Google API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Get an API key through [Google AI Studio](https://aistudio.google.com/).

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Launch the Streamlit application.
2. Upload a PDF document.
3. Click **Process PDF**.
4. Wait for the document extraction, chunking, embedding, and indexing process to complete.
5. Enter a question in the chat input.
6. View the generated answer and its source references.

## How RAG Works in This Project

The application follows two main stages:

**1. Document Indexing**

- Extract text from the uploaded PDF.
- Divide the text into smaller chunks.
- Generate embeddings for the chunks.
- Store the embeddings and document metadata in FAISS.

**2. Question Answering**

- Convert the user's question into an embedding.
- Retrieve the most similar document chunks from FAISS.
- Pass the retrieved context and question to Gemini.
- Generate an answer based on the provided document context.
- Display the answer with source references.

## Configuration

The current implementation uses:

| Component | Configuration |
|---|---|
| Embedding model | `models/gemini-embedding-001` |
| Chat model | `gemini-3.8-flash` |
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |
| Retrieved chunks | 3 |

These settings can be adjusted in the respective utility modules.

## Current Limitations

- Text extraction is designed for text-based PDFs; scanned PDFs may require OCR.
- The application currently processes one active document at a time.
- Retrieval returns the top three nearest chunks without a relevance threshold.
- Conversation history is retained for display during the current session, but previous questions are not included as context for subsequent answers.
- Advanced document management and production-level error handling are not yet implemented.

## Future Improvements

- Add OCR support for scanned PDFs.
- Introduce retrieval relevance filtering.
- Support multiple PDF documents.
- Add conversational memory.
- Improve source citation accuracy.
- Enhance the user interface and error handling.
- Deploy the application to a cloud platform.

## Author

**Manikanta Tangi**

A project developed to explore Generative AI, Large Language Models, vector databases, and Retrieval-Augmented Generation using Python.

---

*Built with Python, LangChain, FAISS, Google Gemini, and Streamlit.*
