# AI PDF Knowledge Assistant

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions about their content through an interactive Streamlit interface.

The application uses **Google Gemini** for embedding generation and answer generation, while **FAISS** retrieves relevant information from uploaded documents.

## Features

- Upload PDF documents through a Streamlit interface.
- Extract text from PDFs using PyMuPDF.
- Preserve page-level metadata for source references.
- Split documents into smaller text chunks.
- Generate semantic embeddings using Google Gemini.
- Store and retrieve embeddings using FAISS.
- Ask questions about uploaded PDF documents.
- Generate contextual answers using Gemini.
- Display source references with PDF filenames and page numbers.
- Save and reload the FAISS vector store locally.

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

The application follows a Retrieval-Augmented Generation (RAG) architecture.

```text
             PDF Upload
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
                 |
          User Question
                 |
                 v
           Query Embedding
                 |
                 v
          Similarity Search
                 |
                 v
       Retrieved Document Chunks
                 |
                 v
          Google Gemini LLM
                 |
                 v
        Answer with References
```

## Project Structure

```text
AI-PDF-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── chatbot.py
│
├── test.py
├── test_chatbot.py
├── test_embeddings.py
├── test_faiss.py
├── test_load_faiss.py
└── test_metadata_faiss.py
```

**Note:** The `uploaded_files/` and `vector_db/` directories are generated locally when the application processes documents. They are not required to be committed to the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Manikanta898/AI-PDF-Knowledge-Assistant.git

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

### 4. Configure Google API Key

Create a `.env` file in the project root directory.

```env
GOOGLE_API_KEY=your_google_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

**Important:** Keep your API key private. Never commit your `.env` file to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Run the Streamlit application.
2. Upload a PDF document.
3. Click **Process PDF**.
4. The application extracts text, creates chunks, generates embeddings, and builds the FAISS index.
5. Enter your question in the chat input.
6. View the AI-generated answer and its source references.

## How RAG Works

The application operates in two main stages.

### 1. Document Indexing

- Extract text from the uploaded PDF.
- Split the extracted text into smaller chunks.
- Generate embeddings for each chunk.
- Store the embeddings and metadata in FAISS.

### 2. Question Answering

- Convert the user's question into an embedding.
- Retrieve the three most similar document chunks from FAISS.
- Pass the retrieved context and question to Gemini.
- Generate a contextual answer using the document content.
- Display the answer with source references.

## Model Configuration

| Component | Model / Setting |
|---|---|
| Embedding Model | `models/gemini-embedding-001` |
| Chat Model | `gemini-3.8-flash` |
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Retrieved Chunks | 3 |

## Current Limitations

- Designed primarily for text-based PDFs; scanned documents may require OCR.
- Processes one active PDF document at a time.
- Retrieves the top three nearest chunks without a relevance threshold.
- Previous chat messages are displayed but are not passed as conversational context to Gemini.
- Advanced document management and production-level error handling are not yet implemented.

## Future Enhancements

- OCR support for scanned PDFs.
- Retrieval relevance filtering.
- Support for multiple PDF documents.
- Conversational memory.
- Improved source citation accuracy.
- Enhanced user interface and error handling.
- Cloud deployment.

## Author

**Manikanta Tangi**

This project was developed to explore Generative AI, Large Language Models, vector databases, and Retrieval-Augmented Generation using Python.

**Technologies:** Python | Streamlit | LangChain | FAISS | Google Gemini
