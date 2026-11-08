
"""
vector_store.py

Purpose:
    Create and search a FAISS vector index.

Role in RAG:
    Stores document embeddings and retrieves relevant
    document chunks based on semantic similarity.
"""

# Import FAISS integration from LangChain Community.
from langchain_community.vectorstores import FAISS

# Import our reusable Gemini embedding model.
from utils.embeddings import get_embedding_model


def create_vector_store(chunks):
    """
    Convert document chunks into embeddings and
    store them in a FAISS vector index.

    Args:
        chunks (list):
            List of text chunks extracted from a PDF.

    Returns:
        FAISS:
            A searchable FAISS vector store.

    Raises:
        ValueError:
            If no document chunks are provided.
    """

    # Check whether the input contains any chunks.
    if not chunks:
        raise ValueError("No chunks provided to create the vector store.")

    # Get the initialized Gemini embedding model.
    embedding_model = get_embedding_model()

    # Generate embeddings for all chunks and create the FAISS index.
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    # Return the completed vector store.
    return vector_store


def search_similar_chunks(vector_store, query, k=3):
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        vector_store (FAISS):
            Previously created FAISS vector store.

        query (str):
            User's question or search query.

        k (int):
            Number of relevant chunks to retrieve.
            Defaults to 3.

    Returns:
        list:
            List of LangChain Document objects containing
            the retrieved text and associated metadata.
    """

    # Validate the search query.
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    # Retrieve the top-k most similar document chunks.
    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    # Return the retrieved documents.
    return results
