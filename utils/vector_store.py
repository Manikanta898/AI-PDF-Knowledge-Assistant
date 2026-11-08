
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
    Generate embeddings for document chunks and store them
    in FAISS while preserving their metadata.

    Args:
        chunks (list):
            List of LangChain Document objects.

    Returns:
        FAISS:
            Searchable FAISS vector store.
    """

    # Validate that document chunks are available.
    if not chunks:
        raise ValueError("No document chunks provided.")

    # Retrieve the reusable Gemini embedding model.
    embedding_model = get_embedding_model()

    # Generate embeddings and create the FAISS index.
    # from_documents preserves page and filename metadata.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # Return the searchable vector store.
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
    results = vector_store.similarity_search_with_score(
    query=query,
    k=k
)

    # Return the retrieved documents.
    return results



import os


def save_vector_store(vector_store, path="vector_db"):
    """
    Save the FAISS vector store to local storage.

    Args:
        vector_store (FAISS):
            The FAISS index to save.

        path (str):
            Directory where the index will be stored.

    Returns:
        None
    """

    # Create the directory if it doesn't already exist.
    os.makedirs(path, exist_ok=True)

    # Save the FAISS index and its associated document data.
    vector_store.save_local(path)

    print(f"FAISS vector store saved successfully at: {path}")


def load_vector_store(path="vector_db"):
    """
    Load an existing FAISS vector store from local storage.

    Args:
        path (str):
            Directory containing the saved FAISS index.

    Returns:
        FAISS:
            Previously saved vector store.

    Raises:
        FileNotFoundError:
            If the saved index does not exist.
    """

    # Check whether the saved FAISS index exists.
    if not os.path.exists(os.path.join(path, "index.faiss")):
        raise FileNotFoundError("Saved FAISS index not found.")

    # Retrieve the existing Gemini embedding model.
    embedding_model = get_embedding_model()

    # Load the previously saved vector store.
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    # Return the loaded vector store.
    return vector_store
