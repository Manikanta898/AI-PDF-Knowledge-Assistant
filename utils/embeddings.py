# the below code is to run separately to verify that your API key is loaded correctly

# import os
# from dotenv import load_dotenv

# load_dotenv() #reads variables from your .env file.

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") #retrieves the value of a particular environment variable.

# print("API Key loaded:", GOOGLE_API_KEY is not None)


"""
embeddings.py

Purpose:
    Generate numerical embeddings from text using Google's
    Gemini embedding model.

Role in RAG:
    Converts document chunks and user questions into vectors
    so that semantic similarity search can be performed.
"""

# Import os to access environment variables.
import os

# Load environment variables from the .env file.
from dotenv import load_dotenv

# Import Google's embedding integration for LangChain.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from functools import lru_cache

# Load variables defined in the project's .env file.
load_dotenv()

@lru_cache(maxsize=1)#lru_cache remembers the returned model object. Subsequent calls reuse the same initialized instance instead of creating a new one.

#This is useful because initializing clients repeatedly introduces unnecessary overhead.

def get_embedding_model():
    """
    Initialize and return Google's embedding model.

    Returns:
        GoogleGenerativeAIEmbeddings:
            An initialized embedding model that converts
            text into numerical vectors.

    Raises:
        ValueError:
            If GOOGLE_API_KEY is missing from the environment.
    """

    # Retrieve the Google API key without exposing it in the code.
    api_key = os.getenv("GOOGLE_API_KEY")

    # Stop execution if the API key is not configured.
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please configure it in your .env file."
        )

    # Initialize Google's embedding model.
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Return the initialized model for reuse.
    return embedding_model


def generate_embedding(text):
    """
    Generate an embedding for a single text input.

    Args:
        text (str):
            Text whose semantic representation needs to be generated.

    Returns:
        list:
            A numerical vector representing the meaning of the text.
    """

    # Initialize the embedding model.
    embedding_model = get_embedding_model()

    # Convert the input text into its numerical representation.
    embedding = embedding_model.embed_query(text)

    # Return the generated vector.
    return embedding
