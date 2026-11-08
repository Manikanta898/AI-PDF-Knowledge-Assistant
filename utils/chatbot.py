
"""
chatbot.py

Purpose:
    Connect the retrieved PDF content with Google's Gemini LLM
    and generate answers grounded in the uploaded documents.

Role in RAG:
    Receives relevant chunks from FAISS, combines them with
    the user's question, and generates a natural-language answer.
"""

# Import os to access environment variables.
import os

# Import lru_cache to reuse the initialized model.
from functools import lru_cache

# Import load_dotenv to load credentials from the .env file.
from dotenv import load_dotenv

# Import LangChain's Google Gemini chat model.
from langchain_google_genai import ChatGoogleGenerativeAI

# Import message classes to structure the LLM conversation.
from langchain_core.messages import SystemMessage, HumanMessage


# Load environment variables from the project's .env file.
load_dotenv()


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Initialize and return the Gemini chat model.

    Returns:
        ChatGoogleGenerativeAI:
            An initialized Gemini chat model.

    Raises:
        ValueError:
            If the Google API key is unavailable.
    """

    # Retrieve the Google API key from environment variables.
    api_key = os.getenv("GOOGLE_API_KEY")

    # Validate the API key before initializing the model.
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    # Initialize Gemini using the Google API.
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    # Return the initialized model.
    return chat_model


def generate_answer(query, retrieved_docs):
    """
    Generate an answer using the user's question and
    relevant document chunks retrieved from FAISS.

    Args:
        query (str):
            Question asked by the user.

        retrieved_docs (list):
            List of LangChain Document objects retrieved
            from the FAISS vector store.

    Returns:
        dict:
            Contains the generated answer and source references.

    Raises:
        ValueError:
            If the question or retrieved documents are empty.
    """

    # Validate the user's question.
    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")

    # Validate the retrieved document collection.
    if not retrieved_docs:
        raise ValueError("No relevant documents were retrieved.")

    # Create an empty list to hold formatted document context.
    context_parts = []

    # Create an empty list to hold source information.
    sources = []

    # Process every retrieved document.
    for index, document in enumerate(retrieved_docs, start=1):

        # Retrieve the original PDF filename.
        source_name = document.metadata.get("source", "Unknown source")

        # Retrieve the original PDF page number.
        page_number = document.metadata.get("page", "Unknown page")

        # Create a unique reference label for this retrieved chunk.
        source_id = f"S{index}"

        # Format the document content with its source information.
        context_parts.append(
            f"[{source_id}: {source_name}, Page {page_number}]\n"
            f"{document.page_content}"
        )

        # Store source information separately for the application.
        sources.append({
            "source_id": source_id,
            "source": source_name,
            "page": page_number
        })

    # Combine all retrieved document chunks into one context string.
    context = "\n\n".join(context_parts)

    # Retrieve the initialized Gemini chat model.
    chat_model = get_chat_model()

    # Define instructions that control the model's behaviour.
    system_prompt = """
    You are an AI PDF Knowledge Assistant.

    Your task is to answer questions using only the
    information provided in the document context.

    Follow these instructions:

    1. Answer using only the supplied document context.
    2. Do not use external knowledge.
    3. Do not invent facts or unsupported information.
    4. If the answer is unavailable in the context,
       clearly say that you could not find the answer
       in the uploaded document.
    5. Keep your answers clear and easy to understand.
    6. Cite supporting sources using their reference IDs,
       such as [S1] or [S2].
    """

    # Create the user's message containing the context and question.
    human_prompt = f"""
    Document Context:

    {context}

    User Question:

    {query}

    Answer the question using the document context.
    Include the relevant source reference IDs.
    """

    # Send the system instructions and user message to Gemini.
    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    # Extract the generated answer from the model response.
    # Extract readable text from Gemini's response.
    if isinstance(response.content, str):
        answer = response.content
    else:
        answer = "\n".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
    )

    # Return the answer and its associated source information.
    return {
        "answer": answer,
        "sources": sources
    }
