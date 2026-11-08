
"""
test_chatbot.py

Purpose:
    Test the complete RAG pipeline, including:
    PDF retrieval, Gemini answer generation and citations.
"""

# Import the FAISS loading and search functions.
from utils.vector_store import (
    load_vector_store,
    search_similar_chunks
)

# Import the Gemini answer generation function.
from utils.chatbot import generate_answer


# Step 1: Load the previously saved FAISS index.
vector_store = load_vector_store("vector_db")

print("FAISS vector store loaded successfully!")


# Step 2: Define the user's question.
query = "Which algorithm combines multiple decision trees?"


# Step 3: Retrieve the three most relevant document chunks.
results = search_similar_chunks(
    vector_store=vector_store,
    query=query,
    k=3
)


# Step 4: Extract Document objects from the search results.
# The search function returns (Document, distance score) tuples.
retrieved_docs = [
    document for document, score in results
]


# Step 5: Generate an answer using Gemini.
response = generate_answer(
    query=query,
    retrieved_docs=retrieved_docs
)


# Step 6: Display the user's question.
print("\nUser Question:")
print(query)


# Step 7: Display the generated answer.
print("\nAI Generated Answer:")
print(response["answer"])


# Step 8: Display the source references.
print("\nSources:")

for source in response["sources"]:

    print(
        f'{source["source_id"]}: '
        f'{source["source"]}, '
        f'Page {source["page"]}'
    )
