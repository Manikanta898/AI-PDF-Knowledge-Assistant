
"""
test_load_faiss.py

Purpose:
    Verify that a previously saved FAISS index
    can be loaded and searched without rebuilding it.
"""

# Import the FAISS loading and search functions.
from utils.vector_store import (
    load_vector_store,
    search_similar_chunks
)

# Load the previously saved FAISS index.
vector_store = load_vector_store("vector_db")

print("Existing FAISS index loaded successfully!")

# Define a question for semantic retrieval.
query = "Which algorithm combines multiple decision trees?"

# Search the saved vector store.
results = search_similar_chunks(
    vector_store=vector_store,
    query=query,
    k=3
)

# Display the retrieved results and their distances.
for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print(document.page_content)
    print(f"Distance Score: {score:.4f}")
