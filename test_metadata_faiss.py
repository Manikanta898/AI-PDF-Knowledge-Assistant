
"""
test_metadata_faiss.py

Purpose:
    Test PDF extraction, metadata preservation,
    chunking, FAISS indexing and semantic retrieval.
"""

# Import the page-level PDF loader.
from utils.pdf_loader import load_pdf_pages

# Import the metadata-preserving text splitter.
from utils.text_splitter import split_documents

# Import FAISS operations.
from utils.vector_store import (
    create_vector_store,
    search_similar_chunks,
    save_vector_store
)


# Step 1: Extract PDF content page by page.
documents = load_pdf_pages("data/sample.pdf")

print("Total pages extracted:", len(documents))


# Step 2: Split page documents into smaller chunks.
chunks = split_documents(documents)

print("Total chunks created:", len(chunks))


# Step 3: Inspect the first chunk and its metadata.
print("\nFirst chunk metadata:")
print(chunks[0].metadata)


# Step 4: Create the FAISS vector store.
vector_store = create_vector_store(chunks)

print("\nFAISS vector store created successfully!")


# Step 5: Save the vector store locally.
save_vector_store(vector_store)


# Step 6: Define a sample question.
query = "Which algorithm combines multiple decision trees?"


# Step 7: Retrieve the most relevant chunks.
results = search_similar_chunks(
    vector_store=vector_store,
    query=query,
    k=3
)


# Step 8: Display content, distance and source metadata.
print("\nUser Question:", query)

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    # Display the retrieved content.
    print(document.page_content)

    # Display the FAISS distance.
    print(f"Distance Score: {score:.4f}")

    # Display the original PDF and page number.
    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))
