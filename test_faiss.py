
"""
test_faiss.py

Purpose:
    Test the complete embedding and FAISS retrieval process.

Workflow:
    PDF -> Text Extraction -> Chunking -> Embeddings
        -> FAISS Index -> Save Vector Store -> Semantic Search
"""

# Import the PDF extraction function.
from utils.pdf_loader import extract_text_from_pdf

# Import the text splitting function.
from utils.text_splitter import split_text

# Import FAISS creation, saving and search functions.
from utils.vector_store import (
    create_vector_store,
    save_vector_store,
    search_similar_chunks
)


# Step 1: Extract text from the sample PDF.
text = extract_text_from_pdf("data/sample.pdf")


# Step 2: Split the extracted text into chunks.
chunks = split_text(text)

print(f"Total chunks created: {len(chunks)}")


# Step 3: Create the FAISS vector store.
vector_store = create_vector_store(chunks)

print("FAISS vector store created successfully!")


# Step 4: Save the vector store to local storage.
save_vector_store(vector_store)

print("FAISS vector store saved successfully!")


# Step 5: Define a question related to the PDF.
query = "Which algorithm combines multiple decision trees?"


# Step 6: Retrieve the three most relevant chunks.
results = search_similar_chunks(
    vector_store=vector_store,
    query=query,
    k=3
)


# Step 7: Display the retrieved chunks.
print("\nUser Question:", query)
print("\nRetrieved Results:")

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    # Display the retrieved document content.
    print(document.page_content)

    # Display the FAISS distance score.
    print(f"Distance Score: {score:.4f}")

    # Display document metadata.
    print("Metadata:", document.metadata)
