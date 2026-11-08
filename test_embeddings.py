
"""
test_embeddings.py

Purpose:
    Test whether the Gemini embedding integration works
    and inspect the generated embedding vector.
"""

# Import the function that generates embeddings.
from utils.embeddings import generate_embedding


# Define a sample sentence for testing.
sample_text = "Random Forest combines multiple decision trees."

# Generate the numerical representation of the sentence.
embedding = generate_embedding(sample_text)

# Display the number of dimensions in the vector.
print("Embedding dimensions:", len(embedding))

# Display the first 10 numerical values.
print("First 10 values:", embedding[:10])

# Confirm that an embedding was generated successfully.
print("Embedding generated successfully!")
