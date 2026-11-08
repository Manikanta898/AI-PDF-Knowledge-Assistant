# from langchain_text_splitters import RecursiveCharacterTextSplitter
# def split_text(text):
#     """
#     Split text into overlapping chunks for RAG.
#     Args:
#         text (str): Extracted text from PDF.
#     Returns:
#         list: List of text chunks.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks




"""
Split page-level Document objects into smaller chunks
while preserving their metadata.
"""

# Import LangChain's recursive text splitter.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split page documents into smaller overlapping chunks.

    Original page metadata is automatically preserved
    in every resulting chunk.

    Args:
        documents (list):
            List of LangChain Document objects.

        chunk_size (int):
            Maximum approximate number of characters per chunk.

        chunk_overlap (int):
            Number of overlapping characters between chunks.

    Returns:
        list:
            List of smaller Document objects with metadata.
    """

    # Initialize the recursive text splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Split documents while retaining their metadata.
    chunks = text_splitter.split_documents(documents)

    # Return the generated document chunks.
    return chunks
