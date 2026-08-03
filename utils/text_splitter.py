from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_text(text):
    """
    Split text into overlapping chunks for RAG.
    Args:
        text (str): Extracted text from PDF.
    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks