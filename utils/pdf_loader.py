# import fitz  # PyMuPDF

# def extract_text_from_pdf(pdf_path):
#     """
#     Extracts text from a PDF file page by page.
#     Args:
#         pdf_path (str): Path to the PDF file.

#     Returns:
#         str: Combined text from all pages.
#     """

#     document = fitz.open(pdf_path)

#     text = ""

#     for page in document:
#         text += page.get_text() + "\n" # new line preserves page separation, making later processing cleaner

#     document.close()

#     return text






"""
pdf_loader.py

Additional functionality:
    Extract PDF content page by page while preserving
    filename and page number metadata.
"""

# Import os to extract the PDF filename.
import os

# Import PyMuPDF for reading PDF files.
import fitz

# Import LangChain Document for structured text storage.
from langchain_core.documents import Document


def load_pdf_pages(pdf_path):
    """
    Extract text from a PDF and return one Document per page.

    Args:
        pdf_path (str):
            Path to the PDF file.

    Returns:
        list:
            List of LangChain Document objects containing
            page text and metadata.
    """

    # Create an empty list to store extracted page documents.
    documents = []

    # Extract the PDF filename from its path.
    pdf_name = os.path.basename(pdf_path)

    # Open the PDF safely using a context manager.
    with fitz.open(pdf_path) as pdf:

        # Iterate through every page in the PDF.
        for page_number, page in enumerate(pdf, start=1):

            # Extract text from the current page.
            page_text = page.get_text()

            # Skip pages that contain no extractable text.
            if not page_text.strip():
                continue

            # Create a structured Document with its metadata.
            document = Document(
                page_content=page_text,
                metadata={
                    "source": pdf_name,
                    "page": page_number
                }
            )

            # Add the page document to our collection.
            documents.append(document)

    # Return all extracted page documents.
    return documents
