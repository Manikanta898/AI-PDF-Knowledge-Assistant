import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file page by page.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined text from all pages.
    """

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text