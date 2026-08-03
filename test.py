from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text

text = extract_text_from_pdf("data/sample.pdf")

chunks = split_text(text)

print(f"Total Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print("=" * 50)
    print(f"Chunk {i+1}")
    print(chunk)