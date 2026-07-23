from pypdf import PdfReader

def load_pdf_text(filepath):
    """Extract all text from a PDF file."""
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # step back a bit for overlap
    return chunks

if __name__ == "__main__":
    text = load_pdf_text("document.pdf")
    print(f"Extracted {len(text)} characters of text.\n")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.\n")

    print("--- First chunk preview ---")
    print(chunks[0][:300])