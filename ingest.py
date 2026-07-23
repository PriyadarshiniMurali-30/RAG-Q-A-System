import os
from dotenv import load_dotenv
from google import genai
import chromadb
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- Step 1: Load and chunk the PDF (same as before) ---
def load_pdf_text(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

# --- Step 2: Set up Chroma (our vector database) ---
#chroma_client = chromadb.PersistentClient(path="./chroma_db")
#collection = chroma_client.get_or_create_collection(name="documents")

# --- Step 2: Set up Chroma (our vector database) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Clear out any existing collection so we start fresh with the new document
try:
    chroma_client.delete_collection(name="documents")
except Exception:
    pass  # collection didn't exist yet, that's fine

collection = chroma_client.get_or_create_collection(name="documents")

# --- Step 3: Embed and store each chunk ---
def ingest_document(filepath):
    text = load_pdf_text(filepath)
    chunks = chunk_text(text)
    print(f"Loaded document, split into {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks):
        # Generate embedding for this chunk
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        embedding_vector = result.embeddings[0].values

        # Store chunk + its embedding in Chroma
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding_vector],
            documents=[chunk]
        )
        print(f"  Stored chunk {i+1}/{len(chunks)}")

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_document("bigdoc.pdf")