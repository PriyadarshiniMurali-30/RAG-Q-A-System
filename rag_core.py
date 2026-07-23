import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from groq import Groq
import chromadb
from pypdf import PdfReader

load_dotenv()

def get_secret(key):
    """Read from Streamlit secrets if available (cloud), else from .env (local)."""
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

gemini_client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
groq_client = Groq(api_key=get_secret("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")


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


def ingest_document(filepath):
    """Wipe old data, then chunk + embed + store the new document."""
    try:
        chroma_client.delete_collection(name="documents")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="documents")

    text = load_pdf_text(filepath)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        embedding_vector = result.embeddings[0].values
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding_vector],
            documents=[chunk]
        )

    return len(chunks)


def answer_question(question, top_k=3):
    collection = chroma_client.get_or_create_collection(name="documents")

    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    retrieved_chunks = results["documents"][0]

    numbered_context = "\n\n".join(
        [f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    prompt = f"""Answer the question using ONLY the numbered sources below.
After every claim, cite the source number it came from, like this: [1].
If different parts of your answer come from different sources, cite each one.
If the answer isn't in the sources, say "I don't know based on the provided document."

Sources:
{numbered_context}

Question: {question}

Answer (with citations):"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved_chunks
    }