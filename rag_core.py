import os
import streamlit as st
import re
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


# def chunk_text(text, chunk_size=500, overlap=50):
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start = end - overlap
#     return chunks

def chunk_text(text, target_size=150, max_size=250):
    """
    Split text into semantically coherent chunks by paragraph,
    merging small paragraphs and splitting oversized ones by sentence.

    target_size: aim for roughly this many words per chunk
    max_size: hard ceiling before we force-split a paragraph
    """
    # Step 1: split into paragraphs (blank-line separated)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk_words = []

    def flush_current():
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

    for para in paragraphs:
        para_words = para.split()

        # If a single paragraph alone is too big, split it by sentences
        if len(para_words) > max_size:
            flush_current()
            current_chunk_words = []

            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_buffer = []
            for sentence in sentences:
                sentence_buffer.extend(sentence.split())
                if len(sentence_buffer) >= target_size:
                    chunks.append(" ".join(sentence_buffer))
                    sentence_buffer = []
            if sentence_buffer:
                chunks.append(" ".join(sentence_buffer))
            continue

        # Otherwise, keep merging paragraphs until we hit target_size
        if len(current_chunk_words) + len(para_words) > max_size:
            flush_current()
            current_chunk_words = para_words
        else:
            current_chunk_words.extend(para_words)

    flush_current()
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


def answer_question(question, top_k=5):
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