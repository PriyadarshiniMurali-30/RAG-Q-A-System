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
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # No secrets.toml locally — that's expected, fall back to .env
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

def extract_evidence(answer_text):
    """
    Looks for lines like: [1]: "some exact quote"
    Returns a dictionary like: {1: "some exact quote", 2: "another quote"}
    """
    evidence = {}
    pattern = r'\[(\d+)\]:\s*"([^"]+)"'
    matches = re.findall(pattern, answer_text)
    for source_num, quote in matches:
        evidence[int(source_num)] = quote
    return evidence


def highlight_quote_in_chunk(chunk, quote):
    """
    If `quote` appears inside `chunk`, wraps it in <mark> tags so it renders highlighted.
    If no quote was found for this chunk, returns the chunk unchanged.
    """
    if not quote:
        return chunk
    pattern = re.escape(quote)
    return re.sub(f"({pattern})", r"<mark>\1</mark>", chunk, flags=re.IGNORECASE)

def answer_question(question, top_k=5):
    collection = chroma_client.get_or_create_collection(name="documents")

    # Step 1: Embed the question
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = result.embeddings[0].values

    # Step 2: Retrieve relevant chunks from Chroma
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    retrieved_chunks = results["documents"][0]

    # Step 3: Build a prompt with NUMBERED sources for citation
    numbered_context = "\n\n".join(
        [f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    prompt = f"""Answer the question using ONLY the numbered sources below.
After every claim, cite the source number it came from, like this: [1].
If different parts of your answer come from different sources, cite each one.
If the answer isn't in the sources, say "I don't know based on the provided document."

After your answer, add a section titled "EVIDENCE:" listing, for each source you cited,
the EXACT sentence (copied word-for-word, unchanged) from that source that supports your claim.
Format each line exactly like this: [1]: "exact sentence copied verbatim"

Sources:
{numbered_context}

Question: {question}

Answer (with citations):"""

    # Step 4: Generate the answer
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content


    # # Step 5: Split the main answer from the EVIDENCE section
    # if "EVIDENCE:" in answer:
    #     main_answer, evidence_section = answer.split("EVIDENCE:", 1)
    # else:
    #     main_answer, evidence_section = answer, ""

    # Step 5: Split the main answer from the EVIDENCE section (handles optional ** markdown)
    import re as re_module
    split_result = re_module.split(r'\*{0,2}EVIDENCE:\*{0,2}', answer, maxsplit=1)
    if len(split_result) == 2:
        main_answer, evidence_section = split_result
    else:
        main_answer, evidence_section = answer, ""
    main_answer = main_answer.strip()

    # Step 6: Parse evidence quotes and build highlighted chunk versions
    evidence_map = extract_evidence(evidence_section)
    highlighted_sources = []
    for i, chunk in enumerate(retrieved_chunks):
        quote = evidence_map.get(i + 1)
        highlighted_sources.append(highlight_quote_in_chunk(chunk, quote))

    return {
        "question": question,
        "answer": main_answer.strip(),
        "sources": retrieved_chunks,
        "highlighted_sources": highlighted_sources
    }