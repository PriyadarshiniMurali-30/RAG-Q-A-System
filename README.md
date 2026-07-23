# Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about any uploaded PDF using grounded, cited answers — built to avoid hallucination by only answering from retrieved source content.

**🔗 Live Demo**: [Add your Streamlit Cloud URL here]
**📂 Repo**: [Add your GitHub URL here]

## What it does

Upload any PDF through the web interface, ask questions in plain English, and get answers that are:
- **Grounded** — generated only from the document's actual content, not the model's general knowledge
- **Cited** — every claim is tagged with the exact source chunk it came from, so answers are independently verifiable
- **Evaluated** — accuracy is measured with an automated LLM-as-judge eval harness, not just eyeballed

## Demo

![Streamlit UI screenshot](screenshot.png)
*Upload a PDF, click "Process Document," then ask questions with cited, verifiable answers.*

*(Add a screenshot of your running app here — drag an image into the GitHub README editor, or save one as `screenshot.png` in the repo root.)*

## Architecture

```
PDF Upload (Streamlit UI)
    │
    ▼
Text Extraction (pypdf)
    │
    ▼
Chunking (500 words, 50-word overlap)
    │
    ▼
Embedding (Gemini gemini-embedding-001)
    │
    ▼
Vector Storage (ChromaDB)
    │
    ▼
[User Question] → Embed → Retrieve top-k similar chunks
    │
    ▼
Prompt Construction (numbered sources for citation)
    │
    ▼
Answer Generation (Groq / openai-gpt-oss-120b)
    │
    ▼
Cited Answer + Source References (displayed in UI)
```

Also exposed as a standalone REST API via FastAPI, with interactive docs at `/docs`.

## Key Engineering Decisions

- **Two-provider architecture**: Gemini handles embeddings, Groq handles generation. This split was a deliberate reliability choice — Groq's inference is fast and has strong uptime, while Gemini's free-tier embeddings are more than sufficient for this scale. It also demonstrates provider-agnostic design rather than lock-in to a single vendor.
- **Explicit citation requirement in the prompt**: The model is instructed to tag every claim with a source number (e.g., `[1]`), and the full source text is surfaced alongside the answer so claims can be manually verified — critical for any RAG system used in a context where trust matters.
- **LLM-as-judge evaluation**: Rather than brittle exact-string-match scoring (which fails on valid answers phrased differently, e.g. "4 years" vs. "four years"), a second LLM call independently judges whether the generated answer matches the expected answer in meaning.
- **Explicit groundedness testing**: The eval set includes an intentionally unrelated question ("What is the capital of France?") to verify the system correctly declines to answer rather than hallucinating from general knowledge.
- **Shared core module (`rag_core.py`)**: Ingestion and retrieval logic live in one place, reused identically by the CLI script, the FastAPI service, and the Streamlit frontend — avoids duplicated logic drifting out of sync.
- **Environment-aware secrets handling**: Config reads API keys from Streamlit Cloud's secrets manager when deployed, falling back to a local `.env` file during development — same codebase runs in both environments unmodified.

## Eval Results

**9/9 (100%)** on the initial test set, covering:
- Direct factual retrieval questions
- Numeric/date extraction
- An out-of-scope adversarial question (groundedness test)

**Caveat**: this is a small initial eval set. Future work includes expanding to 20-30 questions covering multi-hop reasoning (answers spanning multiple chunks) and ambiguous phrasing, before considering this production-validated.

## Tech Stack

- **Python**
- **Google Gemini API** — embeddings (`gemini-embedding-001`)
- **Groq API** — generation (`openai/gpt-oss-120b`)
- **ChromaDB** — vector database
- **FastAPI** — REST API layer
- **Streamlit** — web frontend, deployed on Streamlit Community Cloud
- **pypdf** — PDF text extraction

## Setup (local)

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Then open `http://localhost:8501`, upload a PDF, and start asking questions.

   Alternatively, use the CLI or API directly:
   ```bash
   python ingest.py        # ingest bigdoc.pdf via terminal
   python query.py         # ask questions via terminal
   uvicorn api:app --reload  # run as a REST API (docs at /docs)
   python run_eval.py      # run the evaluation suite
   ```

## Deployment

Deployed on **Streamlit Community Cloud**, connected directly to this GitHub repo.

- API keys are stored in Streamlit's secrets manager (never committed to source control)
- `rag_core.py` checks for Streamlit secrets first, falling back to `.env` locally

## Future Improvements

- Semantic chunking (split by document section/headers instead of fixed word count)
- Support for multiple documents with metadata filtering
- Streaming responses for faster perceived latency
- Larger, more adversarial eval set (multi-hop questions, ambiguous phrasing, conflicting sources)
- Persistent vector storage across sessions (currently re-ingests per upload)
