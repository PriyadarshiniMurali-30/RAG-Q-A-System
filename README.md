# Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about any uploaded PDF using grounded, cited answers — built to avoid hallucination by only answering from retrieved source content.

**🔗 Live Demo**: [https://rag-qna-pdf.streamlit.app/]
**📂 Repo**: [https://github.com/PriyadarshiniMurali-30/RAG-Q-A-System/]

## What it does

Upload any PDF through the web interface, ask questions in plain English, and get answers that are:
- **Grounded** — generated only from the document's actual content, not the model's general knowledge
- **Cited** — every claim is tagged with the exact source chunk it came from
- **Evidence-highlighted** — the specific sentence(s) supporting each claim are highlighted within the source chunk, so answers are verifiable at a glance, not just at the chunk level
- **Evaluated** — accuracy is measured with an automated LLM-as-judge eval harness, not just eyeballed

## Demo

<img width="1572" height="882" alt="RAG-qa" src="https://github.com/user-attachments/assets/2c21357e-e1ff-41f8-a9ce-92a90ff47f42" />


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
Answer + Chunk Citations + Highlighted Evidence Quotes (displayed in UI)
```

Also exposed as a standalone REST API via FastAPI, with interactive docs at `/docs`.

## Key Engineering Decisions

- **Two-provider architecture**: Gemini handles embeddings, Groq handles generation. This split was a deliberate reliability choice — Groq's inference is fast and has strong uptime, while Gemini's free-tier embeddings are more than sufficient for this scale. It also demonstrates provider-agnostic design rather than lock-in to a single vendor.
- **Semantic chunking over fixed word-count chunking**: chunks are split along paragraph and sentence boundaries (target ~150 words, hard ceiling 250) rather than blindly cutting every N words. This keeps each chunk focused on one coherent idea, making citations meaningfully easier to verify — the tradeoff being more chunks retrieved per query to maintain the same coverage.
- **Explicit citation + evidence-quote requirement in the prompt**: the model tags every claim with a source number (`[1]`) and separately outputs the exact verbatim sentence used as evidence. That quote is then located within the original chunk and highlighted in the UI — moving citation from "here's roughly where this came from" to "here's the exact sentence."
- **LLM-as-judge evaluation**: Rather than brittle exact-string-match scoring (which fails on valid answers phrased differently, e.g. "4 years" vs. "four years"), a second LLM call independently judges whether the generated answer matches the expected answer in meaning.
- **Explicit groundedness testing**: The eval set includes an intentionally unrelated question ("What is the capital of France?") to verify the system correctly declines to answer rather than hallucinating from general knowledge.
- **Shared core module (`rag_core.py`)**: Ingestion and retrieval logic live in one place, reused identically by the CLI script, the FastAPI service, and the Streamlit frontend — avoids duplicated logic drifting out of sync.
- **Environment-aware secrets handling**: Config reads API keys from Streamlit Cloud's secrets manager when deployed, falling back to a local `.env` file during development — same codebase runs in both environments unmodified.

## Eval Results

**10/10 (100%)** on the initial test set, covering:
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

## Known Limitations

- **Evidence quote highlighting is best-effort**: citations reliably identify the correct source chunk, but sentence-level highlighting depends on the LLM reproducing a quote verbatim. It occasionally fails to match, or highlights an adjacent-but-related sentence instead of the most precise one. Chunk-level citation accuracy is unaffected even when sentence-level highlighting doesn't trigger.
- PDF text extraction doesn't always preserve clean paragraph breaks, so chunking sometimes falls back to sentence-level splitting on dense/tabular content.
- Vector store resets per session on Streamlit Cloud — by design, since the app expects a fresh upload each use rather than a persisted document.

## Future Improvements

- Support for multiple documents with metadata filtering
- Larger, more adversarial eval set (multi-hop questions, ambiguous phrasing, conflicting sources)
- Persistent vector storage across sessions (currently re-ingests per upload)

## Notes

`archive/` contains early standalone scripts written while learning each concept individually (PDF extraction, embeddings, generation), before consolidating shared logic into `rag_core.py`.
