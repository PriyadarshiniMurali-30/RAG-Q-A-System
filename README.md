# Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about a PDF document using grounded, cited answers — built to avoid hallucination by only answering from retrieved source content.

## What it does

Upload a PDF, ask questions in plain English, and get answers that are:
- **Grounded** — generated only from the document's actual content, not the model's general knowledge
- **Cited** — every claim is tagged with the exact source chunk it came from, so answers are independently verifiable
- **Evaluated** — accuracy is measured with an automated LLM-as-judge eval harness, not just eyeballed

## Architecture

```
PDF Document
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
Cited Answer + Source References
```

Exposed as a REST API via FastAPI, with interactive docs at `/docs`.

## Key Engineering Decisions

- **Two-provider architecture**: Gemini handles embeddings, Groq handles generation. This split was a deliberate reliability choice — Groq's inference is fast and has strong uptime, while Gemini's free-tier embeddings are more than sufficient for this scale. It also demonstrates provider-agnostic design rather than lock-in to a single vendor.
- **Explicit citation requirement in the prompt**: The model is instructed to tag every claim with a source number (e.g., `[1]`), and the full source text is surfaced alongside the answer so claims can be manually verified — critical for any RAG system used in a context where trust matters.
- **LLM-as-judge evaluation**: Rather than brittle exact-string-match scoring (which fails on valid answers phrased differently, e.g. "4 years" vs. "four years"), a second LLM call independently judges whether the generated answer matches the expected answer in meaning.
- **Explicit groundedness testing**: The eval set includes an intentionally unrelated question ("What is the capital of France?") to verify the system correctly declines to answer rather than hallucinating from general knowledge.

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
- **pypdf** — PDF text extraction

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

3. Add your PDF as `bigdoc.pdf` in the project root, then ingest it:
   ```bash
   python ingest.py
   ```

4. Run the API server:
   ```bash
   uvicorn api:app --reload
   ```

5. Test it interactively at `http://127.0.0.1:8000/docs`, or via the terminal:
   ```bash
   python query.py
   ```

6. Run the evaluation suite:
   ```bash
   python run_eval.py
   ```

## Future Improvements

- Semantic chunking (split by document section/headers instead of fixed word count)
- Support for multiple documents with metadata filtering
- Streaming responses for faster perceived latency
- Larger, more adversarial eval set (multi-hop questions, ambiguous phrasing, conflicting sources)
