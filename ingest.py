from rag_core import ingest_document

if __name__ == "__main__":
    num_chunks = ingest_document("bigdoc.pdf")
    print(f"✅ Ingested into {num_chunks} chunks.")