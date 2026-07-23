import streamlit as st
import tempfile
import os
from rag_core import ingest_document, answer_question

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG Demo)")
st.caption("Upload a PDF, then ask questions grounded in its content — with citations.")

# --- Session state: track whether a document has been ingested ---
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# --- Step 1: File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Reading, chunking, and embedding your document..."):
            # Save uploaded file to a temporary path so pypdf can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            num_chunks = ingest_document(tmp_path)
            os.remove(tmp_path)  # clean up the temp file

            st.session_state.ingested = True
            st.success(f"✅ Document processed into {num_chunks} chunks. Ready for questions!")

# --- Step 2: Question input (only show once a document is ingested) ---
if st.session_state.ingested:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Thinking..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("📚 View source chunks"):
            for i, chunk in enumerate(result["sources"]):
                st.markdown(f"**[{i+1}]**")
                st.text(chunk)
else:
    st.info("👆 Upload and process a PDF to get started.")