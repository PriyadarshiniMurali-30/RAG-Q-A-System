from rag_core import answer_question

if __name__ == "__main__":
    question = input("Ask a question about the document: ")
    result = answer_question(question)

    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Sources referenced ---")
    for i, chunk in enumerate(result["sources"]):
        print(f"[{i+1}] ({len(chunk.split())} words)")
        print(chunk)
        print()

# --- Highlighted versions (raw HTML, for debugging) ---
# if __name__ == "__main__":
#     question = input("Ask a question about the document: ")
#     result = answer_question(question)

#     print("\n--- Answer ---")
#     print(result["answer"])

#     print("\n--- Checking evidence matching ---")
#     for i, chunk in enumerate(result["highlighted_sources"]):
#         has_highlight = "<mark>" in chunk
#         print(f"[{i+1}] Highlight found: {has_highlight}")
#         if has_highlight:
#             # Print just the highlighted portion, not the whole chunk
#             start = chunk.find("<mark>")
#             end = chunk.find("</mark>") + len("</mark>")
#             print(f"    Highlighted text: {chunk[start:end]}")