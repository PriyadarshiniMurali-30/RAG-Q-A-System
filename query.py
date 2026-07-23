import os
from dotenv import load_dotenv
from google import genai
from groq import Groq
import chromadb

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


def answer_question(question, top_k=3):
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

    # Return structured data instead of just printing
    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks
    }


if __name__ == "__main__":
    question = input("Ask a question about the document: ")
    result = answer_question(question)

    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Sources referenced ---")
    for i, chunk in enumerate(result["sources"]):
        print(f"[{i+1}] {chunk}\n")