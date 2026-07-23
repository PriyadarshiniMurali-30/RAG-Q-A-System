import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Generate an embedding for a piece of text
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="QA Engineer with experience in automation testing"
)

embedding_vector = result.embeddings[0].values

print(f"✅ Embedding generated!")
print(f"Vector length: {len(embedding_vector)}")
print(f"First 5 numbers: {embedding_vector[:5]}")