import os
from dotenv import load_dotenv
from google import genai

# Load the API key from our .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API key found. Check your .env file.")
    exit()

# Create the client
client = genai.Client(api_key=api_key)

# Try a simple generation call to test the connection
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Say hello in one short sentence."
)

print("✅ Connection successful!")
print("Gemini says:", response.text)