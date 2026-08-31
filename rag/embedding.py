import os

from dotenv import load_dotenv
from google import genai


# --------------------------------
# Load environment variables
# --------------------------------

load_dotenv()


# --------------------------------
# Gemini client
# --------------------------------

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# --------------------------------
# Gemini embedding model
# --------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"


# --------------------------------
# Generate embeddings
# --------------------------------

def generate_embeddings(texts):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]