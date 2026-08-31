import os
import pickle

import faiss
import numpy as np
from google import genai
from core.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    TOP_K,
    DISTANCE_THRESHOLD
)
from rag.embedding import generate_embeddings
from core.logger import logger

# # ============================================
# # 1. Load environment variables
# # ============================================

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# GEMINI_MODEL = os.getenv(
#     "GEMINI_MODEL",
#     "gemini-3.6-flash"
# )

# if not GOOGLE_API_KEY:
#     raise ValueError(
#         "GOOGLE_API_KEY not found. "
#         "Please add it to your .env file."
#     )


# ============================================
# 2. Create Gemini client
# ============================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================
# 3. Load FAISS index
# ============================================

INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"


index = faiss.read_index(
    INDEX_PATH
)

print(
    f"FAISS vectors loaded: {index.ntotal}"
)


# ============================================
# 4. Load document chunks
# ============================================

with open(
    CHUNKS_PATH,
    "rb"
) as f:

    chunks = pickle.load(f)


print(
    f"Chunks loaded: {len(chunks)}"
)


# ============================================
# 5. Format conversation history
# ============================================

def format_chat_history(chat_history):

    if not chat_history:

        return "No previous conversation."


    history_text = []

    for message in chat_history:

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":

            history_text.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history_text.append(
                f"Assistant: {content}"
            )


    return "\n".join(history_text)


# ============================================
# 6. Rewrite contextual question
# ============================================

def rewrite_question(
    question,
    chat_history
):

    # ----------------------------------------
    # No history
    # ----------------------------------------

    if not chat_history:

        return question


    # ----------------------------------------
    # Format history
    # ----------------------------------------

    history = format_chat_history(
        chat_history
    )


    # ----------------------------------------
    # Prompt Gemini to rewrite the question
    # ----------------------------------------

    prompt = f"""
You are a query rewriting assistant for a
company document search system.

Your task is to rewrite the user's current
question into a standalone search query that
can be used to search company documents.

Use the previous conversation to understand
references such as:

- that
- it
- they
- them
- this
- those
- what about
- explain that
- tell me more

Do NOT answer the question.

Only rewrite the question.

If the current question is already standalone,
return it unchanged.

Previous Conversation:
----------------------

{history}

----------------------

Current User Question:
{question}

----------------------

Standalone Search Query:
"""


    # ----------------------------------------
    # Call Gemini
    # ----------------------------------------

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )


    rewritten_question = response.text.strip()


    # ----------------------------------------
    # Safety fallback
    # ----------------------------------------

    if not rewritten_question:

        return question


    return rewritten_question


# ============================================
# 7. Search documents
# ============================================

def search_documents(
    question,
    top_k=3
):
    logger.info(
    "Document search started"
    )

    # ----------------------------------------
    # Convert question into Gemini embedding
    # ----------------------------------------

    question_embedding = generate_embeddings(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )


    # ----------------------------------------
    # Search FAISS
    # ----------------------------------------

    distances, indices = index.search(
        question_embedding,
        top_k
    )
    logger.info(
    "FAISS search completed. Requested top_k=%d",
    top_k
    )


    # ----------------------------------------
    # Build results
    # ----------------------------------------

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        # Safety check
        if index_position < 0:
            continue

        chunk = chunks[index_position]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "distance": float(distance)
        })
    logger.info(
        "Retrieved %d document chunks",
        len(results)
    )

    for result in results:

        logger.info(
            "Retrieved source=%s page=%s distance=%.4f",
            result["source"],
            result["page"],
            result["distance"]
        )

    return results


# ============================================
# 8. Generate answer
# ============================================

def generate_answer(
    question,
    chat_history=None
):

    # ----------------------------------------
    # Make sure history exists
    # ----------------------------------------

    if chat_history is None:

        chat_history = []


    # ========================================
    # Query rewriting
    # ========================================

    search_query = rewrite_question(
        question,
        chat_history
    )


    print("\n")
    print("=============================")
    print("ORIGINAL QUESTION")
    print("=============================")

    print(question)


    print("\n")
    print("=============================")
    print("REWRITTEN SEARCH QUERY")
    print("=============================")

    print(search_query)


    # ========================================
    # Retrieve relevant documents
    # ========================================

    results = search_documents(
        search_query,
        top_k=TOP_K
    )


    # ========================================
    # Distance filtering
    # ========================================

    

    filtered_results = [
        result
        for result in results
        if result["distance"] <= DISTANCE_THRESHOLD
    ]
    logger.info(
    "Distance filtering: %d/%d chunks passed threshold %.2f",
    len(filtered_results),
    len(results),
    DISTANCE_THRESHOLD
    )

    results = filtered_results

    # ========================================
    # If nothing relevant was found
    # ========================================

    if not results:
        logger.warning(
        "No relevant document chunks found"
        )

        return (
            "I couldn't find that information "
            "in the company documents.",
            []
        )


    # ========================================
    # Build document context
    # ========================================

    context = "\n\n".join(
        [
            f"Source: {result['source']}, "
            f"Page: {result['page']}\n"
            f"{result['text']}"
            for result in results
        ]
    )


    # ========================================
    # Format conversation history
    # ========================================

    history = format_chat_history(
        chat_history
    )


    # ========================================
    # Create RAG prompt
    # ========================================

    prompt = f"""
You are a company document assistant.

Answer the user's question using ONLY the
information provided in the company documents.

Do not use outside knowledge.

Previous conversation is provided only to
understand the context of the current question.

The actual answer must come ONLY from the
company documents.

If the answer is not present in the documents,
say exactly:

"I couldn't find that information in the
company documents."

Be concise, accurate, and professional.

========================================
PREVIOUS CONVERSATION
========================================

{history}

========================================
RETRIEVED COMPANY DOCUMENTS
========================================

{context}

========================================
CURRENT USER QUESTION
========================================

{question}

========================================
ANSWER
========================================
"""


    # ========================================
    # Call Gemini
    # ========================================

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )


    # ========================================
    # Get answer
    # ========================================

    answer = response.text


    # ========================================
    # Return answer and sources
    # ========================================

    return answer, results


# ============================================
# 9. Test the RAG system
# ============================================

if __name__ == "__main__":

    question = (
        "How many vacation days do employees receive?"
    )


    print("\n")
    print("=============================")
    print("QUESTION")
    print("=============================")

    print(question)


    answer, results = generate_answer(
        question
    )


    print("\n")
    print("=============================")
    print("ANSWER")
    print("=============================")

    print(answer)


    print("\n")
    print("=============================")
    print("SOURCES")
    print("=============================")

    for result in results:

        print(
            f"{result['source']} | "
            f"Page {result['page']} | "
            f"Distance {result['distance']}"
        )