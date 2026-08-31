import time
from fastapi import FastAPI, HTTPException
from api.models import AskRequest, AskResponse
from rag.pipeline import generate_answer
from core.logger import logger


# ============================================
# Create FastAPI application
# ============================================

app = FastAPI(
    title="Northstar Document Assistant API",
    description="Backend API for the Northstar RAG chatbot",
    version="1.0.0"
)


# ============================================
# Root endpoint
# ============================================

@app.get("/")
def root():

    return {
        "message": "Northstar Document Assistant API is running"
    }


# ============================================
# Health check
# ============================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================
# Ask endpoint
# ============================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(request: AskRequest):
    start_time = time.perf_counter()

    logger.info(
    "API request received"
    )

    logger.info(
    "Processing question. Length=%d characters",
    len(request.question)
    )
    try:

        # ----------------------------------------
        # Convert chat history
        # ----------------------------------------

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        logger.info(
            "Chat history messages: %d",
            len(chat_history)
        )

        # ----------------------------------------
        # Generate RAG answer
        # ----------------------------------------

        answer, results = generate_answer(
            request.question,
            chat_history=chat_history
        )

        elapsed_time = time.perf_counter() - start_time

        logger.info(
            "RAG completed. Retrieved %d sources",
             len(results)
        )
        logger.info(
            "Request completed in %.2f seconds",
            elapsed_time
        )
        # ----------------------------------------
        # Build sources
        # ----------------------------------------

        sources = []

        for result in results:

            sources.append({

                "source": result["source"],

                "page": result["page"],

                "distance": result["distance"],

                "text": result["text"]

            })


        # ----------------------------------------
        # Return successful response
        # ----------------------------------------

        logger.info(
            "API request completed successfully"
        )
        return {

            "question": request.question,

            "answer": answer,

            "sources": sources

        }


    except Exception as e:

        error_message = str(e)

        elapsed_time = time.perf_counter() - start_time

        logger.exception(
            "RAG API Error after %.2f seconds",
            elapsed_time
        )


    # ----------------------------------------
    # Gemini quota / rate limit
    # ----------------------------------------

    if (
        "quota" in error_message.lower()
        or "rate limit" in error_message.lower()
        or "resource exhausted" in error_message.lower()
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                "The AI service has temporarily reached "
                "its usage limit. Please try again later."
            )
        )


    # ----------------------------------------
    # General backend error
    # ----------------------------------------

    raise HTTPException(
        status_code=500,
        detail=(
            "Unable to process the request. "
            "Please try again later."
        )
    )