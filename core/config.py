import os

from dotenv import load_dotenv


# ============================================
# Load environment variables
# ============================================

load_dotenv()


# ============================================
# Gemini configuration
# ============================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================
# RAG configuration
# ============================================

TOP_K = int(
    os.getenv(
        "TOP_K",
        "3"
    )
)

DISTANCE_THRESHOLD = float(
    os.getenv(
        "DISTANCE_THRESHOLD",
        "1.2"
    )
)


# ============================================
# API configuration
# ============================================

API_URL = os.getenv(
    "API_URL",
    "http://api:8000/ask"
)


# ============================================
# Validate required configuration
# ============================================

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Please add it to your .env file."
    )