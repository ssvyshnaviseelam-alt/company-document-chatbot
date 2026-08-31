from pydantic import BaseModel, Field


# ============================================
# Chat Message
# ============================================

class ChatMessage(BaseModel):

    role: str

    content: str


# ============================================
# Ask Request
# ============================================

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question asked by the user"
    )

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages"
    )


# ============================================
# Source Response
# ============================================

class Source(BaseModel):

    source: str

    page: int

    distance: float

    text: str


# ============================================
# Ask Response
# ============================================

class AskResponse(BaseModel):

    question: str

    answer: str

    sources: list[Source]