import requests
import streamlit as st

# ============================================
# FastAPI backend
# ============================================

from core.config import API_URL

# ============================================
# Call FastAPI backend
# ============================================

def ask_backend(question, chat_history):
    try:
        response = requests.post(
            API_URL,
            json={
                "question": question,
                "chat_history": chat_history
            },
            timeout=60
        )

    # ----------------------------------------
    # Successful response
    # ----------------------------------------

        if response.status_code == 200:

            return response.json()
    # ----------------------------------------
    # Gemini quota / rate limit
    # ----------------------------------------

        if response.status_code == 429:

            st.warning(
                "⏳ The AI service has temporarily "
                "reached its usage limit. Please try again later."
            )

            return None
        # ----------------------------------------
        # Bad request
        # ----------------------------------------

        if response.status_code == 400:

            st.error(
                "⚠️ Invalid request. Please check your question."
            )

            return None
         # ----------------------------------------
        # Other API errors
        # ----------------------------------------

        st.error(
            "⚠️ The document assistant is temporarily "
            "unavailable. Please try again."
        )

        return None
    # ----------------------------------------
    # FastAPI unavailable
    # ----------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "🔌 Cannot connect to the AI backend. "
            "Please make sure FastAPI is running."
        )

        return None
 # ----------------------------------------
    # Request timeout
    # ----------------------------------------

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The request took too long. "
            "Please try again."
        )

        return None
    # ----------------------------------------
    # Unexpected error
    # ----------------------------------------

    except Exception as e:

        print(
            f"Streamlit API Error: {str(e)}"
        )

        st.error(
            "⚠️ Something went wrong while "
            "processing your request."
        )

        return None






# ============================================
# Page configuration
# ============================================

st.set_page_config(
    page_title="Northstar Document Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================
# Initialize session state
# ============================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================
# Sidebar
# ============================================

with st.sidebar:

    st.title("🤖 Northstar")

    st.caption(
        "AI-Powered Company Document Assistant"
    )

    st.divider()


    # ----------------------------------------
    # Knowledge Base
    # ----------------------------------------

    st.subheader("📚 Knowledge Base")

    st.write(
        "Your assistant currently has access "
        "to the following documents:"
    )

    st.markdown(
        """
        📘 **Employee Handbook**

        📗 **Leave Policy**

        📙 **Travel Policy**
        """
    )


    st.divider()


    # ----------------------------------------
    # Knowledge Base Statistics
    # ----------------------------------------

    st.subheader("📊 Knowledge Base")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            "3"
        )

    with col2:

        st.metric(
            "Chunks",
            "16"
        )


    st.divider()


    # ----------------------------------------
    # New Conversation
    # ----------------------------------------

    if st.button(
        "🗑️ New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ----------------------------------------
    # How it works
    # ----------------------------------------

    with st.expander(
        "ℹ️ How this assistant works"
    ):

        st.markdown(
            """
            **1. Ask a question**

            Your question is converted into an embedding.

            **2. Semantic search**

            FAISS searches the company knowledge base
            for the most relevant document chunks.

            **3. Context retrieval**

            Relevant chunks are provided to Gemini.

            **4. Grounded answer**

            Gemini generates an answer using the
            retrieved company documents.

            **5. Sources**

            You can inspect the documents and
            retrieved content used for the answer.
            """
        )


    # ----------------------------------------
    # Technology
    # ----------------------------------------

    with st.expander(
        "🛠️ Technology"
    ):

        st.markdown(
            """
            - **Python**
            - **Streamlit**
            - **FAISS**
            - **Gemini Embeddings**
            - **Google Gemini**
            - **RAG**
            - **Semantic Search**
            """
        )


    st.divider()


    st.caption(
        "Northstar Technologies"
    )

    st.caption(
        "Powered by FAISS + Gemini"
    )


# ============================================
# Main Application
# ============================================

st.title(
    "🤖 Northstar Document Assistant"
)

st.caption(
    "Ask questions about company policies and "
    "documents using AI-powered semantic search."
)


# ============================================
# Welcome / Empty State
# ============================================

if not st.session_state.messages:

    st.info(
        """
        👋 **Welcome to Northstar Document Assistant**

        I can help you find information from the
        company's internal documents.

        **Available knowledge:**

        📘 Employee Handbook  
        📗 Leave Policy  
        📙 Travel Policy
        """
    )


    st.markdown(
        "### 💡 Try asking"
    )


    # ----------------------------------------
    # Suggested Questions
    # ----------------------------------------

    suggested_questions = [

        "How many vacation days do employees receive?",

        "How many sick days are available?",

        "Can employees work remotely?",

        "What is the company travel policy?"

    ]


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "🏖️ Vacation Policy",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggested_questions[0]
            )

            st.rerun()


        if st.button(
            "🏠 Remote Work",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggested_questions[2]
            )

            st.rerun()


    with col2:

        if st.button(
            "🤒 Sick Leave",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggested_questions[1]
            )

            st.rerun()


        if st.button(
            "✈️ Travel Policy",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggested_questions[3]
            )

            st.rerun()


# ============================================
# Display Existing Conversation
# ============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # ----------------------------------------
        # Display sources
        # ----------------------------------------

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            with st.expander(
                "📚 View Sources"
            ):

                st.caption(
                    "Documents retrieved from the knowledge base:"
                )


                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"### 📄 Source {i}"
                    )

                    st.write(
                        f"**Document:** "
                        f"{source['source']}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{source['page']}"
                    )

                    st.write(
                        f"**Distance:** "
                        f"{source['distance']:.4f}"
                    )


                    with st.expander(
                        "▶ View Retrieved Content"
                    ):

                        st.markdown(
                            source["text"]
                        )


# ============================================
# Chat Input
# ============================================

question = st.chat_input(
    "Ask a question about the company documents..."
)


# ============================================
# Handle Suggested Question
# ============================================

if (
    "pending_question" in st.session_state
    and not question
):

    question = st.session_state.pending_question

    del st.session_state.pending_question


# --------------------------------
# Process question
# --------------------------------

if question:

    # --------------------------------
    # Display user message
    # --------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------
    # Save user message
    # --------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # --------------------------------
    # Prepare chat history
    # --------------------------------

    chat_history = st.session_state.messages[:-1]


    # --------------------------------
    # Generate answer
    # --------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching company documents..."
        ):

            response = ask_backend(
                question,
                chat_history
            )


        # --------------------------------
        # Check API response
        # --------------------------------

        if response is not None:

            # --------------------------------
            # Extract answer and sources
            # --------------------------------

            answer = response["answer"]

            results = response["sources"]


            # --------------------------------
            # Display answer
            # --------------------------------

            st.markdown(answer)


            # --------------------------------
            # Display sources
            # --------------------------------

            if results:

                with st.expander(
                    "📚 View Sources"
                ):

                    st.caption(
                        "Documents used to generate this answer:"
                    )

                    for i, result in enumerate(
                        results,
                        start=1
                    ):

                        st.write(
                            f"**Source {i}:** "
                            f"📄 {result['source']} "
                            f"— Page {result['page']}"
                        )

                        if "text" in result:

                            with st.expander(
                                f"▶ View Retrieved Content {i}"
                            ):

                                st.write(
                                    result["text"]
                                )


            # --------------------------------
            # Save assistant message
            # --------------------------------

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer,

                "sources": results

            })