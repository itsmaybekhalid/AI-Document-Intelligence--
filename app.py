"""AI Document Intelligence - local Streamlit application.

Run locally with:

    streamlit run app.py

Document processing, chunking, embeddings and FAISS retrieval all run on this
machine. Only prompts are sent to the Qwen3-14B API running on Kaggle
(see kaggle_api.py), and the answers come back through ngrok.
"""

import logging
import os
import tempfile

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from document import DocumentProcessingError, parse_document, split_document
from extraction import ExtractionError, extract_information
from rag import RetrievalError, answer_question, build_retriever, get_embeddings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (secrets live in .env, never in code)
# --------------------------------------------------------------------------- #
load_dotenv()

API_URL = os.getenv("KAGGLE_API_URL", "").strip().rstrip("/")
API_SECRET = os.getenv("API_SECRET", "").strip()
LLM_TIMEOUT_SECONDS = 300  # Qwen3-14B on a single GPU can be slow

# Summarisation settings
SUMMARY_TYPES = {
    "Short Summary": "Write a short summary of at most 80 words in one paragraph.",
    "Detailed Summary": (
        "Write a detailed summary of about 250 words that covers every "
        "important part of the text."
    ),
    "Key Points": (
        "Write 5 to 10 bullet points with the most important information. "
        "Start each bullet with '- '."
    ),
}
SUMMARY_CHUNK_SIZE = 4000  # characters per map step in the map-reduce summary


class ApiError(Exception):
    """Raised when the Kaggle Qwen3 API cannot be reached or fails."""


# --------------------------------------------------------------------------- #
# Kaggle API client (the only place that talks to the Qwen3 API)
# --------------------------------------------------------------------------- #
def call_kaggle_api(prompt: str, max_new_tokens: int = 500) -> str:
    """Send a prompt to Qwen3-14B on Kaggle and return the generated text."""
    if not API_URL:
        raise ApiError(
            "KAGGLE_API_URL is not set. Put the ngrok URL printed by "
            "kaggle_api.py into your .env file."
        )
    if not API_SECRET:
        raise ApiError("API_SECRET is not set. Put the API secret into your .env file.")

    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": int(max_new_tokens)},
            headers={"Authorization": f"Bearer {API_SECRET}"},
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise ApiError(
            "The Qwen3 API took too long to respond. Please try again."
        ) from None
    except requests.exceptions.RequestException:
        raise ApiError(
            "Could not reach the Kaggle API. Check KAGGLE_API_URL in .env and "
            "make sure the ngrok tunnel is still running."
        ) from None

    if response.status_code == 401:
        raise ApiError("The API secret was rejected. Check API_SECRET in .env.")
    if response.status_code == 503:
        raise ApiError(
            "The Qwen3 model is not ready on Kaggle yet. Wait a moment and try again."
        )
    if response.status_code != 200:
        raise ApiError(f"The Qwen3 API returned an error (HTTP {response.status_code}).")

    try:
        return str(response.json()["response"]).strip()
    except (ValueError, KeyError, TypeError):
        raise ApiError("The Qwen3 API returned an unexpected response.") from None


# --------------------------------------------------------------------------- #
# Summarisation (simple map-reduce)
# --------------------------------------------------------------------------- #
def _final_prompt(instructions: str, text: str) -> str:
    return (
        "You summarize documents.\n"
        f"{instructions}\n"
        "Use only the information in the text below. Do not invent facts.\n"
        "Write the summary in the same language as the document.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )


def _part_prompt(instructions: str, part: str) -> str:
    return (
        "You are summarizing one part of a longer document.\n"
        f"{instructions}\n"
        "Use only the information in the text below. Do not invent facts.\n\n"
        f"Text part:\n{part}\n\nSummary of this part:"
    )


def _reduce_prompt(instructions: str, partials: str) -> str:
    return (
        "Below are summaries of the parts of one document, in order.\n"
        "Combine them into one final summary. Remove repetitions and keep "
        "only the important information. Do not invent facts.\n"
        f"{instructions}\n\n"
        f"Part summaries:\n{partials}\n\nFinal summary:"
    )


def summarize_document(
    text: str,
    summary_type: str,
    llm=call_kaggle_api,
) -> str:
    """Summarize a document.

    Short texts are sent in one call. Longer texts use a simple map-reduce:
    each chunk is summarized, then the partial summaries are combined.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("There is no text to summarize.")

    instructions = SUMMARY_TYPES.get(summary_type, SUMMARY_TYPES["Short Summary"])
    chunks = split_document(
        Document(page_content=text),
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=100,
    )

    # Small document: one single call.
    if len(chunks) == 1:
        return llm(_final_prompt(instructions, text), 900)

    # Map: summarize every chunk.
    partials = []
    for chunk in chunks:
        partial = (llm(_part_prompt(instructions, chunk.page_content), 350) or "").strip()
        if partial:
            partials.append(partial)
    if not partials:
        raise ApiError("The Qwen3 API returned no summaries for this document.")

    # Reduce: combine the partial summaries into the final summary.
    combined = "\n\n".join(
        f"Part {i + 1}:\n{partial}" for i, partial in enumerate(partials)
    )
    return llm(_reduce_prompt(instructions, combined), 1200)


# --------------------------------------------------------------------------- #
# Cached work: the document, the embedding model, and the FAISS index
# --------------------------------------------------------------------------- #
@st.cache_data(max_entries=10, show_spinner=False)
def process_pdf(file_bytes: bytes, file_name: str) -> tuple[str, list[Document]]:
    """Parse and chunk the uploaded PDF once per file."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        document = parse_document(temp_path)
        document.metadata["source"] = file_name
        return document.page_content, split_document(document)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embeddings():
    return get_embeddings()


@st.cache_resource(max_entries=3, show_spinner="Building the FAISS index...")
def build_index(chunks: list[Document]):
    return build_retriever(chunks, load_embeddings())


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def run_action(action, spinner_text: str):
    """Run a slow action and show a human-readable error instead of a traceback."""
    try:
        with st.spinner(spinner_text):
            return action()
    except (
        ApiError,
        DocumentProcessingError,
        RetrievalError,
        ExtractionError,
        ValueError,
    ) as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Unexpected error")
        st.error(
            "Something went wrong. Check the terminal running Streamlit for details."
        )
    return None


def qa_section(chunks: list[Document]) -> None:
    """Document Q&A using RAG."""
    st.markdown("#### Ask a question about your document")

    with st.form("qa_form"):
        question = st.text_input(
            "Question",
            placeholder="What are the main objectives of this document?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Ask", type="primary", icon=":material/search:"
        )

    if not submitted:
        return
    if not question.strip():
        st.warning("Please type a question first.")
        return

    result = run_action(
        lambda: answer_question(question, build_index(chunks), call_kaggle_api),
        "Searching the document and generating an answer...",
    )
    if not result:
        return

    st.markdown("**Answer:**")
    st.write(result["answer"])

    if result["sources"]:
        with st.expander("Retrieved context"):
            for i, source in enumerate(result["sources"], start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(source)


def summarization_section(text: str) -> None:
    """Document summarisation."""
    st.markdown("#### Summarize your document")

    summary_type = st.segmented_control(
        "Summary Type",
        list(SUMMARY_TYPES),
        default="Short Summary",
    )

    if st.button("Generate Summary", type="primary", icon=":material/lightbulb:"):
        summary = run_action(
            lambda: summarize_document(text, summary_type or "Short Summary"),
            "Generating the summary... (long documents take a few minutes)",
        )
        if summary:
            st.markdown("**Summary:**")
            st.write(summary)


def extraction_section(text: str) -> None:
    """Structured information extraction."""
    st.markdown("#### Extract structured information")

    if st.button("Extract Information", type="primary", icon=":material/database:"):
        information = run_action(
            lambda: extract_information(text, call_kaggle_api),
            "Asking Qwen3 to extract and validate the information...",
        )
        if information is not None:
            st.markdown("**Structured Information:**")
            st.json(information.model_dump())


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="AI Document Intelligence",
        page_icon=":material/description:",
        layout="centered",
    )

    st.title("AI Document Intelligence")
    st.caption("Upload a PDF, then ask questions, summarize it, or extract information.")

    if not API_URL or not API_SECRET:
        st.info(
            "KAGGLE_API_URL and API_SECRET are not set in .env. "
            "Run kaggle_api.py on Kaggle and copy the values it prints."
        )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    feature = st.segmented_control(
        "Choose Feature",
        ["Document Q&A", "Summarization", "Information Extraction"],
        default="Document Q&A",
    )
    if feature is None:
        return

    if uploaded_file is None:
        st.caption("Upload a PDF file to enable the features.")
        return

    processed = run_action(
        lambda: process_pdf(uploaded_file.getvalue(), uploaded_file.name),
        "Reading the document with Docling...",
    )
    if processed is None:
        st.stop()

    text, chunks = processed
    st.success(f"Loaded **{uploaded_file.name}** - {len(chunks)} chunks ready.")

    if feature == "Document Q&A":
        qa_section(chunks)
    elif feature == "Summarization":
        summarization_section(text)
    else:
        extraction_section(text)


if __name__ == "__main__":
    main()
