"""RAG pipeline.

Pipeline:
    chunks -> embeddings -> FAISS -> retriever -> relevant context
           -> RAG prompt -> Kaggle Qwen3 API -> answer

This file does not talk to the LLM itself. The caller passes a simple
`llm(prompt, max_new_tokens) -> str` function (app.py provides one that calls
the Kaggle API), which keeps the retrieval logic easy to test in isolation.
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
# Swap this constant to use a different embedding model.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# How many chunks are retrieved per question.
TOP_K = 4

# intfloat/multilingual-e5-* models expect these prefixes (they were trained
# with them). Set both to "" if you switch to a model that does not use them.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Returned when the retrieved context does not contain the answer.
NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded document."

# Type of the LLM function supplied by the caller.
LLMCall = Callable[[str, int], str]


class RetrievalError(Exception):
    """Raised when embeddings, FAISS, or similarity search fail."""


RAG_PROMPT = PromptTemplate.from_template(
    """You are a careful assistant that answers questions using ONLY the document \
context below.

Rules:
- Use only the information found in the context.
- Do not use outside knowledge. Do not invent, guess, or add information.
- If the answer is not present in the context, reply exactly: {not_found}
- Answer in the same language as the question (English or Arabic).

Context:
{context}

Question: {question}

Answer:"""
)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class MultilingualE5Embeddings(HuggingFaceEmbeddings):
    """Adds the 'query:' / 'passage:' prefixes required by E5 models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"{PASSAGE_PREFIX}{text}" for text in texts])

    def embed_query(self, text: str) -> list[list[float]]:
        return super().embed_query(f"{QUERY_PREFIX}{text}")


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model.

    E5 models need prefixes for good retrieval quality; other models do not.
    Vectors are normalised so FAISS's L2 distance ranks like cosine similarity,
    which is what E5 models are trained for.
    """
    kwargs = {
        "model_name": model_name,
        "encode_kwargs": {"normalize_embeddings": True},
    }
    if "e5" in model_name.lower():
        return MultilingualE5Embeddings(**kwargs)
    return HuggingFaceEmbeddings(**kwargs)


# --------------------------------------------------------------------------- #
# Vector store and retriever
# --------------------------------------------------------------------------- #
def build_retriever(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings | None = None
):
    """Embed the chunks, build a FAISS index, and return a retriever."""
    if not chunks:
        raise RetrievalError("There are no chunks to index.")

    if embeddings is None:
        embeddings = get_embeddings()

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RetrievalError(
            "Could not build the FAISS index. The embedding model may have "
            "failed to encode the document."
        ) from exc

    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def retrieve(retriever, question: str) -> list[Document]:
    """Return the top chunks most similar to the question."""
    try:
        return list(retriever.invoke(question))
    except Exception as exc:
        raise RetrievalError("Similarity search failed while finding context.") from exc


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_context(chunks: list[Document]) -> str:
    """Join retrieved chunks into one labelled context block."""
    return "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.page_content}" for i, chunk in enumerate(chunks)
    )


def build_rag_prompt(question: str, context: str) -> str:
    """Render the RAG prompt from the question and the retrieved context."""
    return RAG_PROMPT.format(
        question=question,
        context=context,
        not_found=NOT_FOUND_MESSAGE,
    )


# --------------------------------------------------------------------------- #
# End-to-end question answering
# --------------------------------------------------------------------------- #
def answer_question(
    question: str,
    retriever,
    llm: LLMCall,
    max_new_tokens: int = 600,
) -> dict:
    """Answer a question from the document.

    Returns {"answer": str, "sources": list[str]} where sources are the
    retrieved chunks the answer was based on.
    """
    chunks = retrieve(retriever, question)

    # Nothing relevant was found: do not call the model at all.
    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    prompt = build_rag_prompt(question, build_context(chunks))
    answer = (llm(prompt, max_new_tokens) or "").strip()

    if not answer:
        answer = NOT_FOUND_MESSAGE

    return {"answer": answer, "sources": [c.page_content for c in chunks]}
