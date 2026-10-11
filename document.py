"""Document processing.

Pipeline:
    PDF -> Docling -> extracted Markdown/text -> LangChain Document -> chunks

This file only knows about Docling and text splitting. Everything else in the
project works with plain text and LangChain Documents.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Chunking settings
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000      # maximum characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# Docling loads heavy parsing models, so the converter is created only once.
_converter: DocumentConverter | None = None


class DocumentProcessingError(Exception):
    """Raised when a PDF cannot be turned into usable text."""


def _get_converter() -> DocumentConverter:
    """Return a shared DocumentConverter instance."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def parse_document(file_path) -> Document:
    """Convert a PDF file into a single LangChain Document.

    Raises DocumentProcessingError with a human readable message when the file
    is missing, empty, corrupted, or contains no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError("The uploaded file could not be found.")
    if path.stat().st_size == 0:
        raise DocumentProcessingError("The uploaded PDF file is empty (0 bytes).")

    try:
        result = _get_converter().convert(str(path))
    except Exception as exc:  # Docling raises several different error types
        raise DocumentProcessingError(
            "Docling could not parse this PDF. The file may be corrupted, "
            "password protected, or not a real PDF."
        ) from exc

    docling_document = getattr(result, "document", None)
    markdown = docling_document.export_to_markdown() if docling_document else ""
    text = (markdown or "").strip()

    if not text:
        raise DocumentProcessingError(
            "No text could be extracted from this PDF. It may be a scanned "
            "or image-only document without OCR text."
        )

    return Document(page_content=text, metadata={"source": path.name})


def split_document(
    document: Document,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split a LangChain Document into overlapping chunks for retrieval."""
    if not document.page_content.strip():
        raise DocumentProcessingError("The document contains no text to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents([document])
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise DocumentProcessingError("The document could not be split into chunks.")

    return chunks
