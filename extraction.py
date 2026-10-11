"""Structured information extraction.

Pipeline:
    document text -> Qwen3 prompt -> JSON string -> Pydantic validation
                  -> DocumentInformation

Change the fields of DocumentInformation below to extract different data:
the prompt and the validation are both generated from that schema.
"""

import ast
import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

# Type of the LLM function supplied by the caller: llm(prompt, max_new_tokens).
LLMCall = Callable[[str, int], str]

MAX_NEW_TOKENS = 900
MAX_RETRIES = 2  # extra attempts when the model returns invalid JSON


class DocumentInformation(BaseModel):
    """Structured data extracted from the uploaded document."""

    title: str | None = None
    document_type: str | None = None
    main_topic: str | None = None
    key_points: list[str] = []
    important_entities: list[str] = []


class ExtractionError(Exception):
    """Raised when the model output cannot be turned into valid data."""


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _field_spec() -> str:
    """Describe the required fields, derived from the Pydantic model."""
    return "\n".join(
        f"- {name}: {field.annotation}"
        for name, field in DocumentInformation.model_fields.items()
    )


def build_extraction_prompt(document_text: str) -> str:
    """Ask the model for a JSON object matching the Pydantic schema."""
    return (
        "You extract structured information from a document.\n"
        "Return ONLY one JSON object with exactly these fields:\n"
        f"{_field_spec()}\n\n"
        "Rules:\n"
        "- Every value must come from the document. Never copy text from these "
        "instructions.\n"
        "- The object you return IS the final result: do not wrap it inside "
        "'properties' or any other key.\n"
        "- If a value is not present in the document, use null for strings and "
        "an empty list for lists. Never invent information.\n"
        "- Do not write any text before or after the JSON, and do not use "
        "markdown code fences.\n\n"
        f"Document:\n\"\"\"\n{document_text}\n\"\"\"\n\n"
        "JSON output:"
    )


def build_correction_prompt(document_text: str, previous_answer: str, error: str) -> str:
    """Ask the model to fix a response that failed parsing or validation."""
    return (
        "Your previous answer could not be parsed as the required JSON.\n"
        f"Error: {error}\n\n"
        f"Previous answer:\n{previous_answer}\n\n"
        "Return ONLY the corrected JSON object with exactly these fields:\n"
        f"{_field_spec()}\n\n"
        "Remember: the object you return IS the final result, every value must "
        "come from the document, and there must be no other text and no "
        "markdown code fences.\n\n"
        f"Document:\n\"\"\"\n{document_text}\n\"\"\"\n\n"
        "Corrected JSON:"
    )


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #
def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Tolerates markdown code fences, surrounding explanation text, and
    Python-style literals (None/True) that models often emit instead of JSON.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ExtractionError("the model returned an empty response")

    # Drop ```json ... ``` fences if the model added them anyway.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ExtractionError("the response does not contain a JSON object")

    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to Python literals: e.g. {'title': None} instead of null.
        try:
            data = ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as exc:
            raise ExtractionError(f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ExtractionError("the JSON value is not an object")
    return data


def extract_information(
    document_text: str,
    llm: LLMCall,
    max_retries: int = MAX_RETRIES,
) -> DocumentInformation:
    """Extract and validate structured information, retrying on bad output."""
    if not document_text.strip():
        raise ExtractionError("there is no text to extract information from")

    prompt = build_extraction_prompt(document_text)
    last_error: Exception | None = None

    for _attempt in range(max_retries + 1):
        raw = (llm(prompt, MAX_NEW_TOKENS) or "").strip()
        try:
            data = parse_json(raw)
            return DocumentInformation.model_validate(data)
        except (ExtractionError, ValidationError) as exc:
            last_error = exc
            prompt = build_correction_prompt(document_text, raw, str(exc))

    raise ExtractionError(
        f"The model did not return valid structured data after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
