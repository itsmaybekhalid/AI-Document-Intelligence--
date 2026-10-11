# AI Document Intelligence System

## Project Overview

AI Document Intelligence is a system that understands uploaded PDF documents. It
reads a PDF locally, turns it into searchable chunks, and lets you work with it in
three ways:

1. Ask questions and get answers grounded in the document (RAG)
2. Generate summaries of any length
3. Extract structured information as validated JSON

The heavy language model (**Qwen3-14B**) runs on a **Kaggle GPU**, while
document processing, chunking, embeddings and vector search run **locally** — so
you get a powerful LLM without paying for an external API.

## Features

```text
✓ Document Q&A / RAG
✓ Document Summarization
✓ Structured Information Extraction
```

**Document Q&A / RAG** — The question is embedded, the most relevant chunks are
retrieved from FAISS, and Qwen3 answers using _only_ that context. If the answer
is not in the document, the system replies:

```text
I couldn't find this information in the uploaded document.
```

Retrieved source chunks can be displayed below every answer.

**Summarization** — Short Summary, Detailed Summary, or Key Points. Long
documents use a simple map-reduce: chunks are summarized individually, then the
partial summaries are combined into one final result.

**Structured Information Extraction** — Qwen3 returns JSON that is validated with
Pydantic. Invalid output triggers an automatic correction retry, and the result
is always a valid, schema-conforming JSON object:

```json
{
  "title": "...",
  "document_type": "...",
  "main_topic": "...",
  "key_points": [],
  "important_entities": []
}
```

## Architecture

The system is split across two environments connected by ngrok:

```text
                         LOCAL
────────────────────────────────────────

User
 ↓
Streamlit
 ↓
PDF Upload
 ↓
Docling
 ↓
LangChain
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Relevant Context
 ↓
Prompt
 ↓
ngrok


                         KAGGLE
────────────────────────────────────────

ngrok
 ↓
FastAPI
 ↓
Qwen3-14B
 ↓
Response


                         LOCAL
────────────────────────────────────────

Response
 ↓
Streamlit
```

Only prompts are sent to Kaggle — never the whole PDF:

- **Q&A** sends the retrieved context + your question
- **Summarization** sends the required document chunks
- **Extraction** sends the cleaned document text

Authentication: every request carries an `Authorization: Bearer <API_SECRET>`
header, and the endpoint rejects calls without it.

## Tech Stack

| Technology                    | Role                                                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python**                    | Everything is written in plain, readable Python.                                                                                                                     |
| **Streamlit**                 | Local web UI: upload, feature selection, and result display. Caches the processed document, the embedding model, and the FAISS index.                                |
| **Docling**                   | Parses the PDF into clean Markdown/text (tables and headings preserved).                                                                                             |
| **LangChain**                 | Document objects, text splitting, prompt templates, embeddings interface, retriever, and FAISS integration.                                                          |
| **Sentence Transformers**     | Embedding model `intfloat/multilingual-e5-base` — multilingual, works for English and Arabic documents. Easy to swap via one constant in `rag.py`.                   |
| **FAISS**                     | Local vector similarity search over the document chunks.                                                                                                             |
| **Qwen3-14B**                 | The main LLM (via Hugging Face Transformers). Runs on the Kaggle GPU; automatically falls back to the 4-bit `Qwen/Qwen3-14B-AWQ` build when GPU memory is limited.   |
| **PyTorch**                   | CUDA inference for Qwen3-14B on the Kaggle GPU (bf16 on Ampere+, fp16 otherwise).                                                                                    |
| **Hugging Face Transformers** | Loads the model and tokenizer and applies Qwen's chat template.                                                                                                      |
| **Pydantic**                  | Defines the extraction schema and validates every model response. The schema is the single source of truth — change the fields and the prompt updates automatically. |
| **FastAPI**                   | Serves the `/generate` endpoint on Kaggle.                                                                                                                           |
| **ngrok**                     | Exposes the Kaggle FastAPI server through a public URL.                                                                                                              |
| **Requests + python-dotenv**  | The local app calls the API with `requests`; secrets live in `.env`.                                                                                                 |

## Setup

### 1. Install the local dependencies

```bash
pip install -r requirements.txt
```

This installs everything for the local side (Streamlit, Docling, LangChain,
FAISS, Pydantic, Requests, dotenv) as well as the Kaggle-side packages, which
are separated by comments in `requirements.txt`.

### 2. Create your `.env` file

```text
KAGGLE_API_URL=
API_SECRET=
```

You will fill in both values after starting `kaggle_api.py` on Kaggle (next
section). Never hard-code secrets in Python files.

## Kaggle Setup

**`kaggle_api.py` is the file executed on Kaggle.** It is the only file that
runs there.

1. **Open a Kaggle Notebook** (https://www.kaggle.com/code) and upload or
   paste `kaggle_api.py`.
2. **Enable GPU**: _Add on → Accelerator → GPU T4 x2 (or newer)_.
3. **(Recommended) Add secrets** in _Add on → Secrets_:
   - `NGROK_AUTHTOKEN` — free token from https://dashboard.ngrok.com/get-started/your-authtoken
   - `API_SECRET` — any string you invent (if omitted, the script generates one)
4. **Run the file**:

   ```python
   !python kaggle_api.py
   ```

   The script then:
   - installs any missing packages (`fastapi`, `uvicorn`, `pyngrok`, `accelerate`)
   - detects CUDA and picks a suitable dtype
   - loads `Qwen/Qwen3-14B` — or the 4-bit `Qwen/Qwen3-14B-AWQ` build when the
     GPU has less than 34 GB (install `autoawq` first in that case:
     `!pip install autoawq`)
   - starts FastAPI with the authenticated `/generate` endpoint
   - opens an ngrok tunnel

5. **Copy the printed values**:

   ```text
   KAGGLE_API_URL=https://xxxx-xx-xx-xx.ngrok-free.app
   API_SECRET=...
   ```

   Put them into your local `.env`. If ngrok restarts, the URL changes — update
   `.env` accordingly.

## Local Setup

With the Kaggle API running and `.env` filled in:

```bash
streamlit run app.py
```

Then in the browser:

1. Upload a PDF (parsing with Docling takes a moment on first use)
2. Choose a feature: **Document Q&A**, **Summarization**, or **Information Extraction**
3. Ask, generate, or extract — results appear directly below

## Project Structure

```text
ai-document-intelligence/
│
├── app.py            # Local Streamlit app: UI, feature selection, Kaggle API
│                     # client, and the map-reduce summarization flow
├── document.py       # Document processing only: PDF -> Docling -> LangChain
│                     # Document -> chunks
├── rag.py            # RAG pipeline only: embeddings, FAISS, retriever,
│                     # context building, and the RAG prompt
├── extraction.py     # Structured extraction only: prompts, JSON parsing,
│                     # retries, and Pydantic validation
├── kaggle_api.py     # THE file that runs on Kaggle: loads Qwen3-14B, serves
│                     # FastAPI /generate, opens the ngrok tunnel
│
├── requirements.txt  # Dependencies (local and Kaggle, separated by comments)
├── README.md
└── .env              # KAGGLE_API_URL and API_SECRET (never committed)
```

### How the pieces fit together

```text
                 PDF
                  │
                  ▼
               Docling
                  │
                  ▼
             LangChain
                  │
                  ▼
               Chunks
                  │
             ┌────┼─────┐
             │    │     │
             ▼    ▼     ▼
            RAG Summary Extract
             │    │     │
             └────┼─────┘
                  │
                  ▼
              Qwen3-14B
                  │
                  ▼
              Streamlit
```

## Error Handling

All failures are reported as short, human-readable messages in the UI — never
raw stack traces:

- invalid / empty / unparseable PDF
- empty extracted text (e.g. scanned documents)
- embedding or FAISS failures
- Kaggle API unreachable (ngrok down) or rejecting the secret
- Qwen generation failures
- invalid JSON / Pydantic validation failures (retried automatically)

## Future Features

Not implemented yet — planned for later:

```text
Future:
- Document Comparison
- Multi-document RAG
- Advanced OCR
- Agents
- More structured extraction schemas
```
