# Technical architecture

## Overview

The RAG Document Assistant is built around two separate workflows:

1. Document indexing
2. Question answering

This separation is important because documents should be indexed once and then queried many times.

## Indexing workflow

```text
PDF file
   |
   v
PDF loader
   |
   v
page-level text extraction
   |
   v
chunking
   |
   v
embedding generation
   |
   v
Chroma vector store
```

### Main components

| Component | File | Responsibility |
| --- | --- | --- |
| Document loader | `app/ingestion/document_loader.py` | Extract text page by page from PDFs |
| Chunker | `app/indexing/chunker.py` | Split pages into smaller chunks |
| Embedding service | `app/indexing/embeddings.py` | Convert text chunks into embeddings |
| Vector store | `app/indexing/vector_store.py` | Store and query chunks in Chroma |
| Indexing pipeline | `app/pipelines/indexing_pipeline.py` | Orchestrate the full indexing process |

## Query workflow

```text
User question
   |
   v
optional conversation memory
   |
   v
retrieval query
   |
   v
vector search
   |
   v
retrieved chunks
   |
   v
prompt construction
   |
   v
LLM answer generation
   |
   v
answer with sources
```

### Main components

| Component | File | Responsibility |
| --- | --- | --- |
| Retriever | `app/retrieval/retriever.py` | Retrieve relevant chunks |
| Query expander | `app/retrieval/query_expander.py` | Generate alternative queries when multi-query is enabled |
| Prompt builder | `app/generation/prompt_builder.py` | Build grounded RAG prompts |
| Answer generator | `app/generation/answer_generator.py` | Call the LLM and return the answer |
| Citation formatter | `app/generation/citation_formatter.py` | Build clean source display |
| Conversation memory | `app/memory/conversation_memory.py` | Keep short chat history |
| RAG pipeline | `app/pipelines/rag_pipeline.py` | Orchestrate retrieval and generation |

## Metadata strategy

Each chunk stores metadata:

```python
{
    "document_id": "...",
    "source": "document.pdf",
    "page": 1,
    "content_hash": "...",
    "chunk_id": "..."
}
```

This metadata is essential for:

- citations
- source filtering
- document deletion
- safer re-indexing
- debugging retrieval results

## Document ID and content hash

The document loader computes a SHA-256 hash of the PDF file.

This allows the system to identify a specific version of a document.

The `document_id` is built from:

```text
clean filename + short content hash
```

Example:

```text
my_report_a1b2c3d4e5f6
```

## Safer re-indexing

Before indexing a PDF, the indexing pipeline deletes existing chunks with the same source filename.

This prevents stale chunks from remaining in the vector store when a document is updated.

## Retrieval modes

The project supports two retrieval modes.

### Single-query retrieval

Default mode.

```text
question -> embedding -> vector search
```

This is cheaper and suitable for most questions.

### Multi-query retrieval

Optional mode.

```text
question -> alternative queries -> multiple searches -> deduplication -> final chunks
```

This can improve recall for vague or ambiguous questions, but it increases API usage.

## Conversation memory

The assistant stores short session-level conversation history.

Memory is used to understand follow-up questions, but it is not treated as a factual source.

The factual source remains the retrieved document context.

## Interfaces

The project provides two interfaces:

### CLI

Implemented in:

```text
run.py
```

Supports:

- indexing
- asking questions
- interactive chat
- listing documents
- deleting documents
- source filtering
- multi-query retrieval
- retrieved context debugging

### Streamlit UI

Implemented in:

```text
app/ui/streamlit_app.py
```

Supports:

- PDF upload
- indexing
- document management
- chat interface
- memory
- source display
- retrieval details
- retrieved context inspection

## Design principles

The project follows these principles:

- separate indexing from querying
- keep metadata from the beginning
- avoid duplicated versioned files
- keep UI separate from core RAG logic
- make retrieval inspectable
- make tests free from external API calls
- keep advanced features optional when they increase cost
