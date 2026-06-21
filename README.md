# RAG Document Assistant

A local document question-answering assistant based on Retrieval-Augmented Generation.

The application allows users to upload PDF documents, index them into a local vector store, ask questions about their content, and receive grounded answers with sources and citations.

## Project objective

The goal of this project is to build a professional first version of a document-based RAG assistant.

The assistant demonstrates:

- PDF document ingestion
- Text extraction
- Chunking with metadata
- Embedding generation
- Local vector storage with Chroma
- Semantic retrieval
- Answer generation with document-grounded context
- Source and citation display
- Conversation memory
- Optional multi-query retrieval
- Streamlit user interface
- CLI interface
- Basic retrieval evaluation
- Unit tests for core logic

## Main features

### Document indexing

- Upload or place PDF files in `data/raw/`
- Extract text page by page
- Split text into chunks
- Preserve metadata:
  - source filename
  - page number
  - document ID
  - content hash
  - chunk ID
- Store embeddings in a persistent local Chroma vector store
- Re-index safely by deleting old chunks from the same source before storing new ones

### Question answering

- Ask questions about indexed documents
- Retrieve the most relevant chunks from the vector store
- Generate grounded answers using retrieved document context
- Display sources with document name, page, chunk ID, and distance score
- Optionally inspect retrieved chunks for debugging

### Conversation memory

The assistant keeps short session-level memory so it can handle follow-up questions.

Example:

```text
User: What are the main technical skills?
User: Which ones are related to AI?
```

The memory is used to understand the follow-up question, while factual answers remain grounded in retrieved document chunks.

### Optional multi-query retrieval

The project supports optional multi-query retrieval.

When enabled, the system generates alternative search queries to improve retrieval recall for vague or ambiguous questions.

This feature is disabled by default to control API cost.

## Architecture overview

The project separates the RAG system into clear layers:

```text
PDF document
   |
   v
Ingestion
   |
   v
Chunking
   |
   v
Embeddings
   |
   v
Vector store
   |
   v
Retriever
   |
   v
Prompt builder
   |
   v
Answer generator
   |
   v
Answer with sources
```

The application has two main workflows:

### Indexing workflow

```text
PDF -> text extraction -> chunks -> embeddings -> Chroma vector store
```

### Query workflow

```text
question -> retrieval -> prompt construction -> LLM generation -> answer + sources
```

## Project structure

```text
rag-document-assistant/
|
|-- app/
|   |-- config.py
|   |
|   |-- ingestion/
|   |   `-- document_loader.py
|   |
|   |-- indexing/
|   |   |-- chunker.py
|   |   |-- embeddings.py
|   |   `-- vector_store.py
|   |
|   |-- retrieval/
|   |   |-- query_expander.py
|   |   `-- retriever.py
|   |
|   |-- generation/
|   |   |-- answer_generator.py
|   |   |-- citation_formatter.py
|   |   `-- prompt_builder.py
|   |
|   |-- memory/
|   |   `-- conversation_memory.py
|   |
|   |-- pipelines/
|   |   |-- indexing_pipeline.py
|   |   `-- rag_pipeline.py
|   |
|   `-- ui/
|       `-- streamlit_app.py
|
|-- data/
|   |-- raw/
|   `-- vector_store/
|
|-- evaluation/
|   |-- evaluation_questions.json
|   `-- run_retrieval_eval.py
|
|-- tests/
|
|-- docs/
|   |-- architecture.md
|   |-- evaluation.md
|   `-- portfolio.md
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
`-- run.py
```

## Technology stack

- Python 3.9.9
- PyMuPDF for PDF text extraction
- LangChain text splitters for chunking
- OpenAI embeddings for semantic representation
- Chroma for local vector storage
- OpenAI generation model for answer generation
- Streamlit for the web interface
- Pytest for unit tests

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`.

Minimum required content:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage from CLI

### Index a PDF

Place a PDF file inside:

```text
data/raw/
```

Then run:

```bash
python run.py index
```

### List indexed documents

```bash
python run.py documents
```

### Ask a question

```bash
python run.py ask "What is this document about?"
```

### Ask with source filtering

```bash
python run.py ask "What are the main technical skills?" --source "document.pdf"
```

### Show retrieved context

```bash
python run.py ask "What are the main technical skills?" --show-context
```

### Start chat mode with memory

```bash
python run.py chat
```

### Enable optional multi-query retrieval

```bash
python run.py ask "What are the AI-related skills?" --multi-query
```

## Usage with Streamlit

Run:

```bash
streamlit run app/ui/streamlit_app.py
```

The Streamlit interface supports:

- PDF upload
- document indexing
- indexed document listing
- document deletion
- retrieval scope selection
- chat-style question answering
- conversation memory
- optional multi-query retrieval
- source display
- retrieved context inspection

## Testing

Run unit tests:

```bash
pytest
```

The unit tests are designed to avoid OpenAI API calls.

They test:

- document ID generation
- conversation memory
- citation formatting
- prompt construction
- retriever result deduplication

## Retrieval evaluation

A small retrieval evaluation dataset is available in:

```text
evaluation/evaluation_questions.json
```

Run retrieval evaluation:

```bash
python -m evaluation.run_retrieval_eval
```

Run evaluation with multi-query retrieval:

```bash
python -m evaluation.run_retrieval_eval --multi-query
```

This evaluation may call the OpenAI API because it uses the real retrieval pipeline.
