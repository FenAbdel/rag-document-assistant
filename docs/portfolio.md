# Portfolio notes

## CV-ready project description in French

Assistant documentaire RAG permettant de répondre à des questions sur des documents PDF uploadés. Le projet intègre l'extraction de texte, le découpage en chunks, la génération d'embeddings, le stockage vectoriel avec Chroma, la récupération de contexte, la génération de réponses sourcées, la mémoire conversationnelle et une interface Streamlit.

## Short CV bullet version in French

- Développement d'un assistant documentaire RAG en Python permettant de questionner des PDF avec réponses contextualisées et sources.
- Mise en place d'un pipeline d'ingestion : extraction PDF, chunking, embeddings OpenAI, stockage vectoriel Chroma et récupération sémantique.
- Ajout de citations, mémoire conversationnelle, filtrage par document, multi-query retrieval optionnel, interface Streamlit, tests unitaires et évaluation de la qualité de récupération.

## LinkedIn / GitHub project description

I built a local RAG Document Assistant that allows users to upload PDF documents and ask questions about their content.

The project includes a complete RAG pipeline: PDF ingestion, page-level text extraction, chunking with metadata, OpenAI embeddings, Chroma vector storage, semantic retrieval, grounded answer generation, citations, conversation memory, optional multi-query retrieval, CLI commands, Streamlit interface, unit tests, and a simple retrieval evaluation framework.

The goal of this project was to build a professional first version of a document-based RAG assistant while focusing on clean architecture, traceability, retrieval quality, and cost-conscious design.

## Skills demonstrated

### RAG and LLM engineering

- Retrieval-Augmented Generation
- prompt construction
- grounded answer generation
- source citation strategy
- conversation memory
- multi-query retrieval
- retrieval debugging

### Data and document processing

- PDF text extraction
- chunking strategy
- metadata preservation
- document hashing
- document indexing
- vector store management

### Vector search

- embeddings
- Chroma vector database
- semantic search
- source filtering
- result deduplication

### Software engineering

- modular Python architecture
- service separation
- pipeline design
- configuration management
- CLI interface
- Streamlit interface
- unit testing
- project documentation

### Evaluation

- retrieval evaluation dataset
- retrieval quality checks
- testable core logic
- cost-conscious evaluation strategy

## Honest positioning

This project is a strong junior-level RAG project.

It does not claim production-grade deployment or enterprise-level scale.

Its value is in demonstrating a complete and understandable RAG architecture with clean engineering decisions, citations, memory, optional retrieval improvement, tests, and documentation.

## Improvement roadmap

Possible next improvements:

1. Add DOCX, TXT, and Markdown support
2. Add OCR for scanned PDFs
3. Add hybrid search with keyword + vector retrieval
4. Add reranking for better precision
5. Add FastAPI backend
6. Add Docker support
7. Add authentication
8. Add cloud deployment
9. Add local embedding model option
10. Add monitoring and logging
11. Add answer quality evaluation
12. Add evaluation dashboard
13. Add persistent document metadata database
14. Add user-level document collections
