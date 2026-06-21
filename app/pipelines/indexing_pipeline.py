from pathlib import Path
from typing import Dict, Optional

from app.ingestion.document_loader import load_pdf_pages
from app.indexing.chunker import chunk_pages
from app.indexing.embeddings import OpenAIEmbeddingService
from app.indexing.vector_store import (
    delete_chunks_by_source,
    get_vector_store_count,
    upsert_chunks,
)


class DocumentIndexingPipeline:
    """
    Pipeline responsible for indexing documents into the vector store.

    This pipeline handles:
    - removing old chunks for the same source
    - PDF loading
    - text extraction
    - chunking
    - embedding creation
    - storing chunks in Chroma
    """

    def __init__(
        self,
        embedding_service: Optional[OpenAIEmbeddingService] = None,
    ):
        self.embedding_service = embedding_service or OpenAIEmbeddingService()

    def index_pdf(self, file_path: Path) -> Dict:
        """
        Index one PDF file into the vector store.

        Before indexing, old chunks from the same source are deleted
        to avoid stale retrieval results.
        """
        deleted_chunks = delete_chunks_by_source(file_path.name)

        pages = load_pdf_pages(file_path)
        chunks = chunk_pages(pages)

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        stored_count = upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        total_count = get_vector_store_count()

        document_id = None
        content_hash = None

        if chunks:
            document_id = chunks[0]["metadata"].get("document_id")
            content_hash = chunks[0]["metadata"].get("content_hash")

        return {
            "document": file_path.name,
            "document_id": document_id,
            "content_hash": content_hash,
            "deleted_existing_chunks": deleted_chunks,
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
            "embeddings_created": len(embeddings),
            "chunks_stored": stored_count,
            "total_chunks_in_vector_store": total_count,
        }