from typing import Dict, List, Optional

from app.config import DEFAULT_TOP_K
from app.indexing.embeddings import OpenAIEmbeddingService
from app.indexing.vector_store import query_similar_chunks


class DocumentRetriever:
    """
    High-level retrieval service.

    It receives a user question, embeds it, searches the vector store,
    and returns the most relevant chunks with their metadata.
    """

    def __init__(
        self,
        embedding_service: Optional[OpenAIEmbeddingService] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.embedding_service = embedding_service or OpenAIEmbeddingService()
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for one retrieval query.

        If source_filter is provided, retrieval is restricted to one document.
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        query_embedding = self.embedding_service.embed_query(cleaned_question)

        retrieved_chunks = query_similar_chunks(
            query_embedding=query_embedding,
            top_k=self.top_k,
            source_filter=source_filter,
        )

        return self._normalize_results(retrieved_chunks)

    def retrieve_from_queries(
        self,
        queries: List[str],
        source_filter: Optional[str] = None,
        top_k_per_query: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve chunks from multiple search queries, deduplicate them,
        and keep the best final chunks.
        """
        clean_queries = [
            query.strip()
            for query in queries
            if query and query.strip()
        ]

        if not clean_queries:
            raise ValueError("At least one retrieval query is required.")

        top_k_per_query = top_k_per_query or self.top_k
        final_top_k = final_top_k or self.top_k

        best_chunks_by_id = {}

        for query in clean_queries:
            query_embedding = self.embedding_service.embed_query(query)

            raw_chunks = query_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k_per_query,
                source_filter=source_filter,
            )

            normalized_chunks = self._normalize_results(raw_chunks)

            for chunk in normalized_chunks:
                chunk_id = chunk.get("chunk_id")

                if not chunk_id:
                    continue

                chunk["retrieval_query"] = query

                existing_chunk = best_chunks_by_id.get(chunk_id)

                if existing_chunk is None:
                    best_chunks_by_id[chunk_id] = chunk
                    continue

                existing_distance = existing_chunk.get("distance")
                current_distance = chunk.get("distance")

                if existing_distance is None:
                    best_chunks_by_id[chunk_id] = chunk
                elif current_distance is not None and current_distance < existing_distance:
                    best_chunks_by_id[chunk_id] = chunk

        merged_chunks = list(best_chunks_by_id.values())

        merged_chunks = sorted(
            merged_chunks,
            key=lambda chunk: (
                chunk.get("distance") is None,
                chunk.get("distance"),
            ),
        )

        final_chunks = merged_chunks[:final_top_k]

        for rank, chunk in enumerate(final_chunks, start=1):
            chunk["rank"] = rank

        return final_chunks

    def _normalize_results(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Normalize retrieved chunks into a clean structure for the generation layer.

        This keeps the retrieval output stable even if the vector database
        implementation changes later.
        """
        normalized_chunks = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            normalized_chunks.append(
                {
                    "rank": rank,
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source"),
                    "page": chunk["metadata"].get("page"),
                    "document_id": chunk["metadata"].get("document_id"),
                    "content_hash": chunk["metadata"].get("content_hash"),
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "distance": chunk.get("distance"),
                    "metadata": chunk["metadata"],
                }
            )

        return normalized_chunks