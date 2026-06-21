import app.retrieval.retriever as retriever_module
from app.retrieval.retriever import DocumentRetriever


class FakeEmbeddingService:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


def test_retrieve_from_queries_deduplicates_and_keeps_best_distance(
    monkeypatch,
):
    def fake_query_similar_chunks(
        query_embedding,
        top_k=3,
        source_filter=None,
    ):
        if source_filter == "doc.pdf":
            return [
                {
                    "content": "Chunk about AI skills.",
                    "metadata": {
                        "source": "doc.pdf",
                        "page": 1,
                        "document_id": "doc_hash",
                        "content_hash": "hash",
                        "chunk_id": "chunk_1",
                    },
                    "distance": 0.30,
                },
                {
                    "content": "Chunk about SQL.",
                    "metadata": {
                        "source": "doc.pdf",
                        "page": 1,
                        "document_id": "doc_hash",
                        "content_hash": "hash",
                        "chunk_id": "chunk_2",
                    },
                    "distance": 0.50,
                },
            ]

        return []

    monkeypatch.setattr(
        retriever_module,
        "query_similar_chunks",
        fake_query_similar_chunks,
    )

    retriever = DocumentRetriever(
        embedding_service=FakeEmbeddingService(),
        top_k=3,
    )

    results = retriever.retrieve_from_queries(
        queries=["AI skills", "machine learning skills"],
        source_filter="doc.pdf",
        top_k_per_query=2,
        final_top_k=2,
    )

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[1]["chunk_id"] == "chunk_2"