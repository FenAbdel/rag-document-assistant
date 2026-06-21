from typing import Dict, List, Optional

import chromadb

from app.config import COLLECTION_NAME, VECTOR_STORE_DIR


def get_collection():
    """
    Create or load the local Chroma collection.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def list_indexed_documents() -> List[Dict]:
    """
    List documents currently indexed in the vector store.

    The vector store contains chunks, not documents directly.
    So we rebuild a document-level view by grouping chunks by source.
    """
    collection = get_collection()

    results = collection.get(
        include=["metadatas"],
    )

    metadatas = results.get("metadatas", [])

    grouped_documents = {}

    for metadata in metadatas:
        if not metadata:
            continue

        source = metadata.get("source")

        if not source:
            continue

        if source not in grouped_documents:
            grouped_documents[source] = {
                "source": source,
                "chunk_count": 0,
                "pages": set(),
                "document_ids": set(),
                "content_hashes": set(),
            }

        grouped_documents[source]["chunk_count"] += 1

        page = metadata.get("page")
        document_id = metadata.get("document_id")
        content_hash = metadata.get("content_hash")

        if page is not None:
            grouped_documents[source]["pages"].add(page)

        if document_id:
            grouped_documents[source]["document_ids"].add(document_id)

        if content_hash:
            grouped_documents[source]["content_hashes"].add(content_hash)

    documents = []

    for document in grouped_documents.values():
        pages = sorted(document["pages"])

        documents.append(
            {
                "source": document["source"],
                "chunk_count": document["chunk_count"],
                "page_count": len(pages),
                "pages": pages,
                "document_ids": sorted(document["document_ids"]),
                "content_hashes": sorted(document["content_hashes"]),
            }
        )

    return sorted(documents, key=lambda item: item["source"])


def count_chunks_by_source(source: str) -> int:
    """
    Count chunks already stored for a given source document.
    """
    collection = get_collection()

    results = collection.get(
        where={"source": source},
    )

    return len(results["ids"])


def delete_chunks_by_source(source: str) -> int:
    """
    Delete all chunks linked to a given source document.

    Returns the number of chunks that were found before deletion.
    """
    collection = get_collection()

    existing_count = count_chunks_by_source(source)

    if existing_count == 0:
        return 0

    collection.delete(
        where={"source": source},
    )

    return existing_count


def upsert_chunks(
    chunks: List[Dict],
    embeddings: List[List[float]],
) -> int:
    """
    Store chunks, metadata, and embeddings in Chroma.

    We use upsert instead of add so running the script multiple times
    updates existing chunks instead of crashing on duplicate IDs.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must be equal. "
            "Got {} chunks and {} embeddings.".format(
                len(chunks),
                len(embeddings),
            )
        )

    if not chunks:
        return 0

    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def get_vector_store_count() -> int:
    """
    Return the number of stored chunks in the collection.
    """
    collection = get_collection()
    return collection.count()


def query_similar_chunks(
    query_embedding: List[float],
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Similarity search over the vector store.

    If source_filter is provided, retrieval is restricted to one document.
    """
    collection = get_collection()

    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if source_filter:
        query_arguments["where"] = {"source": source_filter}

    results = collection.query(**query_arguments)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks