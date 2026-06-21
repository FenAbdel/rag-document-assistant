from typing import Dict, List


def build_sources(retrieved_chunks: List[Dict]) -> List[Dict]:
    """
    Build a clean, programmatic source list from retrieved chunks.

    We do not rely only on the LLM to generate citations because
    the LLM can forget, duplicate, or incorrectly format sources.
    """
    sources = []
    seen_chunk_ids = set()

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)

        sources.append(
            {
                "rank": chunk.get("rank"),
                "label": "Source {}".format(chunk.get("rank")),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "distance": chunk.get("distance"),
            }
        )

    return sources


def format_sources_for_console(sources: List[Dict]) -> str:
    """
    Format sources for command-line display.
    """
    if not sources:
        return "No sources available."

    lines = []

    for source in sources:
        distance = source.get("distance")

        if isinstance(distance, float):
            distance_text = "{:.4f}".format(distance)
        else:
            distance_text = str(distance)

        line = (
            "- [{label}] {source}, page {page}, "
            "chunk_id: {chunk_id}, distance: {distance}"
        ).format(
            label=source.get("label"),
            source=source.get("source"),
            page=source.get("page"),
            chunk_id=source.get("chunk_id"),
            distance=distance_text,
        )

        lines.append(line)

    return "\n".join(lines)


def format_retrieved_chunks_for_console(
    retrieved_chunks: List[Dict],
    preview_characters: int = 700,
) -> str:
    """
    Format retrieved chunks for debugging retrieval quality.
    """
    if not retrieved_chunks:
        return "No retrieved chunks available."

    parts = []

    for chunk in retrieved_chunks:
        distance = chunk.get("distance")

        if isinstance(distance, float):
            distance_text = "{:.4f}".format(distance)
        else:
            distance_text = str(distance)

        preview = chunk.get("content", "")[:preview_characters]

        part = (
            "Result {rank}\n"
            "Source: {source}\n"
            "Page: {page}\n"
            "Chunk ID: {chunk_id}\n"
            "Distance: {distance}\n"
            "Preview:\n"
            "{preview}"
        ).format(
            rank=chunk.get("rank"),
            source=chunk.get("source"),
            page=chunk.get("page"),
            chunk_id=chunk.get("chunk_id"),
            distance=distance_text,
            preview=preview,
        )

        parts.append(part)

    separator = "\n\n" + ("-" * 50) + "\n\n"

    return separator.join(parts)
