from app.generation.citation_formatter import (
    build_sources,
    format_sources_for_console,
)


def test_build_sources_deduplicates_chunks():
    retrieved_chunks = [
        {
            "rank": 1,
            "source": "doc.pdf",
            "page": 1,
            "chunk_id": "doc_p1_c1",
            "distance": 0.12,
        },
        {
            "rank": 2,
            "source": "doc.pdf",
            "page": 1,
            "chunk_id": "doc_p1_c1",
            "distance": 0.12,
        },
    ]

    sources = build_sources(retrieved_chunks)

    assert len(sources) == 1
    assert sources[0]["source"] == "doc.pdf"
    assert sources[0]["chunk_id"] == "doc_p1_c1"


def test_format_sources_for_console():
    sources = [
        {
            "rank": 1,
            "label": "Source 1",
            "source": "doc.pdf",
            "page": 2,
            "chunk_id": "doc_p2_c1",
            "distance": 0.123456,
        }
    ]

    formatted = format_sources_for_console(sources)

    assert "doc.pdf" in formatted
    assert "page 2" in formatted
    assert "0.1235" in formatted