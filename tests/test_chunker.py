from app.indexing.chunker import chunk_pages


def test_chunk_pages_keeps_metadata():
    pages = [
        {
            "content": "This is a test document. " * 100,
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        }
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "test.pdf"
    assert chunks[0]["metadata"]["page"] == 1
    assert "chunk_id" in chunks[0]["metadata"]
    assert chunks[0]["content"]