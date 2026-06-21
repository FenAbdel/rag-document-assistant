from app.generation.prompt_builder import build_rag_prompt


def test_build_rag_prompt_includes_context_question_and_rules():
    retrieved_chunks = [
        {
            "rank": 1,
            "source": "doc.pdf",
            "page": 1,
            "chunk_id": "doc_p1_c1",
            "content": "The document mentions Python and SQL.",
        }
    ]

    chat_history = [
        {"role": "user", "content": "What are the main skills?"},
        {"role": "assistant", "content": "Python and SQL."},
    ]

    prompt = build_rag_prompt(
        question="Which ones are related to data?",
        retrieved_chunks=retrieved_chunks,
        chat_history=chat_history,
    )

    assert "Recent conversation:" in prompt
    assert "Document context:" in prompt
    assert "The document mentions Python and SQL." in prompt
    assert "Which ones are related to data?" in prompt
    assert "Use only the document context" in prompt
    assert "Do not invent facts" in prompt