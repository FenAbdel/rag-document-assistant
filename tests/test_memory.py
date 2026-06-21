from app.memory.conversation_memory import (
    append_turn,
    build_retrieval_query,
    format_chat_history,
    trim_chat_history,
)


def test_trim_chat_history_keeps_recent_messages():
    chat_history = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
    ]

    trimmed = trim_chat_history(chat_history, max_messages=2)

    assert len(trimmed) == 2
    assert trimmed[0]["content"] == "Question 2"
    assert trimmed[1]["content"] == "Answer 2"


def test_format_chat_history_empty():
    assert format_chat_history([]) == "No previous conversation."


def test_build_retrieval_query_includes_history_and_question():
    chat_history = [
        {"role": "user", "content": "What are the main skills?"},
        {"role": "assistant", "content": "Python and SQL."},
    ]

    query = build_retrieval_query(
        question="Which ones are related to AI?",
        chat_history=chat_history,
    )

    assert "Recent conversation:" in query
    assert "What are the main skills?" in query
    assert "Which ones are related to AI?" in query


def test_append_turn_adds_user_and_assistant_messages():
    chat_history = []

    updated = append_turn(
        chat_history=chat_history,
        question="What is the document about?",
        answer="It is about a RAG assistant.",
    )

    assert len(updated) == 2
    assert updated[0]["role"] == "user"
    assert updated[1]["role"] == "assistant"