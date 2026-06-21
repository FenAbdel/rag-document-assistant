from typing import Dict, List

from app.config import MAX_MEMORY_MESSAGES


def trim_chat_history(
    chat_history: List[Dict],
    max_messages: int = MAX_MEMORY_MESSAGES,
) -> List[Dict]:
    """
    Keep only the most recent chat messages.

    This avoids sending too much conversation history to the model.
    """
    if not chat_history:
        return []

    return chat_history[-max_messages:]


def format_chat_history(chat_history: List[Dict]) -> str:
    """
    Format chat history into a readable text block for prompts.
    """
    if not chat_history:
        return "No previous conversation."

    formatted_messages = []

    for message in trim_chat_history(chat_history):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if not content:
            continue

        formatted_messages.append(
            "{}: {}".format(role.capitalize(), content)
        )

    if not formatted_messages:
        return "No previous conversation."

    return "\n".join(formatted_messages)


def build_retrieval_query(
    question: str,
    chat_history: List[Dict],
) -> str:
    """
    Build a retrieval query using the current question and recent history.

    This helps retrieval for follow-up questions without making
    an additional LLM call.
    """
    cleaned_question = question.strip()

    if not chat_history:
        return cleaned_question

    recent_history = format_chat_history(chat_history)

    retrieval_query = """
Recent conversation:
{recent_history}

Current question:
{question}
""".strip().format(
        recent_history=recent_history,
        question=cleaned_question,
    )

    return retrieval_query


def append_turn(
    chat_history: List[Dict],
    question: str,
    answer: str,
) -> List[Dict]:
    """
    Add one user-assistant turn to memory and trim it.
    """
    updated_history = list(chat_history)

    updated_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    updated_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return trim_chat_history(updated_history)