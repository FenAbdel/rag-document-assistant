from typing import Dict, List, Optional

from app.memory.conversation_memory import format_chat_history


def format_retrieved_context(retrieved_chunks: List[Dict]) -> str:
    """
    Convert retrieved chunks into a readable context block for the LLM.

    Each chunk receives a source label so the model can refer to it.
    """
    context_parts = []

    for chunk in retrieved_chunks:
        source_label = (
            "[Source {rank}] {source}, page {page}, chunk_id: {chunk_id}"
        ).format(
            rank=chunk["rank"],
            source=chunk["source"],
            page=chunk["page"],
            chunk_id=chunk["chunk_id"],
        )

        context_parts.append(
            "{source_label}\n{content}".format(
                source_label=source_label,
                content=chunk["content"],
            )
        )

    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict],
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """
    Build the user prompt sent to the LLM.

    The prompt includes:
    - recent conversation history
    - retrieved document context
    - current user question
    - strict answer rules
    """
    context = format_retrieved_context(retrieved_chunks)
    history = format_chat_history(chat_history or [])

    prompt = """
You are a document question-answering assistant.

Your task:
Answer the current question using only the document context provided below.

Recent conversation:
{history}

Document context:
{context}

Current question:
{question}

Answering rules:
- Use the recent conversation only to understand the current question.
- Use only the document context to produce factual claims.
- If the document context does not contain enough information, say clearly that the uploaded document does not provide enough information.
- Do not invent facts.
- Do not use outside knowledge.
- Be clear and concise.
- If the question is a follow-up, resolve what it refers to using the recent conversation.
- At the end, mention the sources you used using this format:
  Sources: [Source 1], [Source 2]
""".strip().format(
        history=history,
        context=context,
        question=question,
    )

    return prompt