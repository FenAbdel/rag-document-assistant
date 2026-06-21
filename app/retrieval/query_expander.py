import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.config import MULTI_QUERY_COUNT, QUERY_REWRITE_MODEL
from app.memory.conversation_memory import format_chat_history


load_dotenv()


class QueryExpander:
    """
    Service responsible for generating alternative retrieval queries.

    This is used only when multi-query retrieval is enabled.
    """

    def __init__(self, model: str = QUERY_REWRITE_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. "
                "Create a .env file and add your OpenAI API key."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_queries(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        number_of_queries: int = MULTI_QUERY_COUNT,
    ) -> List[str]:
        """
        Generate alternative search queries for retrieval.

        The generated queries are not final answers.
        They are only used to improve vector search.
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        history = format_chat_history(chat_history or [])

        prompt = """
You are improving document retrieval for a RAG system.

Recent conversation:
{history}

Current user question:
{question}

Generate {number_of_queries} alternative search queries that could help retrieve relevant document chunks.

Rules:
- Return only the queries.
- One query per line.
- Do not number the queries.
- Do not explain anything.
- Keep the same language as the user question when possible.
- Do not answer the question.
""".strip().format(
            history=history,
            question=cleaned_question,
            number_of_queries=number_of_queries,
        )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=250,
        )

        return self._parse_queries(
            text=response.output_text,
            max_queries=number_of_queries,
        )

    def _parse_queries(
        self,
        text: str,
        max_queries: int,
    ) -> List[str]:
        """
        Parse raw model output into clean query strings.
        """
        queries = []

        for line in text.splitlines():
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            cleaned_line = re.sub(
                r"^[-*\d\.\)\s]+",
                "",
                cleaned_line,
            ).strip()

            if cleaned_line:
                queries.append(cleaned_line)

        unique_queries = []

        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)

        return unique_queries[:max_queries]