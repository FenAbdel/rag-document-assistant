import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

from app.config import GENERATION_MODEL, MAX_OUTPUT_TOKENS
from app.generation.citation_formatter import build_sources
from app.generation.prompt_builder import build_rag_prompt


load_dotenv()


class AnswerGenerator:
    """
    Service responsible for generating grounded answers
    from retrieved document chunks.
    """

    def __init__(self, model: str = GENERATION_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. "
                "Create a .env file and add your OpenAI API key."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        chat_history: List[Dict] = None,
    ) -> Dict:
        """
        Generate an answer from a question and retrieved chunks.

        Returns:
        - answer
        - sources
        - retrieved_chunks
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return {
                "answer": (
                    "The uploaded document does not provide enough "
                    "information to answer this question."
                ),
                "sources": [],
                "retrieved_chunks": [],
            }

        prompt = build_rag_prompt(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history or [],
        )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        answer = response.output_text
        sources = build_sources(retrieved_chunks)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }
