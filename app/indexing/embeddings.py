import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


load_dotenv()


class OpenAIEmbeddingService:
    """
    Service responsible for converting text into embeddings.

    This class isolates the embedding provider.
    Later, we can replace OpenAI with a local embedding model
    without changing the rest of the indexing pipeline.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. "
                "Create a .env file and add your OpenAI API key."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of texts into a list of embeddings.
        """
        if not texts:
            return []

        embeddings = []

        for start_index in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start_index : start_index + EMBEDDING_BATCH_SIZE]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Convert one user query into one embedding.
        """
        return self.embed_texts([query])[0]