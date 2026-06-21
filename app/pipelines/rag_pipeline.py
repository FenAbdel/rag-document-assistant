from typing import Dict, List, Optional

from app.config import (
    MULTI_QUERY_FINAL_TOP_K,
    MULTI_QUERY_TOP_K_PER_QUERY,
)
from app.generation.answer_generator import AnswerGenerator
from app.indexing.embeddings import OpenAIEmbeddingService
from app.memory.conversation_memory import build_retrieval_query
from app.retrieval.query_expander import QueryExpander
from app.retrieval.retriever import DocumentRetriever


class RAGPipeline:
    """
    Pipeline responsible for answering questions using indexed documents.

    This pipeline handles:
    - retrieval query construction
    - optional multi-query expansion
    - document retrieval
    - grounded answer generation
    - source return
    """

    def __init__(
        self,
        embedding_service: Optional[OpenAIEmbeddingService] = None,
        retriever: Optional[DocumentRetriever] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.embedding_service = embedding_service or OpenAIEmbeddingService()

        self.retriever = retriever or DocumentRetriever(
            embedding_service=self.embedding_service,
        )

        self.answer_generator = answer_generator or AnswerGenerator()
        self.query_expander = query_expander

    def answer_question(
        self,
        question: str,
        source_filter: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        use_multi_query: bool = False,
    ) -> Dict:
        """
        Retrieve relevant chunks and generate a grounded answer.

        If use_multi_query is True, alternative search queries are generated
        to improve retrieval recall.
        """
        history = chat_history or []

        base_retrieval_query = build_retrieval_query(
            question=question,
            chat_history=history,
        )

        retrieval_queries = [base_retrieval_query]
        retrieval_mode = "single_query"

        if use_multi_query:
            query_expander = self.query_expander or QueryExpander()

            alternative_queries = query_expander.generate_queries(
                question=question,
                chat_history=history,
            )

            retrieval_queries.extend(alternative_queries)
            retrieval_mode = "multi_query"

            retrieved_chunks = self.retriever.retrieve_from_queries(
                queries=retrieval_queries,
                source_filter=source_filter,
                top_k_per_query=MULTI_QUERY_TOP_K_PER_QUERY,
                final_top_k=MULTI_QUERY_FINAL_TOP_K,
            )

        else:
            retrieved_chunks = self.retriever.retrieve(
                question=base_retrieval_query,
                source_filter=source_filter,
            )

        result = self.answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            chat_history=history,
        )

        result["retrieval_mode"] = retrieval_mode
        result["retrieval_queries"] = retrieval_queries

        return result