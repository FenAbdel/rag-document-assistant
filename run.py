import argparse
from pathlib import Path
from typing import Optional

from app.config import RAW_DATA_DIR
from app.generation.citation_formatter import (
    format_retrieved_chunks_for_console,
    format_sources_for_console,
)
from app.indexing.vector_store import (
    delete_chunks_by_source,
    list_indexed_documents,
)
from app.memory.conversation_memory import append_turn
from app.pipelines.indexing_pipeline import DocumentIndexingPipeline
from app.pipelines.rag_pipeline import RAGPipeline


def find_first_pdf() -> Path:
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF file found in {RAW_DATA_DIR}. "
            "Add one PDF file to data/raw/ and try again."
        )

    return pdf_files[0]


def index_command() -> None:
    pdf_path = find_first_pdf()

    indexing_pipeline = DocumentIndexingPipeline()
    stats = indexing_pipeline.index_pdf(pdf_path)

    print("\nIndexing completed")
    print("-" * 50)
    print(f"Document: {stats['document']}")
    print(f"Document ID: {stats['document_id']}")
    print(f"Content hash: {stats['content_hash']}")
    print(f"Deleted old chunks: {stats['deleted_existing_chunks']}")
    print(f"Pages extracted: {stats['pages_extracted']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Embeddings created: {stats['embeddings_created']}")
    print(f"Chunks stored: {stats['chunks_stored']}")
    print(
        "Total chunks in vector store: "
        f"{stats['total_chunks_in_vector_store']}"
    )


def documents_command() -> None:
    documents = list_indexed_documents()

    print("\nIndexed documents")
    print("-" * 50)

    if not documents:
        print("No indexed documents found.")
        return

    for document in documents:
        print(f"Source: {document['source']}")
        print(f"Chunks: {document['chunk_count']}")
        print(f"Pages: {document['page_count']}")
        print(f"Document IDs: {', '.join(document['document_ids'])}")
        print("-" * 50)


def delete_document_command(source: str) -> None:
    deleted_count = delete_chunks_by_source(source)

    print("\nDelete document")
    print("-" * 50)
    print(f"Source: {source}")
    print(f"Deleted chunks: {deleted_count}")


def ask_command(
    question: str,
    show_context: bool = False,
    source_filter: Optional[str] = None,
    use_multi_query: bool = False,
) -> None:
    rag_pipeline = RAGPipeline()

    result = rag_pipeline.answer_question(
        question=question,
        source_filter=source_filter,
        use_multi_query=use_multi_query,
    )

    print("\nQuestion")
    print("-" * 50)
    print(question)

    if source_filter:
        print("\nSource filter")
        print("-" * 50)
        print(source_filter)
        print("\nRetrieval mode")

    print("-" * 50)
    print(result.get("retrieval_mode"))

    if use_multi_query:
        print("\nRetrieval queries")
        print("-" * 50)

        for index, query in enumerate(
            result.get("retrieval_queries", []),
            start=1,
        ):
            print(f"{index}. {query}")

    print("\nAnswer")
    print("-" * 50)
    print(result["answer"])

    print("\nSources")
    print("-" * 50)
    print(format_sources_for_console(result["sources"]))

    if show_context:
        print("\nRetrieved context")
        print("-" * 50)
        print(
            format_retrieved_chunks_for_console(
                result["retrieved_chunks"]
            )
        )


def chat_command(
    source_filter: Optional[str] = None,
    show_context: bool = False,
    use_multi_query: bool = False,
) -> None:
    rag_pipeline = RAGPipeline()
    chat_history = []

    print("\nRAG chat session")
    print("-" * 50)
    print("Type 'exit' or 'quit' to stop.")

    if source_filter:
        print(f"Source filter: {source_filter}")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chat session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        result = rag_pipeline.answer_question(
            question=question,
            source_filter=source_filter,
            chat_history=chat_history,
            use_multi_query=use_multi_query,
        )

        print("\nAssistant:")
        print(result["answer"])

        print("\nSources:")
        print(format_sources_for_console(result["sources"]))
        print("\nRetrieval mode:")
        print(result.get("retrieval_mode"))

        if use_multi_query:
            print("\nRetrieval queries:")

            for index, query in enumerate(
                result.get("retrieval_queries", []),
                start=1,
            ):
                print(f"{index}. {query}")
                        
        if show_context:
            print("\nRetrieved context")
            print("-" * 50)
            print(
                format_retrieved_chunks_for_console(
                    result["retrieved_chunks"]
                )
            )

        chat_history = append_turn(
            chat_history=chat_history,
            question=question,
            answer=result["answer"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Document Assistant command-line tool"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "index",
        help="Index the first PDF found in data/raw/",
    )

    subparsers.add_parser(
        "documents",
        help="List indexed documents",
    )

    delete_parser = subparsers.add_parser(
        "delete-document",
        help="Delete one indexed document from the vector store",
    )

    delete_parser.add_argument(
        "source",
        type=str,
        help="Source filename to delete from the vector store",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using indexed documents",
    )

    ask_parser.add_argument(
        "question",
        type=str,
        help="Question to ask the RAG assistant",
    )

    ask_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display retrieved chunks used to generate the answer",
    )

    ask_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Restrict retrieval to one indexed source filename",
    )
    ask_parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Enable optional multi-query retrieval",
    )
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive RAG chat session with memory",
    )

    chat_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Restrict retrieval to one indexed source filename",
    )

    chat_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display retrieved chunks used to generate each answer",
    )
    chat_parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Enable optional multi-query retrieval",
    )
    args = parser.parse_args()

    if args.command == "index":
        index_command()

    elif args.command == "documents":
        documents_command()

    elif args.command == "delete-document":
        delete_document_command(args.source)

    elif args.command == "ask":
        ask_command(
            question=args.question,
            show_context=args.show_context,
            source_filter=args.source,
            use_multi_query=args.multi_query,
        )

    elif args.command == "chat":
        chat_command(
            source_filter=args.source,
            show_context=args.show_context,
            use_multi_query=args.multi_query,
        )

if __name__ == "__main__":
    main()
