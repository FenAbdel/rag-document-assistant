from pathlib import Path
import sys
from typing import Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    RAW_DATA_DIR,
)
from app.indexing.vector_store import (
    delete_chunks_by_source,
    get_vector_store_count,
    list_indexed_documents,
)
from app.memory.conversation_memory import append_turn
from app.pipelines.indexing_pipeline import DocumentIndexingPipeline
from app.pipelines.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save an uploaded PDF into the local raw data folder.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(uploaded_file.name).name
    file_path = RAW_DATA_DIR / safe_filename

    with open(file_path, "wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    return file_path


@st.cache_resource
def get_indexing_pipeline() -> DocumentIndexingPipeline:
    """
    Cache indexing services across Streamlit reruns.
    """
    return DocumentIndexingPipeline()


@st.cache_resource
def get_rag_pipeline() -> RAGPipeline:
    """
    Cache RAG services across Streamlit reruns.
    """
    return RAGPipeline()


def get_source_options() -> List[str]:
    """
    Return indexed source filenames for UI selection.
    """
    documents = list_indexed_documents()
    return [document["source"] for document in documents]


def display_indexing_stats(stats: Dict) -> None:
    """
    Display document indexing statistics in the UI.
    """
    st.success("Document indexed successfully.")

    st.caption("Document ID: {}".format(stats["document_id"]))
    st.caption("Content hash: {}".format(stats["content_hash"]))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Deleted old chunks", stats["deleted_existing_chunks"])
    col2.metric("Pages", stats["pages_extracted"])
    col3.metric("Chunks", stats["chunks_created"])
    col4.metric("Stored chunks", stats["chunks_stored"])

    st.caption(
        "Total chunks in vector store: "
        "{}".format(stats["total_chunks_in_vector_store"])
    )


def display_sources(sources: List[Dict]) -> None:
    """
    Display sources as clean source cards.
    """
    if not sources:
        st.info("No sources available.")
        return

    for source in sources:
        distance = source.get("distance")

        if isinstance(distance, float):
            distance_text = "{:.4f}".format(distance)
        else:
            distance_text = str(distance)

        with st.container(border=True):
            st.markdown("**{}**".format(source.get("label")))
            st.write("Document:", source.get("source"))
            st.write("Page:", source.get("page"))
            st.caption("Chunk ID: {}".format(source.get("chunk_id")))
            st.caption("Distance: {}".format(distance_text))


def display_retrieved_context(retrieved_chunks: List[Dict]) -> None:
    """
    Display retrieved chunks for retrieval debugging.
    """
    if not retrieved_chunks:
        st.info("No retrieved context available.")
        return

    for chunk in retrieved_chunks:
        distance = chunk.get("distance")

        if isinstance(distance, float):
            distance_text = "{:.4f}".format(distance)
        else:
            distance_text = str(distance)

        title = "Result {rank} — {source}, page {page}, distance {distance}".format(
            rank=chunk.get("rank"),
            source=chunk.get("source"),
            page=chunk.get("page"),
            distance=distance_text,
        )

        with st.expander(title):
            st.caption("Chunk ID: {}".format(chunk.get("chunk_id")))

            retrieval_query = chunk.get("retrieval_query")
            if retrieval_query:
                st.caption("Retrieved by query: {}".format(retrieval_query))

            st.write(chunk.get("content"))


def display_indexed_documents() -> None:
    """
    Display indexed documents and allow deletion.
    """
    st.header("Indexed documents")

    try:
        indexed_documents = list_indexed_documents()

        if not indexed_documents:
            st.caption("No indexed documents yet.")
            return

        for document in indexed_documents:
            with st.expander(document["source"]):
                st.write("Chunks:", document["chunk_count"])
                st.write("Pages:", document["page_count"])

                document_ids = document.get("document_ids", [])
                content_hashes = document.get("content_hashes", [])

                if document_ids:
                    st.caption("Document ID: {}".format(document_ids[0]))

                if content_hashes:
                    st.caption("Content hash: {}".format(content_hashes[0]))

        document_to_delete = st.selectbox(
            "Delete indexed document",
            options=[""] + [
                document["source"]
                for document in indexed_documents
            ],
        )

        if st.button("Delete selected document", use_container_width=True):
            if not document_to_delete:
                st.warning("Please select a document to delete.")
            else:
                deleted_count = delete_chunks_by_source(document_to_delete)
                st.success(
                    "Deleted {} chunks from {}.".format(
                        deleted_count,
                        document_to_delete,
                    )
                )
                st.rerun()

    except Exception as error:
        st.warning("Could not load indexed documents.")
        st.caption(str(error))


def display_project_settings() -> None:
    """
    Display current important configuration values.
    """
    with st.expander("Current configuration", expanded=False):
        st.write("Embedding model:", EMBEDDING_MODEL)
        st.write("Generation model:", GENERATION_MODEL)
        st.write("Chunk size:", CHUNK_SIZE)
        st.write("Chunk overlap:", CHUNK_OVERLAP)
        st.write("Default top_k:", DEFAULT_TOP_K)


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state variables.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None


def render_chat_history() -> None:
    """
    Render previous user and assistant messages.
    """
    if not st.session_state["chat_history"]:
        st.info("Ask a question to start the conversation.")
        return

    for message in st.session_state["chat_history"]:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            with st.chat_message("user"):
                st.write(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)


def main() -> None:
    initialize_session_state()

    st.title("📄 RAG Document Assistant")
    st.caption(
        "Upload PDFs, index them into a local vector store, "
        "and ask grounded questions with citations."
    )

    with st.sidebar:
        st.header("Document indexing")

        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
        )

        if st.button("Index document", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload a PDF file first.")
            else:
                with st.spinner("Indexing document..."):
                    file_path = save_uploaded_file(uploaded_file)
                    indexing_pipeline = get_indexing_pipeline()
                    stats = indexing_pipeline.index_pdf(file_path)

                st.session_state["last_indexed_document"] = stats["document"]
                display_indexing_stats(stats)

        st.divider()

        try:
            vector_store_count = get_vector_store_count()
            st.metric("Chunks in vector store", vector_store_count)
        except Exception as error:
            st.warning("Vector store is not available yet.")
            st.caption(str(error))

        if "last_indexed_document" in st.session_state:
            st.caption(
                "Last indexed document: {}".format(
                    st.session_state["last_indexed_document"]
                )
            )

        st.divider()
        display_indexed_documents()

        st.divider()
        display_project_settings()

    source_options = get_source_options()

    if not source_options:
        st.warning(
            "No indexed documents found. Upload and index a PDF before asking questions."
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        retrieval_scope = st.selectbox(
            "Retrieval scope",
            options=["All indexed documents"] + source_options,
        )

    with col2:
        use_multi_query = st.checkbox(
            "Multi-query retrieval",
            value=False,
            help=(
                "Generates alternative search queries. "
                "Useful for vague questions, but costs more API calls."
            ),
        )

    source_filter: Optional[str] = None

    if retrieval_scope != "All indexed documents":
        source_filter = retrieval_scope

    show_context = st.checkbox(
        "Show retrieved context",
        value=False,
        help="Useful for debugging retrieval quality.",
    )

    st.subheader("Conversation")

    render_chat_history()

    question = st.chat_input("Ask a question about the indexed documents")

    if question:
        if not source_options:
            st.warning("Please index at least one document first.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving context and generating answer..."):
            rag_pipeline = get_rag_pipeline()
            result = rag_pipeline.answer_question(
                question=question,
                source_filter=source_filter,
                chat_history=st.session_state["chat_history"],
                use_multi_query=use_multi_query,
            )

        with st.chat_message("assistant"):
            st.write(result["answer"])

        st.session_state["chat_history"] = append_turn(
            chat_history=st.session_state["chat_history"],
            question=question,
            answer=result["answer"],
        )

        st.session_state["last_result"] = result

    last_result = st.session_state.get("last_result")

    if last_result:
        st.divider()

        st.subheader("Sources")
        display_sources(last_result["sources"])

        with st.expander("Retrieval details", expanded=False):
            st.write("Retrieval mode:", last_result.get("retrieval_mode"))

            retrieval_queries = last_result.get("retrieval_queries", [])

            if retrieval_queries:
                st.write("Retrieval queries:")

                for index, retrieval_query in enumerate(
                    retrieval_queries,
                    start=1,
                ):
                    st.markdown("{}. {}".format(index, retrieval_query))

        if show_context:
            st.subheader("Retrieved context")
            display_retrieved_context(last_result["retrieved_chunks"])

    with st.expander("Conversation memory", expanded=False):
        if not st.session_state["chat_history"]:
            st.caption("No conversation memory yet.")
        else:
            for message in st.session_state["chat_history"]:
                st.markdown(
                    "**{}:** {}".format(
                        message["role"].capitalize(),
                        message["content"],
                    )
                )

        if st.button("Clear conversation memory"):
            st.session_state["chat_history"] = []
            st.session_state["last_result"] = None
            st.rerun()


if __name__ == "__main__":
    main()
