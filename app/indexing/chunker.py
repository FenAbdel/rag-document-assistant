from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Split extracted PDF pages into smaller chunks.

    Each output chunk keeps the original page metadata
    and receives a unique chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []

    for page in pages:
        page_content = page["content"]
        metadata = page["metadata"]

        split_texts = splitter.split_text(page_content)
        document_id = metadata.get("document_id")

        if not document_id:
            document_id = Path(metadata.get("source", "document")).stem

        for chunk_index, chunk_text in enumerate(split_texts, start=1):
            chunk_id = "{}_p{}_c{}".format(
                document_id,
                metadata["page"],
                chunk_index,
            )

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                    },
                }
            )

    return chunks
