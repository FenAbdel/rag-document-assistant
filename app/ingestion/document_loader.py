import hashlib
import re
from pathlib import Path
from typing import Dict, List

import pymupdf


def compute_file_hash(file_path: Path) -> str:
    """
    Compute a SHA-256 hash for a file.

    This helps us identify the exact content version of a document.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


def build_document_id(file_path: Path, content_hash: str) -> str:
    """
    Build a readable and stable document ID from filename and content hash.
    """
    filename_stem = file_path.stem.lower()

    clean_stem = re.sub(r"[^a-z0-9]+", "_", filename_stem)
    clean_stem = clean_stem.strip("_")

    short_hash = content_hash[:12]

    return "{}_{}".format(clean_stem, short_hash)


def load_pdf_pages(file_path: Path) -> List[Dict]:
    """
    Load a PDF file and extract text page by page.

    Returns:
        A list of dictionaries.
        Each dictionary represents one page with its text and metadata.
    """
    if not file_path.exists():
        raise FileNotFoundError("File not found: {}".format(file_path))

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Unsupported file type: {}".format(file_path.suffix))

    content_hash = compute_file_hash(file_path)
    document_id = build_document_id(
        file_path=file_path,
        content_hash=content_hash,
    )

    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "content": text,
                    "metadata": {
                        "document_id": document_id,
                        "source": file_path.name,
                        "page": page_index,
                        "content_hash": content_hash,
                    },
                }
            )

    return pages