from pathlib import Path

from app.ingestion.document_loader import build_document_id


def test_build_document_id_creates_clean_id():
    file_path = Path("My CV Final.pdf")
    content_hash = "abcdef1234567890"

    document_id = build_document_id(
        file_path=file_path,
        content_hash=content_hash,
    )

    assert document_id == "my_cv_final_abcdef123456"