import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def get_int_env(variable_name: str, default_value: int) -> int:
    """
    Read an integer environment variable with a safe default.
    """
    raw_value = os.getenv(variable_name)

    if raw_value is None:
        return default_value

    try:
        return int(raw_value)
    except ValueError:
        raise ValueError(
            "{} must be an integer. Got: {}".format(
                variable_name,
                raw_value,
            )
        )


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

SUPPORTED_FILE_EXTENSIONS = {".pdf"}

CHUNK_SIZE = get_int_env("CHUNK_SIZE", 800)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 150)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)
EMBEDDING_BATCH_SIZE = get_int_env("EMBEDDING_BATCH_SIZE", 64)

DEFAULT_TOP_K = get_int_env("DEFAULT_TOP_K", 3)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gpt-4.1-mini",
)
MAX_OUTPUT_TOKENS = get_int_env("MAX_OUTPUT_TOKENS", 700)

MAX_MEMORY_MESSAGES = get_int_env("MAX_MEMORY_MESSAGES", 6)

MULTI_QUERY_COUNT = get_int_env("MULTI_QUERY_COUNT", 3)
MULTI_QUERY_TOP_K_PER_QUERY = get_int_env(
    "MULTI_QUERY_TOP_K_PER_QUERY",
    3,
)
MULTI_QUERY_FINAL_TOP_K = get_int_env(
    "MULTI_QUERY_FINAL_TOP_K",
    3,
)
QUERY_REWRITE_MODEL = os.getenv(
    "QUERY_REWRITE_MODEL",
    GENERATION_MODEL,
)