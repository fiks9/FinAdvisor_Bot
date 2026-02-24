"""
app/rag/vectorstore.py — ChromaDB persistent collection manager
===============================================================
Manages a single ChromaDB collection that holds all document chunks.

Key design decisions:
  - PersistentClient: data survives process restarts (crucial for Railway)
  - Single collection named "financial_docs"
  - Embeddings provided externally (from embeddings.py) so ChromaDB
    never calls any API on its own

Usage flow:
  1. Run scripts/ingest_data.py ONCE to populate the DB
  2. init_vectorstore() in post_init() opens the existing collection
  3. get_vectorstore() returns it to the retriever on every query
"""

import logging
from pathlib import Path
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "financial_docs"

# Module-level store reference set by init_vectorstore()
_vectorstore: Chroma | None = None


def init_vectorstore(
    chroma_path: Path,
    embeddings:  HuggingFaceEmbeddings,
) -> Chroma:
    """
    Open (or create) the ChromaDB collection at chroma_path.
    Called once at startup from post_init() in main.py.

    If the collection is empty (first run), the bot will still start,
    but answers won't have document context until ingest_data.py is run.

    Args:
        chroma_path: Directory where ChromaDB stores its data files.
        embeddings:  The embedding function to use (must match what was
                     used during ingestion, otherwise search will fail silently).

    Returns:
        A LangChain Chroma vectorstore ready for similarity_search().
    """
    global _vectorstore

    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    _vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    count = _vectorstore._collection.count()
    logger.info(
        "ChromaDB opened at '%s' | collection '%s' | %d chunks",
        chroma_path,
        COLLECTION_NAME,
        count,
    )

    if count == 0:
        logger.warning(
            "ChromaDB collection is empty! "
            "Run 'python scripts/ingest_data.py' to populate it."
        )

    return _vectorstore


def get_vectorstore() -> Chroma:
    """
    Return the initialised vectorstore.
    Raises RuntimeError if called before init_vectorstore().
    """
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore not initialised. Call init_vectorstore() in post_init()."
        )
    return _vectorstore
