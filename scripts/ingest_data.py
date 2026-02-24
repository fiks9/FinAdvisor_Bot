"""
scripts/ingest_data.py — One-time ChromaDB ingestion script
============================================================
Run this script ONCE (or whenever documents in data/docs/ change) to:
  1. Load all .txt files from data/docs/
  2. Split them into overlapping chunks
  3. Generate embeddings via all-MiniLM-L6-v2 (locally, no API cost)
  4. Persist vectors into ChromaDB at the path from config

Usage:
    python scripts/ingest_data.py

Re-running is safe — it clears the collection first, then re-ingests.
This avoids duplicate chunks if documents were edited.

Typical runtime: 10–30 seconds (first run downloads ~80MB model).
"""

import sys
import logging
from pathlib import Path

# Make sure project root is on the path so `from app.xxx` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.logger import setup_logging
from app.rag.parser import load_and_split
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import COLLECTION_NAME

import chromadb
from langchain_chroma import Chroma

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def ingest() -> None:
    cfg = get_settings()

    docs_dir    = Path("data/docs")
    chroma_path = cfg.CHROMA_PATH

    # ── 1. Load & split documents ─────────────────────────────────────
    logger.info("Loading documents from %s …", docs_dir)
    chunks = load_and_split(
        docs_dir=docs_dir,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

    if not chunks:
        logger.error("No chunks found. Add .txt files to %s and try again.", docs_dir)
        sys.exit(1)

    logger.info("Prepared %d chunks for ingestion", len(chunks))

    # ── 2. Load embedding model ───────────────────────────────────────
    logger.info("Loading embedding model: %s …", cfg.EMBEDDING_MODEL)
    embeddings = get_embeddings(cfg.EMBEDDING_MODEL)

    # ── 3. Connect to ChromaDB and reset collection ───────────────────
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete existing collection to avoid duplicates on re-ingestion
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)

    # ── 4. Ingest chunks into ChromaDB ────────────────────────────────
    logger.info("Generating embeddings and inserting into ChromaDB …")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    final_count = vectorstore._collection.count()
    logger.info("Ingestion complete! %d chunks stored in '%s'", final_count, chroma_path)

    # ── 5. Quick sanity search ────────────────────────────────────────
    logger.info("Running sanity search for 'ліміт переказу' …")
    results = vectorstore.similarity_search("ліміт переказу", k=2)
    for i, r in enumerate(results, 1):
        preview = r.page_content[:120].replace("\n", " ")
        logger.info("  Result %d [%s]: %s…", i, r.metadata.get("filename"), preview)

    logger.info("Ingestion pipeline finished successfully.")


if __name__ == "__main__":
    ingest()
