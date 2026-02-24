"""
app/rag/embeddings.py — Local Sentence Transformers embedding wrapper
======================================================================
Wraps sentence-transformers to be compatible with the LangChain
Embeddings interface, which ChromaDB and the retriever expect.

Why local embeddings?
  - Zero API cost (runs on CPU, no network call)
  - No rate limits — safe for bulk ingestion of documents
  - all-MiniLM-L6-v2 is a proven, compact (80MB) model with good
    semantic similarity quality for European languages including Ukrainian.

Performance note:
  First call downloads the model (~80MB) to ~/.cache/huggingface/hub/.
  Subsequent calls load from cache — fast startup.
  Embedding 100 chunks takes ~1–3 seconds on a modern CPU.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    @lru_cache ensures the model is loaded from disk exactly once per
    process lifetime — not on every call. This saves ~1–2 seconds per request.

    Args:
        model_name: Sentence-transformers model identifier.
                    We use all-MiniLM-L6-v2 — fast, small, good quality.

    Returns:
        LangChain-compatible Embeddings object ready to use with ChromaDB.
    """
    logger.info("Loading embedding model: %s", model_name)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # force CPU — safe for Railway free tier
        encode_kwargs={
            "normalize_embeddings": True,  # L2 normalization → cosine similarity ≡ dot product
            "batch_size": 32,              # process 32 chunks at a time during ingestion
        },
    )

    logger.info("Embedding model loaded successfully")
    return embeddings
