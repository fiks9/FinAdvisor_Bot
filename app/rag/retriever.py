"""
app/rag/retriever.py — Semantic search over ChromaDB
=====================================================
Single public function: retrieve(query) → list of relevant text strings.
Called by the LangChain chain to build the {context} placeholder.

How it works:
  1. Query text → embedding vector (same model used during ingestion)
  2. ChromaDB finds the N nearest vectors by cosine similarity
  3. Returns the page_content of the top-K chunks as plain strings

Why plain strings (not Documents)?
  The prompt template expects a single formatted string for {context}.
  Converting here keeps chain.py clean.
"""

import logging

from app.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Find the most semantically relevant document chunks for a user query.

    Args:
        query: The user's question in plain text (Ukrainian or Russian).
        k:     Number of chunks to return. Keep low (3–5) to avoid
               overflowing the LLM context window.

    Returns:
        List of text strings, ordered by relevance (most relevant first).
        Returns an empty list if the collection is empty or no match found.
    """
    vectorstore = get_vectorstore()

    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error("ChromaDB similarity_search failed: %s", e)
        return []

    chunks = [doc.page_content for doc in results]

    # Log retrieved sources for debugging (helps tune chunk size)
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("filename", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        logger.debug("RAG result %d/%d [%s]: %s…", i, k, src, preview)

    logger.info("Retrieved %d chunks for query: %r", len(chunks), query[:60])
    return chunks


def format_context(chunks: list[str]) -> str:
    """
    Join retrieved chunks into a single formatted context string
    ready to insert into the LLM prompt.

    Separates chunks with a clear divider so the LLM understands
    these are separate excerpts, not one continuous text.

    Returns empty string if no chunks — the prompt handles this gracefully.
    """
    if not chunks:
        return ""

    formatted = "\n\n---\n\n".join(chunks)
    return f"Релевантні уривки з фінансових документів:\n\n{formatted}"
