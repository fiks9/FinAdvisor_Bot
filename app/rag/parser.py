"""
app/rag/parser.py — Document loader and recursive text splitter
===============================================================
Reads all .txt files from the docs directory and splits them into
overlapping chunks ready for embedding.

Why chunks?
  An LLM has a limited context window. Instead of feeding the entire
  document (potentially 50k tokens), we store small chunks and retrieve
  only the 3 most relevant ones at query time.

Why overlap?
  If we split at exactly 800 chars, a sentence might be cut in two pieces.
  An overlap of 100 chars ensures the beginning of each chunk repeats
  the end of the previous one, so no sentence ever loses its context.
"""

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Separators tried in order — RecursiveCharacterTextSplitter falls back
# to the next separator when the chunk is still too large after splitting.
_SEPARATORS = [
    "\n=== ",   # our custom section headers  (e.g. === ЛІМІТИ ===)
    "\n\n",     # paragraph break
    "\n",       # line break
    ". ",       # sentence boundary
    " ",        # word boundary
    "",         # character boundary (last resort)
]


def load_documents(docs_dir: Path) -> list[Document]:
    """
    Load all .txt files from docs_dir.
    Each file becomes one Document with metadata containing:
      - source: relative file path (shown to user as citation)
      - filename: just the stem, e.g. "payment_limits_faq"
    """
    docs: list[Document] = []

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", docs_dir)
        return docs

    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={
                "source":   str(path),
                "filename": path.stem,
            },
        )
        docs.append(doc)
        logger.debug("Loaded %s (%d chars)", path.name, len(text))

    logger.info("Loaded %d document(s) from %s", len(docs), docs_dir)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size:    int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split a list of Documents into smaller overlapping chunks.

    Each output chunk keeps the metadata of its source document, plus:
      - chunk_index: position of this chunk within its source file

    Args:
        docs:          List of Documents to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: How many characters to repeat across consecutive chunks.

    Returns:
        Flat list of chunk Documents, ordered by (source, position).
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []

    for doc in docs:
        chunks = splitter.split_documents([doc])
        # Enrich metadata with chunk position for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        all_chunks.extend(chunks)
        logger.debug(
            "Split '%s' → %d chunks",
            doc.metadata.get("filename", "unknown"),
            len(chunks),
        )

    logger.info(
        "Total chunks: %d (from %d document(s), avg %.0f chars/chunk)",
        len(all_chunks),
        len(docs),
        sum(len(c.page_content) for c in all_chunks) / max(len(all_chunks), 1),
    )
    return all_chunks


def load_and_split(
    docs_dir:      Path,
    chunk_size:    int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Convenience wrapper: load + split in one call."""
    docs   = load_documents(docs_dir)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks
