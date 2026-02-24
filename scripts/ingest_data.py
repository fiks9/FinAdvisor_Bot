"""
scripts/ingest_data.py — One-time ChromaDB ingestion script
============================================================
Run this script ONCE (or when documents change) to:
  1. Load & chunk all .txt files from data/docs/
  2. Generate embeddings via all-MiniLM-L6-v2
  3. Persist vectors into ChromaDB at CHROMA_PATH

Usage:
  python scripts/ingest_data.py

TODO (Крок 12): implement full ingestion pipeline
"""
