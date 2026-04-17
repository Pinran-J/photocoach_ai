"""
ETL – Transform
Cleans raw documents, splits them into chunks, and deduplicates by content hash.
"""

import hashlib
import logging
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Smaller chunks than the old pipeline (600 vs 1000 tokens) so retrieval
# surfaces more precise, self-contained tips rather than long passages.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def _clean_text(text: str) -> str:
    """Remove boilerplate noise that degrades embedding quality."""
    # Collapse runs of whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Strip common wiki/PDF artefacts (edit links, citation brackets)
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def _content_hash(doc: Document) -> str:
    """MD5 of (source + content) — used as the deterministic Pinecone vector ID."""
    raw = (doc.metadata.get("source", "") + doc.page_content).encode()
    return hashlib.md5(raw).hexdigest()


def clean_documents(docs: list[Document]) -> list[Document]:
    """Apply text cleaning to every document in-place (returns new list)."""
    cleaned = []
    for doc in docs:
        cleaned_content = _clean_text(doc.page_content)
        if len(cleaned_content) < 50:  # skip near-empty pages
            continue
        cleaned.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    logger.info("Cleaning: %d → %d documents (removed %d near-empty)", len(docs), len(cleaned), len(docs) - len(cleaned))
    return cleaned


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks using tiktoken tokenisation."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Chunking: %d documents → %d chunks", len(docs), len(chunks))
    return chunks


def deduplicate(chunks: list[Document]) -> tuple[list[Document], list[str]]:
    """Remove exact duplicate chunks by content hash.
    Returns (unique_chunks, ids) where ids are deterministic MD5 strings
    suitable for use as Pinecone vector IDs (idempotent upserts)."""
    seen: set[str] = set()
    unique_chunks: list[Document] = []
    ids: list[str] = []

    for chunk in chunks:
        chunk_id = _content_hash(chunk)
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_chunks.append(chunk)
            ids.append(chunk_id)

    removed = len(chunks) - len(unique_chunks)
    if removed:
        logger.info("Deduplication: removed %d duplicate chunks (%d remaining)", removed, len(unique_chunks))
    return unique_chunks, ids


def transform(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """Full transform stage: clean → chunk → deduplicate.
    Returns (chunks, ids) ready for the load stage."""
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    chunks, ids = deduplicate(chunks)
    return chunks, ids
