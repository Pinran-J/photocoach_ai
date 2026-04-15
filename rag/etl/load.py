"""
ETL – Load
Idempotent upsert into Pinecone.  Re-running with the same content is safe
because every chunk uses a deterministic MD5 hash as its vector ID.
"""

import logging
from dotenv import dotenv_values
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

INDEX_NAME = "photocoach-ai-index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
UPSERT_BATCH_SIZE = 100  # stay within Pinecone's recommended upsert batch size


def _get_or_create_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it does not already exist."""
    if not pc.has_index(INDEX_NAME):
        logger.info("Creating Pinecone index '%s'…", INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created.")
    else:
        logger.info("Index '%s' already exists — reusing.", INDEX_NAME)


def load(chunks: list[Document], ids: list[str]) -> int:
    """Embed and upsert chunks into Pinecone.
    Using explicit IDs means Pinecone will overwrite any existing vector with
    the same ID, making this operation fully idempotent.

    Returns the number of vectors upserted.
    """
    if not chunks:
        logger.warning("No chunks to load — skipping.")
        return 0

    config = dotenv_values(".env")
    pc = Pinecone(api_key=config["PINECONE_API_KEY"])
    _get_or_create_index(pc)

    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    total = 0
    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch_chunks = chunks[i : i + UPSERT_BATCH_SIZE]
        batch_ids = ids[i : i + UPSERT_BATCH_SIZE]
        vector_store.add_documents(batch_chunks, ids=batch_ids)
        total += len(batch_chunks)
        logger.info("Upserted batch %d/%d (%d vectors so far)", i // UPSERT_BATCH_SIZE + 1,
                    -(-len(chunks) // UPSERT_BATCH_SIZE), total)

    logger.info("Load complete: %d vectors upserted to '%s'.", total, INDEX_NAME)
    return total
