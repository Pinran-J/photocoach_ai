"""
ETL – Load
Idempotent upsert into Pinecone.  Re-running with the same content is safe
because every chunk uses a deterministic MD5 hash as its vector ID.

Uses the Pinecone REST API directly (not langchain-pinecone) to avoid the
SDK's internal ThreadPool, which requires POSIX semaphores unavailable in Lambda.
"""

import logging
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

INDEX_NAME = "photocoach-ai-index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
UPSERT_BATCH_SIZE = 50


def _get_or_create_index(pc: Pinecone) -> str:
    """Create the Pinecone index if it does not already exist. Returns the index host."""
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
    return pc.describe_index(INDEX_NAME).host


def load(chunks: list[Document], ids: list[str]) -> int:
    """Embed and upsert chunks into Pinecone via direct REST API.

    Returns the number of vectors upserted.
    """
    if not chunks:
        logger.warning("No chunks to load — skipping.")
        return 0

    load_dotenv()  # no-op in Lambda (no .env), loads from file locally
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    openai_api_key = os.environ["OPENAI_API_KEY"]

    pc = Pinecone(api_key=pinecone_api_key)
    index_host = _get_or_create_index(pc)
    openai_client = OpenAI(api_key=openai_api_key)

    total = 0
    num_batches = -(-len(chunks) // UPSERT_BATCH_SIZE)  # ceiling division

    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch_chunks = chunks[i : i + UPSERT_BATCH_SIZE]
        batch_ids = ids[i : i + UPSERT_BATCH_SIZE]
        batch_texts = [c.page_content for c in batch_chunks]

        # Embed via OpenAI SDK
        emb_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )
        embeddings = [e.embedding for e in emb_response.data]

        # Build Pinecone vector dicts
        vectors = [
            {
                "id": vid,
                "values": emb,
                "metadata": {"text": text, **chunk.metadata},
            }
            for vid, emb, text, chunk in zip(batch_ids, embeddings, batch_texts, batch_chunks)
        ]

        # Upsert directly via Pinecone REST API — avoids SDK ThreadPool/semaphore issue in Lambda
        resp = requests.post(
            f"https://{index_host}/vectors/upsert",
            json={"vectors": vectors},
            headers={
                "Api-Key": pinecone_api_key,
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()

        total += len(batch_chunks)
        logger.info(
            "Upserted batch %d/%d (%d vectors so far)",
            i // UPSERT_BATCH_SIZE + 1,
            num_batches,
            total,
        )

    logger.info("Load complete: %d vectors upserted to '%s'.", total, INDEX_NAME)
    return total
