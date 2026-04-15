"""
ETL – Pipeline orchestrator
Runs Extract → Transform → Load in sequence and reports a summary.
Call run_pipeline() directly or via the scheduler in schedule.py.
"""

import logging
import time
from rag.etl.extractors import extract_all, WEB_SOURCES, PDF_DIR
from rag.etl.transform import transform
from rag.etl.load import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_folder: str = PDF_DIR,
    web_urls: list[str] = WEB_SOURCES,
) -> dict:
    """
    Full ETL run.  Safe to call multiple times — idempotent via content-hash IDs.

    Returns a summary dict with counts at each stage.
    """
    start = time.time()
    logger.info("=== PhotoCoach ETL pipeline started ===")

    # --- Extract ---
    logger.info("Stage 1/3: Extract")
    raw_docs = extract_all(pdf_folder=pdf_folder, web_urls=web_urls)

    # --- Transform ---
    logger.info("Stage 2/3: Transform")
    chunks, ids = transform(raw_docs)

    # --- Load ---
    logger.info("Stage 3/3: Load")
    upserted = load(chunks, ids)

    elapsed = time.time() - start
    summary = {
        "raw_documents": len(raw_docs),
        "chunks_after_transform": len(chunks),
        "vectors_upserted": upserted,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("=== Pipeline complete in %.1fs: %s ===", elapsed, summary)
    return summary


if __name__ == "__main__":
    run_pipeline()
