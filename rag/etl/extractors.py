"""
ETL – Extract
Loads raw documents from two sources:
  1. Local PDFs from data/photography_ingestion/
  2. Curated photography web pages (Wikipedia + open guides)
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Curated list of freely-scrapable photography knowledge pages.
# Add or remove URLs here as the knowledge base grows.
WEB_SOURCES: list[str] = [
    # Foundational exposure / technical
    "https://en.wikipedia.org/wiki/Exposure_(photography)",
    "https://en.wikipedia.org/wiki/Aperture",
    "https://en.wikipedia.org/wiki/Shutter_speed",
    "https://en.wikipedia.org/wiki/Film_speed",
    "https://en.wikipedia.org/wiki/Depth_of_field",
    "https://en.wikipedia.org/wiki/White_balance",
    "https://en.wikipedia.org/wiki/Bokeh",
    "https://en.wikipedia.org/wiki/Camera_lens",
    # Composition
    "https://en.wikipedia.org/wiki/Rule_of_thirds",
    "https://en.wikipedia.org/wiki/Composition_(visual_arts)",
    "https://en.wikipedia.org/wiki/Golden_ratio",
    "https://en.wikipedia.org/wiki/Leading_line",
    # Genres
    "https://en.wikipedia.org/wiki/Street_photography",
    "https://en.wikipedia.org/wiki/Portrait_photography",
    "https://en.wikipedia.org/wiki/Landscape_photography",
    "https://en.wikipedia.org/wiki/Macro_photography",
    "https://en.wikipedia.org/wiki/Night_photography",
    # Lighting
    "https://en.wikipedia.org/wiki/Rembrandt_lighting",
    "https://en.wikipedia.org/wiki/Fill_light",
    "https://en.wikipedia.org/wiki/Natural_lighting",
]

PDF_DIR = "data/photography_ingestion"


def extract_pdfs(folder: str = PDF_DIR) -> list[Document]:
    """Load all PDF files from the specified folder."""
    docs: list[Document] = []
    if not os.path.isdir(folder):
        logger.warning("PDF directory '%s' not found — skipping PDF extraction.", folder)
        return docs

    pdf_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", folder)
        return docs

    for path in pdf_files:
        try:
            pages = PyPDFLoader(path).load()
            docs.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), path)
        except Exception as e:
            logger.error("Failed to load PDF '%s': %s", path, e)

    return docs


def extract_web(urls: list[str] = WEB_SOURCES) -> list[Document]:
    """Fetch and parse web pages. Each URL is loaded independently so one
    failure does not abort the entire batch."""
    docs: list[Document] = []
    for url in urls:
        try:
            pages = WebBaseLoader(url).load()
            docs.extend(pages)
            logger.info("Fetched %d doc(s) from %s", len(pages), url)
        except Exception as e:
            logger.error("Failed to fetch '%s': %s", url, e)

    return docs


def extract_all(
    pdf_folder: str = PDF_DIR,
    web_urls: list[str] = WEB_SOURCES,
) -> list[Document]:
    """Run both extractors and return the combined raw document list."""
    pdf_docs = extract_pdfs(pdf_folder)
    web_docs = extract_web(web_urls)
    total = len(pdf_docs) + len(web_docs)
    logger.info(
        "Extraction complete: %d PDF pages + %d web pages = %d total",
        len(pdf_docs),
        len(web_docs),
        total,
    )
    return pdf_docs + web_docs
