"""
ETL – Extract
Loads raw documents from three sources:
  1. Local PDFs from data/photography_ingestion/
  2. Curated photography reference pages (Wikipedia)
  3. Live RSS feeds from photography publications (new articles each week)
"""

import os
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
import requests

logger = logging.getLogger(__name__)

# ── Reference pages (stable, scraped for foundational knowledge) ──────────────
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

# ── RSS feeds (dynamic — new articles ingested each weekly run) ───────────────
RSS_SOURCES: list[str] = [
    "https://petapixel.com/feed/",                     # largest photography news site
    "https://photographylife.com/feed",                # technique guides + gear
    "https://digital-photography-school.com/feed/",   # beginner–intermediate tutorials
]

RSS_MAX_AGE_DAYS = 14   # only ingest articles published in the last 14 days
RSS_MAX_ARTICLES = 10   # cap per feed to avoid token cost spikes

PDF_DIR = "data/photography_ingestion"


# ── Extractors ────────────────────────────────────────────────────────────────

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
    """Fetch and parse reference web pages. Each URL is loaded independently so
    one failure does not abort the entire batch."""
    docs: list[Document] = []
    for url in urls:
        try:
            pages = WebBaseLoader(url).load()
            docs.extend(pages)
            logger.info("Fetched %d doc(s) from %s", len(pages), url)
        except Exception as e:
            logger.error("Failed to fetch '%s': %s", url, e)
    return docs


def extract_rss(
    feeds: list[str] = RSS_SOURCES,
    max_age_days: int = RSS_MAX_AGE_DAYS,
    max_articles: int = RSS_MAX_ARTICLES,
) -> list[Document]:
    """Fetch recent articles from RSS feeds and return them as Documents.

    Only articles published within `max_age_days` are ingested, so each weekly
    Lambda run picks up genuinely new content rather than re-processing old posts.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    docs: list[Document] = []

    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=15, headers={"User-Agent": "PhotoCoachBot/1.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            # RSS 2.0: items live under channel/item
            items = root.findall(".//item")
            ingested = 0

            for item in items:
                if ingested >= max_articles:
                    break

                title = (item.findtext("title") or "").strip()
                link  = (item.findtext("link")  or "").strip()
                desc  = (item.findtext("description") or "").strip()
                pub_date_str = item.findtext("pubDate") or ""

                # Parse publish date and skip old articles
                pub_date = _parse_rss_date(pub_date_str)
                if pub_date and pub_date < cutoff:
                    continue

                # Combine title + description as the document content
                content = f"{title}\n\n{desc}".strip()
                if not content:
                    continue

                docs.append(Document(
                    page_content=content,
                    metadata={"source": link or feed_url, "title": title},
                ))
                ingested += 1

            logger.info("Ingested %d article(s) from %s", ingested, feed_url)

        except Exception as e:
            logger.error("Failed to fetch RSS feed '%s': %s", feed_url, e)

    return docs


def _parse_rss_date(date_str: str) -> datetime | None:
    """Parse an RSS pubDate string into a timezone-aware datetime."""
    if not date_str:
        return None
    # Common RSS date formats
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def extract_all(
    pdf_folder: str = PDF_DIR,
    web_urls: list[str] = WEB_SOURCES,
    rss_feeds: list[str] = RSS_SOURCES,
    rss_max_age_days: int = RSS_MAX_AGE_DAYS,
) -> list[Document]:
    """Run all extractors and return the combined raw document list."""
    pdf_docs = extract_pdfs(pdf_folder)
    web_docs = extract_web(web_urls)
    rss_docs = extract_rss(rss_feeds, max_age_days=rss_max_age_days)

    total = len(pdf_docs) + len(web_docs) + len(rss_docs)
    logger.info(
        "Extraction complete: %d PDF pages + %d reference pages + %d RSS articles = %d total",
        len(pdf_docs), len(web_docs), len(rss_docs), total,
    )
    return pdf_docs + web_docs + rss_docs
