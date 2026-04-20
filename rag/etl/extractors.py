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
from bs4 import BeautifulSoup

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
    "https://en.wikipedia.org/wiki/Exposure_triangle",
    "https://en.wikipedia.org/wiki/Histogram_(photography)",
    "https://en.wikipedia.org/wiki/Color_temperature",
    "https://en.wikipedia.org/wiki/Focal_length",
    "https://en.wikipedia.org/wiki/Long-exposure_photography",
    "https://en.wikipedia.org/wiki/High-dynamic-range_imaging",
    "https://en.wikipedia.org/wiki/Flash_(photography)",
    # Composition
    "https://en.wikipedia.org/wiki/Rule_of_thirds",
    "https://en.wikipedia.org/wiki/Composition_(visual_arts)",
    "https://en.wikipedia.org/wiki/Golden_ratio",
    "https://en.wikipedia.org/wiki/Leading_line",
    "https://en.wikipedia.org/wiki/Negative_space",
    "https://en.wikipedia.org/wiki/Symmetry",
    # Genres
    "https://en.wikipedia.org/wiki/Street_photography",
    "https://en.wikipedia.org/wiki/Portrait_photography",
    "https://en.wikipedia.org/wiki/Landscape_photography",
    "https://en.wikipedia.org/wiki/Macro_photography",
    "https://en.wikipedia.org/wiki/Night_photography",
    "https://en.wikipedia.org/wiki/Wildlife_photography",
    "https://en.wikipedia.org/wiki/Sports_photography",
    "https://en.wikipedia.org/wiki/Documentary_photography",
    # Lighting
    "https://en.wikipedia.org/wiki/Rembrandt_lighting",
    "https://en.wikipedia.org/wiki/Fill_light",
    "https://en.wikipedia.org/wiki/Natural_lighting",
    "https://en.wikipedia.org/wiki/Backlighting_(photography)",
    "https://en.wikipedia.org/wiki/Catchlight",
]

# ── RSS feeds (dynamic — new articles ingested each weekly run) ───────────────
RSS_SOURCES: list[str] = [
    "https://petapixel.com/feed/",                # largest photography news site
    "https://photographylife.com/feed",           # technique guides + gear
    "https://fstoppers.com/rss.xml",              # professional-level critiques
    "https://www.thephoblographer.com/feed/",     # street + mirrorless focused
    "https://www.dpreview.com/feeds/articles",    # industry-standard reviews
    "https://www.naturettl.com/feed/",            # wildlife + nature technique
    "https://shotkit.com/feed/",                  # gear reviews + practical technique
    "https://www.outdoorphotographer.com/feed/",  # landscape + outdoor technique
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
    """Fetch recent articles from RSS feeds.

    For each article, attempts to fetch the full article body from the link URL.
    Falls back to the RSS description snippet if the full fetch fails.
    Only articles published within `max_age_days` are ingested.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    docs: list[Document] = []

    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=15, headers={"User-Agent": "PhotoCoachBot/1.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            items = root.findall(".//item")
            ingested = 0

            for item in items:
                if ingested >= max_articles:
                    break

                title        = (item.findtext("title") or "").strip()
                link         = (item.findtext("link")  or "").strip()
                desc         = (item.findtext("description") or "").strip()
                pub_date_str = item.findtext("pubDate") or ""

                pub_date = _parse_rss_date(pub_date_str)
                if pub_date and pub_date < cutoff:
                    continue

                # Try to fetch the full article — much richer than the RSS snippet
                full_text = _fetch_article_text(link) if link else ""
                content = f"{title}\n\n{full_text or desc}".strip()
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


def _fetch_article_text(url: str) -> str:
    """Fetch and extract the main body text from an article URL.

    Strips nav, footer, ads, and scripts, then tries common article
    content selectors before falling back to full page text.
    Returns empty string on any failure so the caller falls back to
    the RSS description snippet.
    """
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "PhotoCoachBot/1.0"},
            allow_redirects=True,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["nav", "footer", "script", "style", "aside", "header", "form"]):
            tag.decompose()

        for selector in [
            "article",
            "main",
            ".post-content",
            ".entry-content",
            ".article-body",
            ".article-content",
            ".content-body",
            "#content",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text

        return soup.get_text(separator="\n", strip=True)

    except Exception:
        return ""


def _parse_rss_date(date_str: str) -> datetime | None:
    """Parse an RSS pubDate string into a timezone-aware datetime."""
    if not date_str:
        return None
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
