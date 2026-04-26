"""
Goodreads book scraper.

Scrapes up to 50,000 books with title, author, genres, and description.
Saves results incrementally to data/books_scraped.csv with checkpoint support
so interrupted runs can be resumed.

Usage:
    python src/extract.py [--max-books 50000] [--output data/books_scraped.csv]

Dependencies (add to requirements.txt):
    requests>=2.31.0
    beautifulsoup4>=4.12.0
    lxml>=4.9.0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.goodreads.com"

# Seed pages: genre shelves and curated lists that expose many book URLs.
# Each page typically contains 20-100 book links.
SEED_SOURCES = [
    # Listopia – each list has up to 100 books per page and ~100 pages
    "/list/show/1.Best_Books_Ever",
    "/list/show/6.Best_Books_of_the_20th_Century",
    "/list/show/264.Books_That_Everyone_Should_Read_At_Least_Once",
    "/list/show/18296.Classic_Novels",
    "/list/show/7457.Goodreads_Choice_Award_Best_Fiction",
    "/list/show/11095.Best_Science_Fiction",
    "/list/show/135.Best_Fantasy_Books",
    "/list/show/2681.Best_Mystery_Thriller",
    "/list/show/9.Best_Horror_Books",
    "/list/show/171.Best_Historical_Fiction",
    "/list/show/4893.Best_Romance_Novels",
    "/list/show/196.Best_Young_Adult_Books",
    "/list/show/87.Best_Children_s_Books",
    "/list/show/53321.Best_Nonfiction_Books",
    "/list/show/120.Best_Biography_Autobiography",
    "/list/show/88.Best_Self_Help",
    "/list/show/102.Best_Books_About_Science",
    "/list/show/7111.Best_Philosophy_Books",
    "/list/show/11709.Best_Graphic_Novels_Comics",
    "/list/show/17245.Best_Short_Story_Collections",
    # Genre shelves (20 books per page, many pages)
    "/shelf/show/fiction",
    "/shelf/show/science-fiction",
    "/shelf/show/fantasy",
    "/shelf/show/mystery",
    "/shelf/show/thriller",
    "/shelf/show/romance",
    "/shelf/show/horror",
    "/shelf/show/historical-fiction",
    "/shelf/show/young-adult",
    "/shelf/show/non-fiction",
    "/shelf/show/biography",
    "/shelf/show/self-help",
    "/shelf/show/classics",
    "/shelf/show/literary-fiction",
    "/shelf/show/graphic-novels",
    "/shelf/show/childrens",
    "/shelf/show/adventure",
    "/shelf/show/dystopia",
    "/shelf/show/crime",
    "/shelf/show/poetry",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Delays (seconds) – randomised to avoid triggering rate limits
MIN_DELAY = 0.5
MAX_DELAY = 1.5
# Number of concurrent scrapers
WORKERS = 8
# Extra back-off on 429 / 503
RATE_LIMIT_BACKOFF = 60  # seconds

# How many consecutive errors before skipping a URL
MAX_RETRIES = 3

CSV_FIELDNAMES = ["book_id", "title", "author", "genres", "description", "url"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    # Prime cookies by hitting the homepage
    try:
        session.get(BASE_URL, timeout=15)
        time.sleep(random.uniform(1, 2))
    except requests.RequestException:
        pass
    return session


def fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES):
    """Fetch a URL, handling rate-limits and transient errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                wait = RATE_LIMIT_BACKOFF * attempt
                log.warning("Rate limited (%s). Sleeping %ds…", resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            log.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt)
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s (attempt %d)", url, exc, attempt)
        time.sleep(random.uniform(MIN_DELAY * attempt, MAX_DELAY * attempt))
    return None


# ---------------------------------------------------------------------------
# Book URL discovery
# ---------------------------------------------------------------------------


def _book_url_from_href(href: str) -> str | None:
    """Return a canonical book URL or None if the href is not a book page."""
    if not href:
        return None
    # Match /book/show/<id> optionally followed by a slug
    if re.search(r"/book/show/\d+", href):
        path = re.match(r"(/book/show/\d+[^?#]*)", href)
        if path:
            return urljoin(BASE_URL, path.group(1))
    return None


def collect_book_urls_from_page(html: str) -> list[str]:
    """Extract all book URLs from a list/shelf/search HTML page."""
    soup = BeautifulSoup(html, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        book_url = _book_url_from_href(a["href"])
        if book_url:
            urls.add(book_url)
    return list(urls)


def iter_seed_pages(
    session: requests.Session,
    seed_path: str,
    max_pages: int = 100,
):
    """Yield (page_number, [book_urls]) for each page of a seed source."""
    for page in range(1, max_pages + 1):
        url = urljoin(BASE_URL, f"{seed_path}?page={page}")
        log.info("Seed %s p%d", seed_path.split("/")[-1], page)
        resp = fetch(session, url)
        if resp is None:
            break
        page_urls = collect_book_urls_from_page(resp.text)
        log.info("  Found %d book URLs on this page", len(page_urls))
        if not page_urls:
            break
        yield page, page_urls
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# ---------------------------------------------------------------------------
# Book data extraction
# ---------------------------------------------------------------------------


def _extract_json_ld(soup: BeautifulSoup) -> dict:
    """Try to parse Book structured data from JSON-LD scripts."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Book":
                        return item
            elif data.get("@type") == "Book":
                return data
        except (json.JSONDecodeError, AttributeError):
            continue
    return {}


def _clean(text: str | None) -> str:
    """Strip whitespace and normalise newlines."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _book_id_from_url(url: str) -> str:
    """Extract the numeric Goodreads book ID from a book URL."""
    m = re.search(r"/book/show/(\d+)", url)
    return m.group(1) if m else ""


def parse_book_page(html: str, url: str = "") -> dict | None:
    """
    Parse a Goodreads book page and return a dict with keys:
    title, author, genres, description.

    Returns None if essential fields (title) cannot be extracted.
    Handles both the pre-2022 and post-2022 Goodreads layouts.
    """
    soup = BeautifulSoup(html, "lxml")

    # -- JSON-LD (most reliable when present) --------------------------------
    jld = _extract_json_ld(soup)

    title = _clean(jld.get("name"))
    author_raw = jld.get("author") or jld.get("creator") or {}
    if isinstance(author_raw, list):
        author = ", ".join(_clean(a.get("name", "")) for a in author_raw if a.get("name"))
    elif isinstance(author_raw, dict):
        author = _clean(author_raw.get("name", ""))
    else:
        author = _clean(str(author_raw))

    genres_raw = jld.get("genre", [])
    if isinstance(genres_raw, str):
        genres_raw = [genres_raw]
    genres = ", ".join(_clean(g) for g in genres_raw if g)

    description = _clean(jld.get("description", ""))

    # -- HTML fallback for missing fields ------------------------------------

    # Title
    if not title:
        # New UI
        tag = soup.select_one('[data-testid="bookTitle"]') or soup.select_one("h1.H1Title")
        if not tag:
            # Old UI
            tag = soup.find(id="bookTitle")
        title = _clean(tag.get_text() if tag else "")

    # Author
    if not author:
        # New UI
        tag = soup.select_one('[data-testid="name"]')
        if not tag:
            # Old UI
            tag = soup.select_one('.authorName span[itemprop="name"]')
        author = _clean(tag.get_text() if tag else "")

    # Genres
    if not genres:
        # New UI
        genre_tags = soup.select('[data-testid="genresList"] .BookPageMetadataSection__genre a')
        if not genre_tags:
            # Old UI
            genre_tags = soup.select(".left .elementList .bookPageGenreLink")
        genres = ", ".join(_clean(t.get_text()) for t in genre_tags)

    # Description
    if not description:
        # New UI
        desc_tag = soup.select_one('[data-testid="description"] .Formatted')
        if not desc_tag:
            # Old UI – the full text is in the last <span> inside #description
            desc_container = soup.find(id="description")
            if desc_container:
                spans = desc_container.find_all("span")
                desc_tag = spans[-1] if spans else desc_container
        description = _clean(desc_tag.get_text() if desc_tag else "")

    if not title:
        return None

    return {
        "book_id": _book_id_from_url(url),
        "title": title,
        "author": author,
        "genres": genres,
        "description": description,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Return the set of URLs already scraped (from the checkpoint file)."""
    if not checkpoint_path.exists():
        return set()
    with checkpoint_path.open() as fh:
        return {line.strip() for line in fh if line.strip()}


def save_checkpoint(checkpoint_path: Path, url: str) -> None:
    with checkpoint_path.open("a") as fh:
        fh.write(url + "\n")



# ---------------------------------------------------------------------------
# Main scraping loop (concurrent)
# ---------------------------------------------------------------------------


class ScrapingState:
    """Thread-safe shared state for concurrent scraping."""

    def __init__(self, already_scraped: set[str], checkpoint_path: Path,
                 writer: csv.DictWriter, csv_fh, max_books: int):
        self.lock = threading.Lock()
        self.already_scraped = already_scraped
        self.checkpoint_path = checkpoint_path
        self.writer = writer
        self.csv_fh = csv_fh
        self.max_books = max_books
        self.scraped_count = len(already_scraped)

    def claim(self, url: str) -> bool:
        """Try to claim a URL for scraping. Returns True if claimed."""
        with self.lock:
            if url in self.already_scraped or self.scraped_count >= self.max_books:
                return False
            self.already_scraped.add(url)
            return True

    def save(self, url: str, book: dict | None) -> None:
        """Save a scraped book result (thread-safe)."""
        with self.lock:
            save_checkpoint(self.checkpoint_path, url)
            if book and self.scraped_count < self.max_books:
                self.writer.writerow(book)
                self.csv_fh.flush()
                self.scraped_count += 1
                log.info("[%d/%d] ✓ %s — %s",
                         self.scraped_count, self.max_books,
                         book["title"], book["author"])
            elif not book:
                log.warning("  ✗ Could not parse %s", url)

    @property
    def done(self) -> bool:
        with self.lock:
            return self.scraped_count >= self.max_books


def scrape_book(session: requests.Session, url: str, state: ScrapingState) -> None:
    """Fetch and save one book."""
    if not state.claim(url):
        return

    resp = fetch(session, url)
    book = parse_book_page(resp.text, url) if resp else None
    state.save(url, book)
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def scrape(
    max_books: int = 50_000,
    output_path: str = "data/books_scraped.csv",
    checkpoint_file: str = "data/.scraped_urls.txt",
    workers: int = WORKERS,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    already_scraped: set[str] = load_checkpoint(checkpoint_path)
    log.info("Resuming: %d URLs already scraped.", len(already_scraped))

    write_header = not output.exists() or output.stat().st_size == 0
    csv_fh = output.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDNAMES)
    if write_header:
        writer.writeheader()

    session = make_session()
    state = ScrapingState(already_scraped, checkpoint_path, writer, csv_fh, max_books)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for seed in SEED_SOURCES:
            if state.done:
                break
            log.info("=== Seed source: %s ===", seed)
            for _, book_urls in iter_seed_pages(session, seed, max_pages=100):
                futures = []
                for url in book_urls:
                    if state.done:
                        break
                    futures.append(pool.submit(scrape_book, session, url, state))
                # Wait for the batch to finish before fetching next page
                for f in as_completed(futures):
                    f.result()  # propagate exceptions
                if state.done:
                    break

    csv_fh.close()
    log.info("Done. %d books saved to %s.", state.scraped_count, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Scrape Goodreads books.")
    parser.add_argument(
        "--max-books",
        type=int,
        default=50_000,
        metavar="N",
        help="Maximum number of books to scrape (default: 50000).",
    )
    parser.add_argument(
        "--output",
        default="data/books_scraped.csv",
        metavar="PATH",
        help="Output CSV path (default: data/books_scraped.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/.scraped_urls.txt",
        metavar="PATH",
        help="Checkpoint file for resume support (default: data/.scraped_urls.txt).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        metavar="N",
        help=f"Number of concurrent scrapers (default: {WORKERS}).",
    )
    args = parser.parse_args()
    scrape(max_books=args.max_books, output_path=args.output,
           checkpoint_file=args.checkpoint, workers=args.workers)


if __name__ == "__main__":
    main()
