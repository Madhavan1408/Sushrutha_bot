import re
import os
import fitz
import logging
import concurrent.futures
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}   # fixed: was "user_agent" (wrong key)
REQUEST_TIMEOUT = 15                       # seconds per URL
MAX_URL_WORKERS = 8                        # parallel URL fetches


# ── Text cleaning ──────────────────────────────────────────────────────────────
_WHITESPACE   = re.compile(r"\s+")
_HYPHEN_SPACE = re.compile(r"-\s+")
_NON_ASCII    = re.compile(r"[^\x20-\x7F]+")   # fixed: was [^\x10-x7F] (broken)
_CONTROL      = re.compile(r"[\x00-\x1F]+")

def clean(text: str) -> str:
    """Remove extra whitespace, soft-hyphens, and non-ASCII / control chars."""
    text = _HYPHEN_SPACE.sub(" ", text)
    text = _NON_ASCII.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


# ── PDF loaders ────────────────────────────────────────────────────────────────
def load_pdf(file_path: str | Path) -> list[Document]:
    """Extract and clean text blocks from a single PDF."""
    documents: list[Document] = []
    with fitz.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            for block in page.get_text("blocks"):
                raw = block[4].strip()
                if raw:
                    documents.append(
                        Document(
                            page_content=clean(raw),
                            metadata={"page": page_num, "source": str(file_path)},
                        )
                    )
    return documents


def load_folder(folder_path: str | Path) -> list[Document]:
    """Load and extract text from every PDF in *folder_path*."""
    documents: list[Document] = []
    pdfs = list(Path(folder_path).glob("*.pdf"))
    log.info("Found %d PDF(s) in '%s'", len(pdfs), folder_path)
    for pdf_path in pdfs:
        log.info("Processing file: %s", pdf_path.name)
        documents.extend(load_pdf(pdf_path))
    return documents


# ── URL loader (with connection-pooling + timeout) ─────────────────────────────
_session = requests.Session()
_session.headers.update(HEADERS)

def url_reader(url: str) -> list[Document]:
    """Scrape, clean, and return the main text content of a web page."""
    try:
        response = _session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Skipping %s — %s", url, exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()

    body = soup.find("main") or soup.body
    if body is None:
        log.warning("No body content found at %s", url)
        return []

    return [
        Document(
            page_content=clean(body.get_text()),
            metadata={"source": url},
        )
    ]


def load_urls(urls: list[str], max_workers: int = MAX_URL_WORKERS) -> list[Document]:
    """Fetch all URLs in parallel and return a flat list of Documents."""
    documents: list[Document] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(url_reader, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                docs = future.result()
                if docs:
                    log.info("✓ %s — %d doc(s)", url, len(docs))
                documents.extend(docs)
            except Exception as exc:
                log.error("Unexpected error for %s: %s", url, exc)
    return documents
