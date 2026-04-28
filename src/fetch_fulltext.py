"""Fetch full paper text from arXiv HTML, open-access journal pages, and PDFs.

Priority order for each paper:
1. arXiv HTML        — cleanest source, no extra deps
2. bioRxiv HTML      — for bioRxiv preprints
3. Unpaywall API     — finds open-access versions of paywalled DOIs (free, no key)
4. Generic URL       — open-access HTML (MDPI, PLoS, BioMed Central, ResearchSquare…)
                       or direct PDF links via pdfplumber
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

import requests

from src.schemas import PaperRecord, UNKNOWN

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
})

# Sections we want to surface early in the text budget
_PRIORITY_RE = re.compile(
    r"(?:method|experiment|evaluation|dataset|result|training|implementation"
    r"|preprocessing|signal\s*processing|data\s*processing|split|validation"
    r"|participants|subjects|patients)",
    re.I,
)


# ---------------------------------------------------------------------------
# HTML text extractor (no external deps)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    SKIP_TAGS = frozenset({"script", "style", "head", "nav", "footer", "noscript",
                           "aside", "form", "button", "iframe", "svg"})
    BLOCK_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th",
                            "div", "section", "article", "blockquote",
                            "figcaption", "caption"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self._skip += 1
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")
        elif tag == "img" and not self._skip:
            attrs_dict = dict(attrs)
            alt = attrs_dict.get("alt", "").strip()
            if alt and len(alt) > 15:
                self._parts.append(f" [Figure: {alt}] ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            t = data.strip()
            if t:
                self._parts.append(t + " ")

    def result(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return re.sub(r" {2,}", " ", raw).strip()


def _strip_html(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.result()


# ---------------------------------------------------------------------------
# Text windowing — surface methods/results within the token budget
# ---------------------------------------------------------------------------

def _windowed_text(text: str, max_chars: int) -> str:
    """
    Return up to max_chars of text, but ensure Methods/Results content is
    included even if it starts after the first 2000 chars.
    """
    if len(text) <= max_chars:
        return text

    head_chars = min(2500, max_chars // 3)
    chunks = [text[:head_chars]]
    used = head_chars
    spans = [(0, head_chars)]
    patterns = [
        r"preprocessing|pre-processing|signal\s*processing|data\s*processing",
        r"downsampl|resampl|sampling\s+(?:rate|frequency)",
        r"cross[\s-]?validation|validation|fold|leave[\s-]one",
        r"train(?:ing)?\s*/?\s*test|split|held[\s-]?out",
        r"implementation|experiment|evaluation",
        r"method",
        r"result",
    ]

    def overlaps(start: int, end: int) -> bool:
        return any(start < old_end and end > old_start for old_start, old_end in spans)

    for pattern in patterns:
        for m in re.finditer(pattern, text, re.I):
            if m.start() < head_chars:
                continue
            remaining = max_chars - used
            if remaining < 800:
                break
            window = min(3500, remaining)
            start = max(head_chars, m.start() - 700)
            end = min(len(text), start + window)
            if overlaps(start, end):
                continue
            chunks.append(text[start:end])
            spans.append((start, end))
            used += end - start
            break

    if len(chunks) > 1:
        return "\n\n[...]\n\n".join(chunks)

    # Last resort: preserve previous behavior if no targeted sections matched.
    m = _PRIORITY_RE.search(text, pos=head_chars)
    if m:
        section_start = m.start()
        return text[:head_chars] + "\n\n[...]\n\n" + text[section_start: section_start + (max_chars - head_chars)]

    return text[:max_chars]


# ---------------------------------------------------------------------------
# PDF extraction (requires pdfplumber)
# ---------------------------------------------------------------------------

def _pdf_from_bytes(content: bytes, max_chars: int) -> str | None:
    extractors = [_pdf_text_pdfplumber, _pdf_text_pymupdf, _pdf_text_pypdf]
    for extractor in extractors:
        text = extractor(content)
        if text and text.strip():
            return _windowed_text(text, max_chars)
    return None


def _pdf_text_pdfplumber(content: bytes) -> str | None:
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages[:30])
    except Exception:
        return None


def _pdf_text_pymupdf(content: bytes) -> str | None:
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc[:30])
    except Exception:
        return None


def _pdf_text_pypdf(content: bytes) -> str | None:
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            return None
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:30])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source-specific fetchers
# ---------------------------------------------------------------------------

def _arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    return m.group(1) if m else None


def _biorxiv_id(url: str) -> str | None:
    m = re.search(r"biorxiv\.org/content/(10\.\d{4,}/\S+?)(?:v\d+)?(?:\.full|\.pdf)?$", url)
    return m.group(1) if m else None


def _doi_from_url(url: str) -> str | None:
    m = re.search(r"doi\.org/(10\.\d{4,}/\S+)", url)
    return m.group(1) if m else None


def _extract_figure_urls(html: str, arxiv_id: str, max_figures: int = 4) -> list[str]:
    """Extract content figure image URLs from an arXiv HTML page."""
    base = f"https://arxiv.org/html/{arxiv_id}/"
    urls: list[str] = []
    for fig_m in re.finditer(r"<figure\b[^>]*>(.*?)</figure>", html, re.DOTALL | re.I):
        img_m = re.search(r'<img\b[^>]+\bsrc=["\']([^"\']+)["\']', fig_m.group(1), re.I)
        if not img_m:
            continue
        src = img_m.group(1)
        # Skip SVG/GIF and anything that looks like a math/formula image
        if any(src.lower().endswith(ext) for ext in (".svg", ".gif", ".ico")):
            continue
        if any(kw in src.lower() for kw in ("math", "formula", "equat", "x-eq")):
            continue
        if not src.startswith("http"):
            src = base + src.lstrip("./")
        urls.append(src)
        if len(urls) >= max_figures:
            break
    return urls


def _fetch_arxiv_html(arxiv_id: str, max_chars: int) -> str | None:
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        r = _SESSION.get(url, timeout=20)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return None
        raw_html = r.text
        text = _strip_html(raw_html)
        if len(text) <= 300:
            return None
        result = _windowed_text(text, max_chars)
        fig_urls = _extract_figure_urls(raw_html, arxiv_id)
        if fig_urls:
            result += "\nFigureURLs: " + " ".join(fig_urls)
        return result
    except Exception:
        return None


def _fetch_biorxiv_html(doi: str, max_chars: int) -> str | None:
    url = f"https://www.biorxiv.org/content/{doi}.full"
    try:
        r = _SESSION.get(url, timeout=20)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return None
        text = _strip_html(r.text)
        return _windowed_text(text, max_chars) if len(text) > 300 else None
    except Exception:
        return None


def _fetch_unpaywall(doi: str, max_chars: int) -> str | None:
    """
    Query the free Unpaywall API to find an open-access version of a paywalled DOI,
    then fetch its full text. No API key required — just an email address.
    """
    api_url = f"https://api.unpaywall.org/v2/{doi}?email=sota-finder@example.com"
    try:
        r = _SESSION.get(api_url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()

        # Try PDF links first, then landing pages, in order of Unpaywall's ranking
        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                content = _SESSION.get(pdf_url, timeout=25).content
                text = _pdf_from_bytes(content, max_chars)
                if text:
                    return text

            landing = loc.get("url_for_landing_page")
            if landing and not _is_paywalled(landing):
                text = _fetch_html_or_pdf(landing, max_chars)
                if text:
                    return text

    except Exception:
        pass
    return None


# Domains / URL patterns that are paywalled — skip the primary URL immediately,
# but Unpaywall may still find an open-access version for these DOIs.
_SKIP_PATTERNS = (
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "onlinelibrary.wiley.com",
    "sciencedirect.com",
    "doi.org/10.1109/",   # IEEE
    "doi.org/10.1145/",   # ACM
    "doi.org/10.1007/",   # Springer
    "doi.org/10.1016/",   # Elsevier
)


def _is_paywalled(url: str) -> bool:
    return any(p in url for p in _SKIP_PATTERNS)


def _is_pdf_url(url: str) -> bool:
    u = url.lower()
    return (
        u.endswith(".pdf")
        or "latest.pdf" in u
        or ("/download/" in u and ("/pdf" in u or re.search(r"/\d+$", u)))
    )


def _fetch_html_or_pdf(url: str, max_chars: int) -> str | None:
    """Fetch a URL and return text whether the response is HTML or PDF."""
    try:
        r = _SESSION.get(url, timeout=20, allow_redirects=True)
        if r.status_code != 200:
            return None
        ct = r.headers.get("Content-Type", "").lower()
        content = r.content

        if "pdf" in ct or content[:4] == b"%PDF" or _is_pdf_url(url):
            return _pdf_from_bytes(content, max_chars)

        if "html" in ct or not ct:
            if "proceedings.mlr.press" in r.url:
                pdf_m = (
                    re.search(r'href=["\']([^"\']+\.pdf)["\']', r.text, re.I)
                    or re.search(r'pdf\s*=\s*\{([^}]+\.pdf)\}', r.text, re.I)
                    or re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+\.pdf)["\']', r.text, re.I)
                )
                if pdf_m:
                    pdf_url = pdf_m.group(1)
                    if pdf_url.startswith("/"):
                        pdf_url = "https://proceedings.mlr.press" + pdf_url
                    pdf_text = _fetch_html_or_pdf(pdf_url, max_chars)
                    if pdf_text:
                        return pdf_text
            text = _strip_html(r.text)
            if len(text) > 1500:
                return _windowed_text(text, max_chars)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Title-based fallback — searches multiple sites when the primary URL fails
# ---------------------------------------------------------------------------

def _arxiv_id_from_title(title: str) -> str | None:
    """Search arXiv by title, return the best-match arXiv ID."""
    try:
        r = _SESSION.get(
            "https://export.arxiv.org/search/",
            params={"query": title, "searchtype": "ti", "max_results": 3},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        ids = re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", r.text)
        return ids[0] if ids else None
    except Exception:
        return None


def _doi_from_title(title: str) -> str | None:
    """Search CrossRef by title to get a DOI."""
    try:
        r = _SESSION.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": 3,
                    "mailto": "sota-finder@example.com"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        return items[0].get("DOI") if items else None
    except Exception:
        return None


def _s2_oa_url_from_title(title: str) -> str | None:
    """Search Semantic Scholar by title, return an open-access URL."""
    try:
        r = _SESSION.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": f'"{title}"',
                    "fields": "openAccessPdf,externalIds", "limit": 3},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        for paper in r.json().get("data", []):
            pdf_info = paper.get("openAccessPdf") or {}
            if pdf_info.get("url"):
                return pdf_info["url"]
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            if arxiv_id:
                return f"https://arxiv.org/abs/{arxiv_id}"
    except Exception:
        pass
    return None


def _fetch_by_title(title: str, max_chars: int) -> str | None:
    """
    Last-resort: search multiple sites by paper title to find an open-access copy.
    Tries arXiv → CrossRef+Unpaywall → Semantic Scholar, in order.
    """
    # 1. arXiv title search
    aid = _arxiv_id_from_title(title)
    if aid:
        text = _fetch_arxiv_html(aid, max_chars)
        if text:
            return text
        # Fall through to arXiv PDF if HTML not available
        text = _fetch_html_or_pdf(f"https://arxiv.org/pdf/{aid}", max_chars)
        if text:
            return text

    # 2. CrossRef DOI → Unpaywall
    doi = _doi_from_title(title)
    if doi:
        text = _fetch_unpaywall(doi, max_chars)
        if text:
            return text

    # 3. Semantic Scholar OA PDF / arXiv link
    oa_url = _s2_oa_url_from_title(title)
    if oa_url:
        aid2 = _arxiv_id(oa_url)
        if aid2 and aid2 != aid:
            text = _fetch_arxiv_html(aid2, max_chars)
            if text:
                return text
        if not _is_paywalled(oa_url):
            text = _fetch_html_or_pdf(oa_url, max_chars)
            if text:
                return text

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _fetch_one(args: tuple[int, PaperRecord, int]) -> tuple[int, str | None]:
    """Worker: fetch full text for a single paper. Returns (index, text_or_None)."""
    i, paper, max_chars = args
    url = paper.paper_url
    title = paper.title or ""

    # 1. arXiv HTML (from URL)
    if url and url != UNKNOWN:
        aid = _arxiv_id(url)
        if aid:
            text = _fetch_arxiv_html(aid, max_chars)
            if text:
                return i, text

        # 2. bioRxiv HTML
        bid = _biorxiv_id(url)
        if bid:
            text = _fetch_biorxiv_html(bid, max_chars)
            if text:
                return i, text

        # 3. For paywalled primary URLs, try Unpaywall first
        doi = _doi_from_url(url)
        if _is_paywalled(url):
            if doi:
                text = _fetch_unpaywall(doi, max_chars)
                if text:
                    return i, text
        else:
            # 4. Generic HTML / PDF (open-access journals, direct links)
            text = _fetch_html_or_pdf(url, max_chars)
            if text:
                return i, text

            # 5. Unpaywall for any remaining DOI
            if doi:
                text = _fetch_unpaywall(doi, max_chars)
                if text:
                    return i, text

    # 6. Title-based search across arXiv / CrossRef / Semantic Scholar
    if title:
        text = _fetch_by_title(title, max_chars)
        if text:
            return i, text

    return i, None


def pdf_text_from_file(file_bytes: bytes, max_chars: int = 25_000) -> str | None:
    """Extract text from an uploaded PDF (raw bytes). Returns None if pdfplumber
    is not installed or the file cannot be parsed."""
    return _pdf_from_bytes(file_bytes, max_chars)


def enrich_with_fulltext(
    papers: list[PaperRecord],
    max_chars: int = 30000,
    workers: int = 8,
) -> list[PaperRecord]:
    """
    Fetch full text for each paper in parallel and store it in notes under a
    'FullText:' marker. Papers where full text is unavailable are unchanged.
    Returns a new list; originals are not mutated.
    """
    result = list(papers)
    fetched = 0

    args = [(i, p, max_chars) for i, p in enumerate(papers)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, full_text in pool.map(_fetch_one, args):
            if full_text:
                paper = papers[i]
                notes = paper.notes or ""
                # Preserve abstract for display; append full text for extraction
                result[i] = paper.model_copy(
                    update={"notes": notes + f" FullText: {full_text}"}
                )
                fetched += 1

    print(f"  [FullText] Fetched full text for {fetched}/{len(papers)} paper(s)")
    return result
