from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from fnmatch import fnmatch

import httpx
import yaml
from duckduckgo_search import DDGS
from pypdf import PdfReader
from readability import Document
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
UA = "NeuralLocalBrowser/1.1 (+https://local)"
# Tighter connect & total timeouts; large pages still get some breathing room
TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
MAX_BYTES = 2_500_000        # ~2.5MB download cap
MAX_TEXT = 120_000           # final text cap
MAX_RETRIES = 3
BASE_DELAY = 0.75            # initial backoff delay
SEARCH_CACHE = 64
FETCH_CACHE = 64

# Load `web` config if available
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
        _web_cfg = _cfg.get("web", {}) or {}
except Exception:  # pragma: no cover - config may be missing
    _web_cfg = {}

# Allowed domains/extensions (supports wildcards like "*.arxiv.org")
_ALLOWED_DOMAINS = list(_web_cfg.get("allowed_domains") or [])
_ALLOWED_EXTS = set(_web_cfg.get("allowed_file_extensions") or [])
# Always allow URLs without an explicit extension
_ALLOWED_EXTS.add("")
# Policy switches
_MAX_FILE_MB = int(_web_cfg.get("max_file_size_mb", 25))
_FETCH_TIMEOUT = int(_web_cfg.get("fetch_timeout", 15))
_SEARCH_PROVIDER = str(_web_cfg.get("search_provider", "duckduckgo"))
_RESPECT_ALLOWED: bool = bool(_ALLOWED_DOMAINS)  # if list empty -> allow none
# Backwards compat with earlier constants
if _FETCH_TIMEOUT and _FETCH_TIMEOUT != 15:
    TIMEOUT.read = float(_FETCH_TIMEOUT)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _domain_allowed(host: str) -> bool:
    """Return True if host is allowed by wildcard-aware allow-list."""
    if not _ALLOWED_DOMAINS:
        # If user configured empty list, block all; if omitted entirely, allow none by default.
        return False
    for pat in _ALLOWED_DOMAINS:
        # support "*" (allow all) or wildcard like "*.example.com"
        if pat == "*" or fnmatch(host, pat):
            return True
    return False


def _ext_allowed(ext: str) -> bool:
    return (ext or "") in _ALLOWED_EXTS


def _guess_ext_from_content_type(ct: str | None) -> str:
    if not ct:
        return ""
    ct = ct.split(";")[0].strip().lower()
    if ct in ("application/pdf",):
        return ".pdf"
    if ct in ("text/markdown", "text/x-markdown"):
        return ".md"
    if ct.startswith("text/"):
        return ".txt"
    if ct in ("application/json",):
        return ".json"
    if ct in ("text/html", "application/xhtml+xml"):
        return ".html"
    return ""


def _cap_bytes(b: bytes, limit: int = MAX_BYTES) -> bytes:
    return b[: max(0, int(limit))]


def _cap_text(s: str, limit: int = MAX_TEXT) -> str:
    if len(s) <= limit:
        return s
    # Try to cut cleanly at a sentence boundary
    cut = s[:limit]
    m = re.search(r"[.!?…]\s+\Z", cut)
    return cut if m else cut.rsplit("\n", 1)[0]


def _youtube_video_id(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if "youtube.com" in host:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        if host == "youtu.be":
            return parsed.path.lstrip("/")
    except Exception:
        return None
    return None


def _extract_title_html(content: bytes) -> Optional[str]:
    try:
        doc = Document(content)  # robust title inference
        title = (doc.short_title() or "").strip()
        return title or None
    except Exception as e:  # pragma: no cover
        logger.debug("readability title failed: %s", e)
        return None


def _summarize_text(text: str, max_chars: int = 1200) -> str:
    """Quick textual summary by taking leading sentences with a soft cap."""
    if len(text) <= max_chars:
        return text
    # sentence-ish split
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out = []
    total = 0
    for p in parts:
        if not p:
            continue
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p) + 1
    return " ".join(out)


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------
@lru_cache(maxsize=SEARCH_CACHE)
def web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """DuckDuckGo text search with basic retry + cache."""
    if not query or _SEARCH_PROVIDER.lower() != "duckduckgo":
        return []
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with DDGS() as ddgs:
                hits = ddgs.text(query, max_results=max_results, safesearch="moderate")
                out: List[Dict[str, str]] = []
                for h in hits or []:
                    out.append({
                        "title": (h.get("title") or "")[:200],
                        "url": h.get("href") or "",
                        "snippet": ((h.get("body") or "")[:500]),
                    })
                return out
        except Exception as e:  # pragma: no cover
            last_err = e
            delay = BASE_DELAY * attempt
            logger.warning("web_search retry %d for %r: %s (sleep %.2fs)", attempt, query, e, delay)
            time.sleep(delay)
    logger.error("web_search failed for %r: %s", query, last_err)
    return []


# -----------------------------------------------------------------------------
# Fetch
# -----------------------------------------------------------------------------
@lru_cache(maxsize=FETCH_CACHE)
def fetch_url(url: str) -> Dict[str, str]:
    """
    Fetch a URL, enforce allow-lists, extract readable text, and return:
    {
      "url": final_url,
      "status": int|None,
      "title": str,
      "text": str,
      "error": str (optional)
    }
    """
    # Normalize scheme
    if not re.match(r"^https?://", url, flags=re.I):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    ext = os.path.splitext(parsed.path)[1].lower()

    if _RESPECT_ALLOWED and not _domain_allowed(host):
        return {"url": url, "status": None, "title": "", "text": "", "error": "domain not allowed"}

    if not _ext_allowed(ext):
        # we might still allow if content-type later maps to an allowed ext; we’ll check post-fetch
        pass

    headers = {
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.7",
        "Cache-Control": "no-cache",
    }

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                # Size guard before reading all bytes (httpx streams internally, but we still cap)
                content = _cap_bytes(r.content, limit=min(MAX_BYTES, _MAX_FILE_MB * 1024 * 1024))
                r.raise_for_status()

                # Decide extension priority: URL -> Content-Type
                ctype = r.headers.get("content-type")
                guessed_ext = _guess_ext_from_content_type(ctype)
                eff_ext = ext or guessed_ext

                # Enforce extension policy now that we know content-type
                if not _ext_allowed(eff_ext):
                    return {
                        "url": str(r.url),
                        "status": r.status_code,
                        "title": "",
                        "text": "",
                        "error": f"file type not allowed ({eff_ext or 'unknown'})",
                    }

                # Extract text/title
                text = _extract_text(str(r.url), content, ctype=ctype)
                title = _extract_title(content, ctype=ctype) or ""

                return {
                    "url": str(r.url),
                    "status": r.status_code,
                    "title": title[:200],
                    "text": _cap_text(text, MAX_TEXT),
                }
        except Exception as e:
            last_exc = e
            delay = BASE_DELAY * attempt
            logger.warning("fetch_url retry %d for %s: %s (sleep %.2fs)", attempt, url, e, delay)
            time.sleep(delay)

    status = None
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response is not None:
        status = last_exc.response.status_code
    logger.error("fetch_url failed for %s: %s", url, last_exc)
    return {"url": url, "status": status, "title": "", "text": "", "error": str(last_exc) if last_exc else ""}


def fetch_urls(urls: List[str]) -> List[Dict[str, str]]:
    """Convenience wrapper: fetch multiple URLs safely."""
    out: List[Dict[str, str]] = []
    for u in urls:
        try:
            out.append(fetch_url(u))
        except Exception as e:  # cached fetch can still raise on cache-miss edge cases
            logger.exception("fetch_urls error for %s: %s", u, e)
            out.append({"url": u, "status": None, "title": "", "text": "", "error": str(e)})
    return out


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def _extract_title(content: bytes, ctype: Optional[str]) -> Optional[str]:
    # PDF title extraction is typically unreliable; use readability for HTML
    if ctype and "html" in ctype.lower():
        return _extract_title_html(content)
    # Fallback: try readability regardless (robust to bad markup)
    t = _extract_title_html(content)
    return t


def _extract_text(url: str, content: bytes, ctype: Optional[str]) -> str:
    # YouTube transcript (highest priority for watch pages)
    vid = _youtube_video_id(url)
    if vid:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            return " ".join([t.get("text", "") for t in transcript])
        except Exception as e:  # pragma: no cover
            logger.debug("YouTube transcript fetch failed for %s: %s", url, e)

    # PDF
    if (ctype and "pdf" in ctype.lower()) or content[:4] == b"%PDF":
        try:
            reader = PdfReader(BytesIO(content))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t.strip())
            return "\n".join(parts)
        except Exception as e:  # pragma: no cover
            logger.warning("PDF extraction failed for %s: %s", url, e)
            return ""

    # JSON (pretty-print short JSON; otherwise summarize stringified JSON)
    if ctype and "json" in ctype.lower():
        try:
            data = json.loads(content.decode("utf-8", errors="ignore"))
            txt = json.dumps(data, ensure_ascii=False, indent=2)
            return _summarize_text(txt, max_chars=MAX_TEXT)
        except Exception as e:
            logger.debug("JSON decode failed for %s: %s", url, e)

    # Plain text / Markdown
    if ctype and (ctype.startswith("text/") or "markdown" in ctype.lower()):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")

    # HTML (trafilatura first; readability fallback)
    try:
        downloaded = trafilatura.load_html(content, url=url)
        txt = trafilatura.extract(
            downloaded,
            include_formatting=False,
            include_images=False,
            include_links=False,
        ) or ""
        if txt.strip():
            return txt
    except Exception as e:  # pragma: no cover
        logger.debug("trafilatura extract failed for %s: %s", url, e)

    try:
        doc = Document(content)
        html = doc.summary()  # HTML subset
        cleaned = trafilatura.utils.clean_text(html) or ""
        return cleaned
    except Exception as e:  # pragma: no cover
        logger.debug("readability extract failed for %s: %s", url, e)
        return ""


# -----------------------------------------------------------------------------
# Optional: quick text summarizer users can call via tools layer
# -----------------------------------------------------------------------------
def summarize_text(text: str, max_chars: int = 1200) -> str:
    return _summarize_text(text, max_chars=max_chars)