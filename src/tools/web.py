import httpx, re, time
from duckduckgo_search import DDGS
import trafilatura
from readability import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from io import BytesIO
from pypdf import PdfReader

UA = "NeuralLocalBrowser/1.0"
# More aggressive connect timeout to fail fast on network issues while
# allowing longer for reading large pages.
TIMEOUT = httpx.Timeout(20.0, connect=5.0)
MAX_BYTES = 2_500_000
MAX_TEXT = 120_000
MAX_RETRIES = 3
RETRY_DELAY = 1.0

def web_search(query: str, max_results: int = 8):
    """DuckDuckGo search with simple retry and graceful fallback."""
    for attempt in range(MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                hits = ddgs.text(query, max_results=max_results, safesearch="moderate")
                out = []
                for h in hits or []:
                    out.append({
                        "title": h.get("title", "")[:200],
                        "url": h.get("href", ""),
                        "snippet": (h.get("body", "") or "")[:500],
                    })
                return out
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return []

def fetch_url(url: str) -> dict:
    """Download a URL with retries and basic error handling."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
    headers = {"User-Agent": UA, "Accept": "*/*"}
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                content = r.content[:MAX_BYTES]
                text = extract_text(str(r.url), content)
                return {
                    "url": str(r.url),
                    "status": r.status_code,
                    "title": extract_title(content) or "",
                    "text": text[:MAX_TEXT],
                }
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    # Fallback result when all retries failed
    status = None
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response is not None:
        status = last_exc.response.status_code
    return {"url": url, "status": status, "title": "", "text": ""}

def extract_title(content: bytes) -> str | None:
    try:
        doc = Document(content)
        return doc.short_title()
    except Exception:
        return None

def extract_text(url: str, content: bytes) -> str:
    # YouTube transcript
    vid = _youtube_video_id(url)
    if vid:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            return " ".join([t.get("text", "") for t in transcript])
        except Exception:
            pass
    # PDF extraction
    if url.lower().endswith(".pdf") or content[:4] == b"%PDF":
        try:
            reader = PdfReader(BytesIO(content))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n".join(parts)
        except Exception:
            return ""
    # Fallback HTML extraction
    try:
        downloaded = trafilatura.load_html(content, url=url)
        txt = trafilatura.extract(downloaded, include_formatting=False, include_images=False) or ""
        if txt.strip():
            return txt
    except Exception:
        pass
    try:
        doc = Document(content)
        html = doc.summary()
        return trafilatura.utils.clean_text(html) or ""
    except Exception:
        return ""


def _youtube_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if "youtube.com" in host:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        if host == "youtu.be":
            return parsed.path.lstrip("/")
    except Exception:
        return None
    return None
