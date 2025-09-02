import httpx, re
from duckduckgo_search import DDGS
import trafilatura
from readability import Document

UA = "NeuralLocalBrowser/1.0"
TIMEOUT = 20.0
MAX_BYTES = 2_500_000
MAX_TEXT = 120_000

def web_search(query: str, max_results: int = 8):
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

def fetch_url(url: str) -> dict:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    headers = {"User-Agent": UA, "Accept": "*/*"}
    with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        content = r.content[:MAX_BYTES]
        text = extract_text(url, content)
        return {
            "url": str(r.url),
            "status": r.status_code,
            "title": extract_title(content) or "",
            "text": text[:MAX_TEXT],
        }

def extract_title(content: bytes) -> str | None:
    try:
        doc = Document(content)
        return doc.short_title()
    except Exception:
        return None

def extract_text(url: str, content: bytes) -> str:
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
