from __future__ import annotations

import httpx
from unittest.mock import patch
import sys
import types

# -----------------------------------------------------------------------------
# Provide dummy modules so tools.web can import without optional deps
# (We still patch specifics per-test where needed.)
# -----------------------------------------------------------------------------
sys.modules.setdefault(
    "duckduckgo_search",
    types.SimpleNamespace(DDGS=None),
)
sys.modules.setdefault(
    "trafilatura",
    types.SimpleNamespace(
        load_html=lambda *a, **k: "",
        extract=lambda *a, **k: "",
        utils=types.SimpleNamespace(clean_text=lambda x: x),
    ),
)
sys.modules.setdefault(
    "readability",
    types.SimpleNamespace(Document=lambda *a, **k: types.SimpleNamespace(short_title=lambda: "", summary=lambda: "")),
)
sys.modules.setdefault(
    "youtube_transcript_api",
    types.SimpleNamespace(YouTubeTranscriptApi=types.SimpleNamespace(get_transcript=lambda *a, **k: [])),
)
sys.modules.setdefault(
    "pypdf",
    types.SimpleNamespace(PdfReader=lambda *a, **k: types.SimpleNamespace(pages=[])),
)

from tools.web import fetch_url, web_search, extract_text


def test_fetch_url_fallback_on_error():
    """If network calls keep failing, fetch_url should return a benign error payload."""
    req = httpx.Request("GET", "https://example.com")

    def boom(*args, **kwargs):
        raise httpx.RequestError("boom", request=req)

    with patch("httpx.Client.get", side_effect=boom):
        result = fetch_url("https://example.com")

    assert result["status"] is None
    assert result["text"] == ""
    assert "error" in result


def test_web_search_basic():
    """Basic happy-path for web_search with a minimal DDG stub."""
    class DummyDDGS:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
        def text(self, query, max_results=8, safesearch="moderate"):
            return [{"title": "Title", "href": "http://x", "body": "Snippet"}]

    with patch("tools.web.DDGS", DummyDDGS):
        results = web_search("test")

    assert results == [{"title": "Title", "url": "http://x", "snippet": "Snippet"}]


def test_extract_text_pdf():
    """PDF path: concatenates text from all pages."""
    pdf_bytes = b"%PDF-1.4\n..."

    class DummyPage:
        def __init__(self, text):
            self._text = text
        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, buf):
            self.pages = [DummyPage("first"), DummyPage("second")]

    with patch("tools.web.PdfReader", DummyReader):
        text = extract_text("http://example.com/doc.pdf", pdf_bytes)

    assert "first" in text and "second" in text


def test_extract_text_youtube_transcript():
    """YouTube path: uses transcript when available."""
    segments = [{"text": "hello"}, {"text": "world"}]

    with patch("tools.web._youtube_video_id", return_value="abc123"), \
         patch("tools.web.YouTubeTranscriptApi.get_transcript", return_value=segments):
        out = extract_text("https://www.youtube.com/watch?v=abc123", b"")
    assert "hello" in out and "world" in out


def test_extract_text_html_trafilatura_fallback():
    """HTML path: if trafilatura.extract returns content, we use it."""
    with patch("tools.web.trafilatura.load_html", return_value="<html>x</html>"), \
         patch("tools.web.trafilatura.extract", return_value="extracted text"):
        out = extract_text("https://example.com/page.html", b"<html>...</html>")
    assert out == "extracted text"


def test_fetch_url_domain_and_extension_filters():
    """Enforce allowed domains and file extensions from module-level sets."""
    # Disallow all but example.com and extensions "", .html
    with patch("tools.web.ALLOWED_DOMAINS", {"example.com"}), \
         patch("tools.web.ALLOWED_EXTENSIONS", {"", ".html"}):
        # Not allowed domain
        res1 = fetch_url("https://notallowed.com/")
        assert res1.get("error") == "domain not allowed"

        # Not allowed extension
        res2 = fetch_url("https://example.com/file.bin")
        assert res2.get("error") == "file type not allowed"

    # When allowing *, ensure a normal HTTP path executes (we'll stub the HTTP call)
    class DummyResp:
        def __init__(self, url="https://example.com", status_code=200, content=b"<html><title>T</title></html>"):
            self._url = url
            self.status_code = status_code
            self.content = content
        @property
        def url(self):
            # mimic httpx.Response.url (URL object has __str__)
            class _U:
                def __str__(self_inner):
                    return "https://example.com/"
            return _U()
        def raise_for_status(self):
            return None

    with patch("tools.web.ALLOWED_DOMAINS", {"*"}), \
         patch("httpx.Client.get", return_value=DummyResp()):
        ok = fetch_url("https://any-domain.test/")
    assert ok["status"] == 200