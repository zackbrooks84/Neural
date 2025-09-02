import httpx
from unittest.mock import patch
import sys
import types

# Provide dummy modules so tools.web can import without optional deps
sys.modules.setdefault("duckduckgo_search", types.SimpleNamespace(DDGS=None))
sys.modules.setdefault("trafilatura", types.SimpleNamespace(load_html=lambda *a, **k: "", extract=lambda *a, **k: "", utils=types.SimpleNamespace(clean_text=lambda x: x)))
sys.modules.setdefault("readability", types.SimpleNamespace(Document=lambda *a, **k: types.SimpleNamespace(short_title=lambda: "", summary=lambda: "")))
sys.modules.setdefault("youtube_transcript_api", types.SimpleNamespace(YouTubeTranscriptApi=types.SimpleNamespace(get_transcript=lambda *a, **k: [])))
sys.modules.setdefault("pypdf", types.SimpleNamespace(PdfReader=lambda *a, **k: types.SimpleNamespace(pages=[])))

from tools.web import fetch_url, web_search, extract_text


def test_fetch_url_fallback_on_error():
    req = httpx.Request("GET", "https://example.com")

    def boom(*args, **kwargs):
        raise httpx.RequestError("boom", request=req)

    with patch("httpx.Client.get", side_effect=boom):
        result = fetch_url("https://example.com")

    assert result["status"] is None
    assert result["text"] == ""


def test_web_search_basic():
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
