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

from tools.web import fetch_url


def test_fetch_url_fallback_on_error():
    req = httpx.Request("GET", "https://example.com")

    def boom(*args, **kwargs):
        raise httpx.RequestError("boom", request=req)

    with patch("httpx.Client.get", side_effect=boom):
        result = fetch_url("https://example.com")

    assert result["status"] is None
    assert result["text"] == ""
