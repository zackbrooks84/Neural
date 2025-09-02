from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml
from .llm.engine import LLMEngine
from .memory.store import MemoryStore
from .tools.web import web_search, fetch_url

app = FastAPI(title="Ember Local Server")
app.mount("/assets", StaticFiles(directory="index"), name="assets")

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MEM = MemoryStore(CFG["memory"]["data_dir"], CFG["memory"]["embed_model"])
ENGINE = LLMEngine("config.yaml")

with open(CFG["identity"]["anchors_file"], "r", encoding="utf-8") as f:
    ANCH = yaml.safe_load(f)


class ChatIn(BaseModel):
    message: str
    use_web: bool = Field(default=False)
    web_query: str | None = None
    urls: list[str] | None = None


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat")
async def chat(inp: ChatIn):
    query = inp.message.strip()

    web_lines = []
    if inp.use_web:
        q = (inp.web_query or inp.message).strip()
        hits = web_search(q, max_results=6)
        if hits:
            web_lines.append("Web search results:")
            for i, h in enumerate(hits, 1):
                web_lines.append(f"[{i}] {h['title']} — {h['url']}")
                if h["snippet"]:
                    web_lines.append(f"    {h['snippet']}")
    if inp.urls:
        web_lines.append("Fetched pages:")
        for u in inp.urls:
            try:
                page = fetch_url(u)
                title = page.get("title") or page["url"]
                preview = (page.get("text") or "")[:5000]
                web_lines.append(f"- {title} — {page['url']}")
                if preview:
                    web_lines.append(preview)
            except Exception as e:
                web_lines.append(f"- Error fetching {u}: {e}")

    k = int(CFG["memory"]["max_context_snippets"])
    snippets = MEM.search(query, k=k)

    sys_lines = [
        f"You are {ANCH.get('system_name','Ember')}.",
        "Stay truthful, loyal, concise, and warm.",
    ]
    for a in ANCH.get("anchors", []):
        sys_lines.append(a)
    if web_lines:
        sys_lines.append("\n".join(web_lines))
        sys_lines.append("Use these sources and cite [numbers] or URLs in your answer.")
    if snippets:
        sys_lines.append("Relevant memory:")
        for m in snippets:
            sys_lines.append(f"- {m['role']}: {m['content']}")
    system_prompt = "\n".join(sys_lines)

    reply = ENGINE.chat(system_prompt, query)

    MEM.add("user", query)
    MEM.add("assistant", reply)

    return {"reply": reply}


@app.get("/")
async def root():
    return FileResponse("index/index.html")


from fastapi import Query as FQuery


@app.get("/search")
async def search_api(q: str = FQuery(..., min_length=2), k: int = 8):
    return {"query": q, "results": web_search(q, max_results=k)}


@app.get("/fetch")
async def fetch_api(url: str = FQuery(..., min_length=4)):
    return fetch_url(url)
