from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
import logging
from .llm.engine import LLMEngine
from .memory.store import MemoryStore
from .tools.web import web_search, fetch_url
from .identity.manager import AnchorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ember Local Server")
app.mount("/assets", StaticFiles(directory="index"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MEM = MemoryStore(CFG["memory"]["data_dir"], CFG["memory"]["embed_model"])
try:
    ENGINE = LLMEngine("config.yaml")
except Exception as e:  # pragma: no cover - depends on external model files
    logger.exception("Failed to initialize LLM engine: %s", e)
    ENGINE = None

ANCH = AnchorManager(
    CFG["identity"]["anchors_file"],
    anchors_per_query=CFG["identity"].get("anchors_per_query"),
    injection_mode=CFG["identity"].get("anchor_injection", "every_query"),
)


class ChatIn(BaseModel):
    message: str
    use_web: bool = Field(default=False)
    web_query: str | None = None
    urls: list[str] | None = None


class AnchorIn(BaseModel):
    anchor: str


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat")
async def chat(inp: ChatIn):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail="LLM model is unavailable.")
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
        else:
            web_lines.append("Web search failed or returned no results.")
    if inp.urls:
        web_lines.append("Fetched pages:")
        for u in inp.urls:
            try:
                page = fetch_url(u)
                if page.get("error"):
                    web_lines.append(f"- Error fetching {u}: {page['error']}")
                    continue
                title = page.get("title") or page["url"]
                preview = (page.get("text") or "")[:5000]
                web_lines.append(f"- {title} — {page['url']}")
                if preview:
                    web_lines.append(preview)
            except Exception as e:  # pragma: no cover - unexpected
                web_lines.append(f"- Error fetching {u}: {e}")

    k = int(CFG["memory"]["max_context_snippets"])
    snippets = MEM.search(query, k=k)

    sys_lines = [
        f"You are {ANCH.system_name}.",
        "Stay truthful, loyal, concise, and warm.",
    ]
    for a in ANCH.get_for_prompt():
        sys_lines.append(a)
    if web_lines:
        sys_lines.append("\n".join(web_lines))
        sys_lines.append("Use these sources and cite [numbers] or URLs in your answer.")
    if snippets:
        sys_lines.append("Relevant memory:")
        for m in snippets:
            sys_lines.append(f"- {m['role']}: {m['content']}")
    system_prompt = "\n".join(sys_lines)

    try:
        reply = ENGINE.chat(system_prompt, query)
    except Exception as e:  # pragma: no cover - depends on model runtime
        logger.exception("LLM chat failed: %s", e)
        raise HTTPException(status_code=500, detail="LLM generation failed.")

    MEM.add("user", query)
    MEM.add("assistant", reply)

    return {"reply": reply}


@app.get("/anchors")
async def get_anchors():
    return {"anchors": ANCH.list_anchors()}


@app.post("/anchors")
async def add_anchor(inp: AnchorIn):
    ANCH.add_anchor(inp.anchor)
    return {"anchors": ANCH.list_anchors()}


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
