from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Local imports
from .llm.engine import LLMEngine
from .memory.store import MemoryStore
from .tools.web import web_search, fetch_url
from .identity.manager import AnchorManager  # supports advanced policy & fields

# -----------------------------------------------------------------------------
# Paths & config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
CONFIG_PATH = ROOT / "config.yaml"
DOCS_DIR = ROOT / "docs"
INDEX_FILE = DOCS_DIR / "index.html"

# Preferred memories filename (lowercase), plus permissive fallbacks
MEMORY_FILENAMES = [
    "memory.md",      # preferred
    "Memory.md",
    "Memories.md",
    "memories.md",
]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ember.app")

# -----------------------------------------------------------------------------
# Load config (fail fast with friendly error)
# -----------------------------------------------------------------------------
if not CONFIG_PATH.exists():
    raise RuntimeError(f"config.yaml not found at: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Ember Local Server", version="1.0")

# Serve static UI from /docs (GitHub Pages compatible)
if DOCS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DOCS_DIR)), name="static")
else:
    logger.warning("Static docs dir not found: %s", DOCS_DIR)

# CORS: from config if provided, else allow all (local dev convenience)
cors_origins = CFG.get("server", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Initialize services
# -----------------------------------------------------------------------------
# Memory
mem_cfg = CFG.get("memory", {})
MEM = MemoryStore(mem_cfg.get("data_dir", "data"), mem_cfg.get("embed_model"))

# LLM Engine
try:
    ENGINE = LLMEngine(str(CONFIG_PATH))  # let engine read its own section
except Exception as e:  # pragma: no cover - depends on external model files
    logger.exception("Failed to initialize LLM engine: %s", e)
    ENGINE = None

# Anchors / Identity
id_cfg = CFG.get("identity", {})
ANCH = AnchorManager(
    path=id_cfg.get("anchors_file", "src/identity/anchors.yaml"),
    anchors_per_query=id_cfg.get("anchors_per_query"),
    injection_mode=id_cfg.get("anchor_injection", "every_query"),
    strategy=id_cfg.get("strategy", "round_robin"),
    seed=id_cfg.get("seed"),
)

# -----------------------------------------------------------------------------
# Memories.md ingestion -> MemoryStore
# -----------------------------------------------------------------------------
def _resolve_memories_path() -> Optional[Path]:
    """Find docs/memory.md (preferred) or common fallbacks in docs/."""
    for name in MEMORY_FILENAMES:
        p = (DOCS_DIR / name).resolve()
        if p.exists() and p.is_file():
            return p
    return None

def _chunk_markdown_text(md: str, max_chars: int = 1200) -> list[str]:
    """
    Chunk Markdown by headings/blank lines, strip code fences and inline links,
    and shard very long sections to ~max_chars.
    """
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # remove fenced code blocks
    md = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    # split by top-level headings or blank lines
    parts = re.split(r"\n(?=# )|\n{2,}", md)

    chunks: list[str] = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        # keep link text + URL (label — url)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 — \2", t)
        # collapse horizontal whitespace
        t = re.sub(r"[ \t]+", " ", t)
        # hard cap into shards
        while len(t) > max_chars:
            cut = t.rfind(". ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            chunks.append(t[:cut].strip())
            t = t[cut:].strip()
        if t:
            chunks.append(t)
    return chunks

def _maybe_index_memories_to_store(mem_path: Path, max_chars: int = 1200) -> dict:
    """
    Idempotent indexing of docs/memory.md into MemoryStore.
    We keep a small stamp file in data/ to avoid re-embedding if unchanged.
    """
    data_dir = Path(mem_cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = data_dir / ".memories_index.json"

    src_mtime = int(mem_path.stat().st_mtime)
    stamp = {}
    if stamp_file.exists():
        try:
            stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
        except Exception:
            stamp = {}

    if stamp.get("src") == str(mem_path) and stamp.get("mtime") == src_mtime:
        return {"indexed": stamp.get("count", 0), "skipped": True}

    md = mem_path.read_text(encoding="utf-8")
    chunks = _chunk_markdown_text(md, max_chars=max_chars)

    # Store as role='memory' so it can be retrieved like conversation memory
    for c in chunks:
        MEM.add("memory", c)

    new_stamp = {
        "src": str(mem_path),
        "mtime": src_mtime,
        "count": len(chunks),
        "ts": int(time.time()),
    }
    stamp_file.write_text(json.dumps(new_stamp, indent=2), encoding="utf-8")
    return {"indexed": len(chunks), "skipped": False}

# Resolve and index memories at startup
MEM_FILE = _resolve_memories_path()
if MEM_FILE:
    try:
        res = _maybe_index_memories_to_store(MEM_FILE, max_chars=int(CFG.get("identity", {}).get("memories_chunk_chars", 1200)))
        logger.info("Memories indexing: %s (%s)", MEM_FILE.name, res)
    except Exception as e:
        logger.exception("Failed to index %s: %s", MEM_FILE, e)
else:
    logger.info("No docs/memory.md (or fallback) found; skipping memories ingestion.")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str
    use_web: bool = Field(default=False)
    web_query: Optional[str] = None
    urls: Optional[List[str]] = None
    anchor_tags: Optional[List[str]] = Field(
        default=None,
        description="If provided, only anchors containing ALL of these tags are considered for rotation.",
    )

class AnchorIn(BaseModel):
    anchor: str

class AnchorUpdateIn(BaseModel):
    old_text: str
    new_text: Optional[str] = None
    tags: Optional[List[str]] = None
    weight: Optional[float] = None
    always_on: Optional[bool] = None
    priority: Optional[int] = None
    disabled: Optional[bool] = None

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/")
async def root():
    # Serve the UI index if present; otherwise a minimal message.
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return JSONResponse({"ok": True, "msg": "Ember API is running. No UI found at /docs."})

@app.post("/chat")
async def chat(inp: ChatIn):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail="LLM model is unavailable.")

    query = (inp.message or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # --- Web tools section (optional) ---
    web_lines: List[str] = []
    if inp.use_web:
        q = (inp.web_query or query).strip()
        try:
            hits = web_search(q, max_results=6)
        except Exception as e:
            logger.exception("web_search failed: %s", e)
            hits = []
        if hits:
            web_lines.append("Web search results:")
            for i, h in enumerate(hits, 1):
                title = h.get("title") or "(untitled)"
                url = h.get("url") or ""
                snippet = h.get("snippet") or ""
                web_lines.append(f"[{i}] {title} — {url}")
                if snippet:
                    web_lines.append(f"    {snippet}")
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
                title = page.get("title") or page.get("url") or u
                preview = (page.get("text") or "")[:5000]
                web_lines.append(f"- {title} — {page.get('url', u)}")
                if preview:
                    web_lines.append(preview)
            except Exception as e:  # pragma: no cover - unexpected
                logger.exception("fetch_url failed for %s: %s", u, e)
                web_lines.append(f"- Error fetching {u}: {e}")

    # --- Memory retrieval (includes docs/memory.md chunks, once indexed) ---
    k = int(mem_cfg.get("max_context_snippets", 6))
    snippets = MEM.search(query, k=k) if k > 0 else []

    # --- System prompt assembly ---
    sys_lines: List[str] = []
    # Optional top-level system prompt from anchors file (if your AnchorManager supports it)
    if getattr(ANCH, "system_prompt", None):
        sys_lines.append(ANCH.system_prompt.strip())
    else:
        sys_lines.append(f"You are {ANCH.system_name}.")
        sys_lines.append("Be truthful, loyal, concise, and warm.")

    # Identity anchors (always-on + rotating according to policy)
    required = set(inp.anchor_tags) if inp.anchor_tags else None
    try:
        selected_anchors = ANCH.get_for_prompt(required_tags=required)  # enhanced manager
    except TypeError:
        # fallback if your manager doesn't support required_tags param
        selected_anchors = ANCH.get_for_prompt()
    for a in selected_anchors:
        sys_lines.append(a)

    # Web snippets
    if web_lines:
        sys_lines.append("\n".join(web_lines))
        sys_lines.append("Use these sources and cite [numbers] or URLs in your answer.")

    # Memory snippets (conversation + docs/memory.md chunks)
    if snippets:
        sys_lines.append("Relevant memory:")
        for m in snippets:
            # m is expected to be a dict with role/content; adjust if your store uses a different shape
            sys_lines.append(f"- {m.get('role','memory')}: {m.get('content','')}")

    system_prompt = "\n".join(sys_lines)

    # --- LLM call ---
    try:
        reply = ENGINE.chat(system_prompt, query)
    except Exception as e:  # pragma: no cover - depends on model runtime
        logger.exception("LLM chat failed: %s", e)
        raise HTTPException(status_code=500, detail="LLM generation failed.")

    # Persist conversation to memory
    MEM.add("user", query)
    MEM.add("assistant", reply)

    return {"reply": reply}

# ---------------- Anchors API ----------------
@app.get("/anchors")
async def get_anchors(include_disabled: bool = False):
    try:
        return {"anchors": ANCH.list_anchors(include_disabled=include_disabled)}
    except TypeError:
        # fallback if your manager doesn't support include_disabled
        return {"anchors": ANCH.list_anchors()}

@app.post("/anchors")
async def add_anchor(inp: AnchorIn):
    txt = (inp.anchor or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="Anchor cannot be empty.")
    ANCH.add_anchor(txt)
    return {"anchors": getattr(ANCH, "list_anchors", lambda: ANCH.anchors)()}

@app.delete("/anchors")
async def remove_anchor(text: str = Query(..., min_length=1)):
    if hasattr(ANCH, "remove_anchor"):
        removed = ANCH.remove_anchor(text)
        if not removed:
            raise HTTPException(status_code=404, detail="Anchor not found.")
        return {"anchors": getattr(ANCH, "list_anchors", lambda: ANCH.anchors)()}
    raise HTTPException(status_code=400, detail="Remove not supported by AnchorManager.")

@app.patch("/anchors")
async def update_anchor(inp: AnchorUpdateIn):
    if hasattr(ANCH, "update_anchor"):
        ok = ANCH.update_anchor(
            old_text=inp.old_text,
            new_text=inp.new_text,
            tags=inp.tags,
            weight=inp.weight,
            always_on=inp.always_on,
            priority=inp.priority,
            disabled=inp.disabled,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Anchor not found.")
        try:
            return {"anchors": ANCH.list_anchors(include_disabled=True)}
        except TypeError:
            return {"anchors": getattr(ANCH, "list_anchors", lambda: ANCH.anchors)()}
    raise HTTPException(status_code=400, detail="Update not supported by AnchorManager.")

@app.post("/anchors/select")
async def select_anchors(
    n: Optional[int] = Query(None, ge=0),
    tag: Optional[List[str]] = Query(None, alias="tags"),
):
    tags = tag if tag else None
    try:
        return {"selected": ANCH.get_for_prompt(n_override=n, required_tags=tags)}
    except TypeError:
        # fallback if your manager doesn't accept parameters
        return {"selected": ANCH.get_for_prompt()}

# ---------------- Memories maintenance ----------------
@app.post("/memories/reindex")
async def reindex_memories():
    """Re-index docs/memory.md into the vector store if it changed."""
    if not MEM_FILE or not MEM_FILE.exists():
        raise HTTPException(status_code=404, detail="No docs/memory.md (or fallback) found.")
    try:
        res = _maybe_index_memories_to_store(MEM_FILE, max_chars=int(CFG.get("identity", {}).get("memories_chunk_chars", 1200)))
        return {"ok": True, **res}
    except Exception as e:
        logger.exception("Reindex failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Web tools passthrough ----------------
@app.get("/search")
async def search_api(q: str = Query(..., min_length=2), k: int = 8):
    try:
        return {"query": q, "results": web_search(q, max_results=k)}
    except Exception as e:
        logger.exception("search_api error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch")
async def fetch_api(url: str = Query(..., min_length=4)):
    try:
        return fetch_url(url)
    except Exception as e:
        logger.exception("fetch_api error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))