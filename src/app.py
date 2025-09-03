# src/app.py
from __future__ import annotations

import hashlib
import logging
import re
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
from .identity.manager import AnchorManager  # advanced policy & fields

# -----------------------------------------------------------------------------
# Paths & config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
CONFIG_PATH = ROOT / "config.yaml"
DOCS_DIR = ROOT / "docs"
INDEX_FILE = DOCS_DIR / "index.html"
MEMORIES_MD = DOCS_DIR / "memory.md"  # <- auto-index source
FINGERPRINT_FILE = ROOT / "data" / ".memories_md.sha256"  # idempotent reindex

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
app = FastAPI(title="Ember Local Server", version="1.1")

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
# Memory.md → Vector store (auto index at startup)
# -----------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _chunk_text(text: str, chunk_chars: int) -> List[str]:
    """
    Chunk by characters but prefer to cut on paragraph boundaries,
    then sentence boundaries, then word boundaries.
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_chars:
            buf = f"{buf}\n\n{p}"
        else:
            # try to split the paragraph if it's huge
            if len(p) > chunk_chars:
                # sentence-ish split
                parts = re.split(r"(?<=[.!?])\s+", p)
                tmp = ""
                for s in parts:
                    if not tmp:
                        tmp = s
                    elif len(tmp) + 1 + len(s) <= chunk_chars:
                        tmp = f"{tmp} {s}"
                    else:
                        # push tmp to either buf or chunks
                        if not buf:
                            buf = tmp
                        elif len(buf) + 2 + len(tmp) <= chunk_chars:
                            buf = f"{buf}\n\n{tmp}"
                        else:
                            flush()
                            buf = tmp
                        tmp = s
                # push remainder
                if tmp:
                    if not buf:
                        buf = tmp
                    elif len(buf) + 2 + len(tmp) <= chunk_chars:
                        buf = f"{buf}\n\n{tmp}"
                    else:
                        flush()
                        buf = tmp
            else:
                flush()
                buf = p
        if len(buf) >= chunk_chars:
            flush()

    flush()
    # final pass: ensure no chunk exceeds chunk_chars by hard wrap (rare)
    out: List[str] = []
    for c in chunks:
        if len(c) <= chunk_chars:
            out.append(c)
        else:
            start = 0
            while start < len(c):
                out.append(c[start : start + chunk_chars])
                start += chunk_chars
    return out

def _fingerprint_path() -> Path:
    # ensure data dir exists
    data_dir = Path(mem_cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / FINGERPRINT_FILE.name

def index_memories_md_if_needed() -> None:
    """
    Idempotently embed docs/memory.md into the MemoryStore as 'memory' entries.
    Re-runs only if the file content (after whitespace normalization) changed.
    """
    chunk_chars = int(id_cfg.get("memories_chunk_chars", 1200))
    if not MEMORIES_MD.exists():
        logger.info("No docs/memory.md found; skipping memory ingestion.")
        return

    raw = _read_text(MEMORIES_MD)
    norm = _normalize_whitespace(raw)
    if not norm:
        logger.info("docs/memory.md is empty after normalization; skipping.")
        return

    fp_current = _sha256_bytes(norm.encode("utf-8"))
    fp_file = _fingerprint_path()
    fp_previous = fp_file.read_text(encoding="utf-8").strip() if fp_file.exists() else ""

    if fp_current == fp_previous:
        logger.info("docs/memory.md unchanged (fingerprint match); skipping re-index.")
        return

    logger.info("Indexing docs/memory.md into vector memory (chunk=%s chars)...", chunk_chars)
    chunks = _chunk_text(norm, chunk_chars)
    added = 0
    for ch in chunks:
        try:
            MEM.add("memory", ch)
            added += 1
        except Exception as e:
            logger.warning("Failed to embed a memory chunk (%d chars): %s", len(ch), e)

    try:
        fp_file.write_text(fp_current, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to persist memory.md fingerprint: %s", e)

    logger.info("Memory ingestion complete: %d chunk(s) added.", added)

# Run the ingestion once at startup
try:
    index_memories_md_if_needed()
except Exception as e:
    logger.exception("Auto-indexing docs/memory.md failed: %s", e)

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

    # --- Memory retrieval ---
    k = int(mem_cfg.get("max_context_snippets", 6))
    snippets = MEM.search(query, k=k) if k > 0 else []

    # --- System prompt assembly ---
    sys_lines: List[str] = []
    # Optional top-level system prompt from anchors file
    if ANCH.system_prompt:
        sys_lines.append(ANCH.system_prompt.strip())
    else:
        sys_lines.append(f"You are {ANCH.system_name}.")
        sys_lines.append("Be truthful, loyal, concise, and warm.")

    # Identity anchors (always-on + rotating according to policy)
    required = set(inp.anchor_tags) if inp.anchor_tags else None
    for a in ANCH.get_for_prompt(required_tags=required):
        sys_lines.append(a)

    # Web snippets
    if web_lines:
        sys_lines.append("\n".join(web_lines))
        sys_lines.append("Use these sources and cite [numbers] or URLs in your answer.")

    # Memory snippets
    if snippets:
        sys_lines.append("Relevant memory:")
        for m in snippets:
            sys_lines.append(f"- {m['role']}: {m['content']}")

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
    return {"anchors": ANCH.list_anchors(include_disabled=include_disabled)}

@app.post("/anchors")
async def add_anchor(inp: AnchorIn):
    txt = (inp.anchor or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="Anchor cannot be empty.")
    ANCH.add_anchor(txt)
    return {"anchors": ANCH.list_anchors()}

@app.delete("/anchors")
async def remove_anchor(text: str = Query(..., min_length=1)):
    removed = ANCH.remove_anchor(text)
    if not removed:
        raise HTTPException(status_code=404, detail="Anchor not found.")
    return {"anchors": ANCH.list_anchors()}

@app.patch("/anchors")
async def update_anchor(inp: AnchorUpdateIn):
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
    return {"anchors": ANCH.list_anchors(include_disabled=True)}

@app.post("/anchors/select")
async def select_anchors(
    n: Optional[int] = Query(None, ge=0),
    tag: Optional[List[str]] = Query(None, alias="tags")
):
    tags = tag if tag else None
    return {"selected": ANCH.get_for_prompt(n_override=n, required_tags=tags)}

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