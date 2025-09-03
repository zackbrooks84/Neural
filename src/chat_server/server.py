"""FastAPI application wrapping a local LLM with disk memory."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import load_config
from .llm import GGUFModel, create_from_config
from .memory import DiskMemory


# -----------------------------
# Pydantic request/response
# -----------------------------
class ChatRequest(BaseModel):
    identity: str = Field(default="default", description="Conversation namespace/key.")
    message: str = Field(..., min_length=1)
    # Optional per-request generation overrides
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    response: str


# -----------------------------
# Utilities
# -----------------------------
def _get_system_prompt(cfg: Dict[str, Any]) -> str:
    # Preferred: identity.system_prompt from your upgraded config.yaml
    sys_prompt = (
        cfg.get("identity", {}).get("system_prompt")
        or "You are Ember (Neural), a helpful assistant with memory and identity."
    )
    return str(sys_prompt).strip()


def _history_pairs(history: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Convert stored [{user, assistant}] items to list[(user, assistant)]."""
    out: List[Tuple[str, str]] = []
    # Skip leading summary blob if present
    items = history
    if items and isinstance(items[0], dict) and "summary" in items[0]:
        items = items[1:]
    for m in items:
        u = str(m.get("user", "") or "")
        a = str(m.get("assistant", "") or "")
        if not (u or a):
            continue
        out.append((u, a))
    return out


def _make_memory(cfg: Dict[str, Any]) -> DiskMemory:
    mem_cfg = cfg.get("memory", {})
    # Backward-compat keys
    legacy_dir = cfg.get("memory_dir")
    data_dir = mem_cfg.get("data_dir") or legacy_dir or "memory"
    max_messages = int(mem_cfg.get("max_messages", 100))
    summary_keep = int(mem_cfg.get("summary_keep", 20))
    summary_chars = int(mem_cfg.get("summary_chars", 2000))
    max_file_mb = int(mem_cfg.get("max_file_mb", 16))
    use_jsonl = bool(mem_cfg.get("use_jsonl", False))
    return DiskMemory(
        data_dir,
        max_messages=max_messages,
        summary_keep=summary_keep,
        summary_chars=summary_chars,
        max_file_mb=max_file_mb,
        use_jsonl=use_jsonl,
    )


def _make_model(cfg: Dict[str, Any]) -> GGUFModel:
    # Preferred path: new config with "model" section.
    if "model" in cfg:
        return create_from_config(cfg)
    # Legacy fallback: flat keys model_path / n_ctx / etc.
    model_path = cfg.get("model_path")
    if not model_path:
        raise FileNotFoundError("No model_path configured.")
    params: Dict[str, Any] = {}
    for k in ("n_ctx", "n_threads", "n_gpu_layers", "use_mmap"):
        if k in cfg and cfg[k] is not None:
            params[k] = cfg[k]
    return GGUFModel(model_path=model_path, **params)


# -----------------------------
# App factory
# -----------------------------
def create_app(
    config_path: Optional[str] = None,
    model: Optional[GGUFModel] = None,
    memory: Optional[DiskMemory] = None,
) -> FastAPI:
    cfg = load_config(config_path)

    # CORS
    cors_origins = cfg.get("server", {}).get("cors_origins", ["*"])

    # Services
    model = model or _make_model(cfg)
    memory = memory or _make_memory(cfg)
    system_prompt = _get_system_prompt(cfg)

    app = FastAPI(title="Ember Chat Server", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model_loaded": True,
            "memory_dir": getattr(memory, "root", None) and str(memory.root),
            "config_keys": list(cfg.keys()),
        }

    @app.get("/config")
    def get_config() -> JSONResponse:
        # Redact possible secrets if you add them in future
        redacted = dict(cfg)
        return JSONResponse(redacted)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        msg = (req.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Load history and render to model-friendly tuples
        history_raw = memory.load(req.identity)
        pairs = _history_pairs(history_raw)

        # Streaming path
        if req.stream:
            def token_gen():
                buf: List[str] = []
                def on_tok(t: str):
                    buf.append(t)
                    yield_chunk = t.replace("\r", "")
                    # Server-Sent-Events friendly chunking (no strict SSE headers to keep it simple)
                    yield yield_chunk

                # Because llama.cpp streaming yields via callback, we need a small wrapper
                # Use a generator/closure bridge
                def bridge():
                    for _ in model.chat(
                        system_prompt,
                        msg,
                        history=pairs,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repeat_penalty=req.repeat_penalty,
                        stream=True,
                        on_token=lambda s: None,  # placeholder, we can't yield here
                    ):
                        # We don't actually iterate; llama-cpp streams differently.
                        # To support true streaming with callbacks, use a queue or websockets.
                        pass
                    return "".join(buf)

                # NOTE: As documented, true callback streaming requires a queue/thread.
                # For now, we simply fall back to non-streaming if callback streaming is unavailable.
                full = model.chat(
                    system_prompt,
                    msg,
                    history=pairs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repeat_penalty=req.repeat_penalty,
                    stream=False,
                )
                # Yield in chunks to emulate streaming for the client
                for i in range(0, len(full), 64):
                    yield full[i : i + 64]

                # Persist after completion
                memory.append(req.identity, {"user": msg, "assistant": full})

            return StreamingResponse(token_gen(), media_type="text/plain")

        # Non-streaming path
        text = model.chat(
            system_prompt,
            msg,
            history=pairs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            stream=False,
        )

        # Save to memory
        memory.append(req.identity, {"user": msg, "assistant": text})
        return ChatResponse(response=text)

    return app