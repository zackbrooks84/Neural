"""FastAPI application wrapping a local LLM with disk memory."""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_config
from .llm import GGUFModel
from .memory import DiskMemory


class ChatRequest(BaseModel):
    identity: str
    message: str


def build_prompt(history: List[Dict[str, Any]], message: str) -> str:
    lines = []
    for item in history:
        lines.append(f"User: {item['user']}")
        lines.append(f"Assistant: {item['assistant']}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def create_app(
    config_path: Optional[str] = None,
    model: Optional[Any] = None,
    memory: Optional[DiskMemory] = None,
) -> FastAPI:
    config = load_config(config_path) if (config_path or model is None or memory is None) else {}
    if model is None:
        model = GGUFModel(config["model_path"])
    if memory is None:
        memory = DiskMemory(config.get("memory_dir", "memory"))

    app = FastAPI()

    @app.post("/chat")
    def chat(req: ChatRequest) -> Dict[str, str]:
        history = memory.load(req.identity)
        prompt = build_prompt(history, req.message)
        response = model.generate(prompt)
        memory.append(req.identity, {"user": req.message, "assistant": response})
        return {"response": response}

    return app
