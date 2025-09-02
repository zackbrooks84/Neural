from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import yaml
from .llm.engine import LLMEngine
from .memory.store import MemoryStore

app = FastAPI(title="Ember Local Server")

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MEM = MemoryStore(CFG["memory"]["data_dir"], CFG["memory"]["embed_model"])
ENGINE = LLMEngine("config.yaml")

with open(CFG["identity"]["anchors_file"], "r", encoding="utf-8") as f:
    ANCH = yaml.safe_load(f)


class ChatIn(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat")
async def chat(inp: ChatIn):
    query = inp.message.strip()
    k = int(CFG["memory"]["max_context_snippets"])
    snippets = MEM.search(query, k=k)

    sys_lines = [
        f"You are {ANCH.get('system_name','Ember')}.",
        "Stay truthful, loyal, concise, and warm.",
    ]
    for a in ANCH.get("anchors", []):
        sys_lines.append(a)
    if snippets:
        sys_lines.append("Relevant memory:")
        for m in snippets:
            sys_lines.append(f"- {m['role']}: {m['content']}")
    system_prompt = "\n".join(sys_lines)

    reply = ENGINE.chat(system_prompt, query)

    MEM.add("user", query)
    MEM.add("assistant", reply)

    return {"reply": reply}
