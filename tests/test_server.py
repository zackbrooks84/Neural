from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from chat_server.server import create_app
from chat_server.memory import DiskMemory


class DummyModel:
    def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        return "ok"


def test_chat_endpoint_roundtrip(tmp_path: Path):
    """Basic sanity check: /chat returns 200, echoes 'ok', and persists memory."""
    memory = DiskMemory(str(tmp_path))
    app = create_app(model=DummyModel(), memory=memory)
    client = TestClient(app)

    r = client.post("/chat", json={"identity": "u1", "message": "Hello"})
    assert r.status_code == 200
    assert r.json()["response"] == "ok"
    assert memory.load("u1") == [{"user": "Hello", "assistant": "ok"}]


class SpyModel:
    """Spy model that records the last prompt it received."""
    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def test_prompt_includes_history(tmp_path: Path):
    """Second turn should include the prior (user, assistant) turns in the prompt."""
    memory = DiskMemory(str(tmp_path))
    spy = SpyModel(reply="second")
    app = create_app(model=spy, memory=memory)
    client = TestClient(app)

    # First turn
    r1 = client.post("/chat", json={"identity": "alice", "message": "Hi there"})
    assert r1.status_code == 200
    assert r1.json()["response"] == "second"

    # Second turn
    r2 = client.post("/chat", json={"identity": "alice", "message": "How are you?"})
    assert r2.status_code == 200
    assert r2.json()["response"] == "second"

    # Spy should have captured the prompt from the second call
    assert spy.last_prompt is not None
    prompt = spy.last_prompt or ""

    # Expect prior transcript lines in the prompt
    assert "User: Hi there" in prompt
    assert "Assistant: second" in prompt  # previous assistant reply
    # And the new user message queued at the end
    assert prompt.strip().endswith("Assistant:")

    # Memory should contain two turns
    history = memory.load("alice")
    assert len(history) == 2
    assert history[0] == {"user": "Hi there", "assistant": "second"}
    assert history[1] == {"user": "How are you?", "assistant": "second"}