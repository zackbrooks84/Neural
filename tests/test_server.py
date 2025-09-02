from __future__ import annotations

from fastapi.testclient import TestClient

from chat_server.server import create_app
from chat_server.memory import DiskMemory


class DummyModel:
    def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        return "ok"


def test_chat_endpoint(tmp_path):
    memory = DiskMemory(str(tmp_path))
    app = create_app(model=DummyModel(), memory=memory)
    client = TestClient(app)
    r = client.post("/chat", json={"identity": "u1", "message": "Hello"})
    assert r.status_code == 200
    assert r.json()["response"] == "ok"
    assert memory.load("u1") == [{"user": "Hello", "assistant": "ok"}]
