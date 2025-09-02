from __future__ import annotations

from chat_server.memory import DiskMemory


def test_disk_memory_roundtrip(tmp_path):
    memory = DiskMemory(str(tmp_path))
    identity = "user123"
    message = {"user": "hi", "assistant": "hello"}
    memory.append(identity, message)
    assert memory.load(identity) == [message]
