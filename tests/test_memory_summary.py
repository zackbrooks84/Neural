from __future__ import annotations

from pathlib import Path
from chat_server.memory import DiskMemory


def test_disk_memory_summarises(tmp_path: Path):
    mem = DiskMemory(str(tmp_path), max_messages=5, summary_keep=2)
    identity = "id"

    # 7 appends -> exceeds max_messages (5) and should create a summary
    for i in range(7):
        mem.append(identity, {"user": f"u{i}", "assistant": f"a{i}"})

    history = mem.load(identity)

    # Expect: [ {summary: ...}, {user:u5, assistant:a5}, {user:u6, assistant:a6} ]
    assert isinstance(history, list)
    assert len(history) == 3, "Expected one summary + last two interactions"
    assert isinstance(history[0], dict) and "summary" in history[0], "First entry must be a summary dict"

    assert history[1] == {"user": "u5", "assistant": "a5"}
    assert history[2] == {"user": "u6", "assistant": "a6"}

    # Summary should be a non-empty string, truncated by implementation if needed
    summary = history[0]["summary"]
    assert isinstance(summary, str) and summary.strip(), "Summary text should be non-empty"