from __future__ import annotations

from pathlib import Path
from chat_server.memory import DiskMemory


def test_disk_memory_roundtrip(tmp_path: Path):
    mem = DiskMemory(str(tmp_path))
    identity = "user123"
    msg = {"user": "hi", "assistant": "hello"}
    mem.append(identity, msg)
    assert mem.load(identity) == [msg]


def test_disk_memory_summarises_on_overflow(tmp_path: Path):
    # Small limits to trigger summarization quickly
    mem = DiskMemory(str(tmp_path), max_messages=5, summary_keep=2)
    identity = "alice"

    # Append 6 messages => should exceed max_messages and create a summary
    for i in range(6):
        mem.append(identity, {"user": f"u{i}", "assistant": f"a{i}"})

    history = mem.load(identity)
    assert isinstance(history, list)
    assert len(history) == 1 + 2  # one summary + summary_keep recent messages
    assert "summary" in history[0]  # first item is the summary entry
    # The most recent two messages should be present verbatim
    assert history[-2] == {"user": "u4", "assistant": "a4"}
    assert history[-1] == {"user": "u5", "assistant": "a5"}


def test_disk_memory_folds_newer_into_existing_summary(tmp_path: Path):
    mem = DiskMemory(str(tmp_path), max_messages=5, summary_keep=2)
    identity = "bob"

    # Trigger initial summary
    for i in range(6):
        mem.append(identity, {"user": f"u{i}", "assistant": f"a{i}"})
    history1 = mem.load(identity)
    assert "summary" in history1[0]
    first_summary = history1[0]["summary"]

    # Add 3 more -> older ones should fold into the existing summary
    for i in range(6, 9):
        mem.append(identity, {"user": f"u{i}", "assistant": f"a{i}"})

    history2 = mem.load(identity)
    assert "summary" in history2[0]
    # Summary should have grown or at least changed
    assert history2[0]["summary"] != first_summary
    # Still one summary + the most recent `summary_keep` messages
    assert len(history2) == 1 + 2
    assert history2[-2] == {"user": "u7", "assistant": "a7"}
    assert history2[-1] == {"user": "u8", "assistant": "a8"}


def test_disk_memory_loads_empty_for_missing_identity(tmp_path: Path):
    mem = DiskMemory(str(tmp_path))
    assert mem.load("nope") == []