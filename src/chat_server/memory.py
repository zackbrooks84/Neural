"""Disk-based conversation memory keyed by identity."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class DiskMemory:
    """Simple JSON-based memory storage on disk with basic summarization.

    Conversations are stored per identity in ``memory_dir``. Each file
    contains a list of message dictionaries with ``user`` and
    ``assistant`` fields.  When a conversation grows beyond
    ``max_messages`` entries the older portion is condensed into a single
    ``summary`` entry to keep memory files from growing without bound.
    The summarisation is intentionally lightweight: older messages are
    concatenated and truncated, preserving only the most recent
    ``summary_keep`` interactions verbatim.
    """

    def __init__(self, memory_dir: str, *, max_messages: int = 100, summary_keep: int = 20) -> None:
        self.memory_dir = memory_dir
        self.max_messages = max_messages
        self.summary_keep = summary_keep
        os.makedirs(self.memory_dir, exist_ok=True)

    def _path(self, identity: str) -> str:
        return os.path.join(self.memory_dir, f"{identity}.json")

    def load(self, identity: str) -> List[Dict[str, Any]]:
        path = self._path(identity)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def append(self, identity: str, message: Dict[str, Any]) -> None:
        """Append ``message`` to ``identity``'s history and summarise if needed."""

        history = self.load(identity)
        history.append(message)

        if len(history) > self.max_messages:
            # First time the limit is exceeded: summarise everything except
            # the most recent ``summary_keep`` interactions.
            to_summarise = history[:-self.summary_keep]
            recent = history[-self.summary_keep:]
            summary_text = self._summarise(to_summarise)
            history = [{"summary": summary_text}] + recent
        elif history and "summary" in history[0] and len(history) > self.summary_keep + 1:
            # Subsequent appends after a summary exists: fold the new older
            # messages into the existing summary so that only the most
            # recent ``summary_keep`` remain verbatim.
            existing = history[0]["summary"]
            to_summarise = history[1:-self.summary_keep]
            if to_summarise:
                summary_text = existing + "\n" + self._summarise(to_summarise)
                history = [{"summary": summary_text}] + history[-self.summary_keep:]

        with open(self._path(identity), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _summarise(self, messages: List[Dict[str, Any]]) -> str:
        """Create a lightweight summary string from ``messages``.

        The summarisation strategy is deliberately simple: join user and
        assistant turns line by line and truncate the result.  This keeps
        dependencies minimal while still capturing the gist of older
        conversation fragments.
        """

        lines: List[str] = []
        for m in messages:
            user = m.get("user")
            if user:
                lines.append(f"user: {user}")
            assistant = m.get("assistant")
            if assistant:
                lines.append(f"assistant: {assistant}")
        joined = "\n".join(lines)
        # Limit summary size to avoid creating overly large entries.
        return joined[:1000]
