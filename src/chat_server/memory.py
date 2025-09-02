"""Disk-based conversation memory keyed by identity."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class DiskMemory:
    """Simple JSON-based memory storage on disk.

    Conversations are stored per identity in ``memory_dir``. Each file
    contains a list of message dictionaries with ``user`` and
    ``assistant`` fields.
    """

    def __init__(self, memory_dir: str) -> None:
        self.memory_dir = memory_dir
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
        history = self.load(identity)
        history.append(message)
        with open(self._path(identity), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
