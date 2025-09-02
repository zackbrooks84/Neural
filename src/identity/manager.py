from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Optional


class AnchorManager:
    """Manage identity anchors with optional rotation and persistence."""

    def __init__(
        self,
        path: str,
        anchors_per_query: Optional[int] = None,
        injection_mode: str = "every_query",
    ) -> None:
        self.path = Path(path)
        self.anchors_per_query = anchors_per_query
        self.injection_mode = injection_mode
        self._data = self._load()
        self._cursor = 0
        self._injected_once = False

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def system_name(self) -> str:
        return self._data.get("system_name", "Assistant")

    @property
    def anchors(self) -> List[str]:
        return self._data.setdefault("anchors", [])

    def list_anchors(self) -> List[str]:
        return list(self.anchors)

    def add_anchor(self, text: str) -> None:
        text = text.strip()
        if text and text not in self.anchors:
            self.anchors.append(text)
            self._save()

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True)

    def get_for_prompt(self) -> List[str]:
        """Return anchors for the current prompt respecting injection rules."""
        if self.injection_mode == "session_start" and self._injected_once:
            return []

        anchors = self.anchors
        if self.anchors_per_query:
            start = self._cursor
            end = start + self.anchors_per_query
            selected = anchors[start:end]
            if len(selected) < self.anchors_per_query and anchors:
                remaining = self.anchors_per_query - len(selected)
                selected.extend(anchors[:remaining])
                self._cursor = remaining
            else:
                self._cursor = end % max(len(anchors), 1)
        else:
            selected = list(anchors)

        if self.injection_mode == "session_start":
            self._injected_once = True
        return selected
