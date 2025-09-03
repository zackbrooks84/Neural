"""Disk-based conversation memory keyed by identity (thread-safe, atomic)."""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _safe_identity(name: str) -> str:
    # Keep it readable but filesystem-safe.
    s = re.sub(r"[^\w.\-@]+", "_", name.strip() or "default")
    return s[:128]  # avoid absurdly long filenames


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


@dataclass
class SummaryPolicy:
    """Controls how memory is summarized/pruned."""
    max_messages: int = 100         # trigger summarization once history exceeds this
    summary_keep: int = 20          # keep last N messages verbatim
    summary_chars: int = 2000       # max chars to keep inside the summary blob
    max_file_mb: int = 16           # safety valve: rotate/trim if file grows too large


# -----------------------------
# DiskMemory
# -----------------------------
class DiskMemory:
    """JSON-based per-identity conversation store with lightweight summarization.

    Layout:
        memory_dir/
          <identity>.json       # list[dict] with optional item[0]["summary"]
          (optional) logs/<identity>.jsonl  # append-only log if JSONL enabled

    Backwards compatible with the original API:
        - load(identity) -> List[Dict[str, Any]]
        - append(identity, message) -> None
    """

    def __init__(
        self,
        memory_dir: str,
        *,
        max_messages: int = 100,
        summary_keep: int = 20,
        summary_chars: int = 2000,
        max_file_mb: int = 16,
        use_jsonl: bool = False,
    ) -> None:
        self.root = Path(memory_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.root / "logs"
        if use_jsonl:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.policy = SummaryPolicy(
            max_messages=max_messages,
            summary_keep=summary_keep,
            summary_chars=summary_chars,
            max_file_mb=max_file_mb,
        )
        self.use_jsonl = use_jsonl
        self._lock = threading.RLock()

    # --------- paths ----------
    def _json_path(self, identity: str) -> Path:
        return self.root / f"{_safe_identity(identity)}.json"

    def _jsonl_path(self, identity: str) -> Path:
        return self.logs_dir / f"{_safe_identity(identity)}.jsonl"

    # --------- core API ----------
    def load(self, identity: str) -> List[Dict[str, Any]]:
        """Load full conversation list for identity (may include a leading summary)."""
        path = self._json_path(identity)
        if not path.exists():
            return []
        try:
            return _read_json(path)
        except Exception:
            # Corruption fallback: keep a backup and start fresh.
            with self._lock:
                bad = path.with_suffix(".corrupt.json")
                try:
                    path.rename(bad)
                except Exception:
                    pass
            return []

    def append(self, identity: str, message: Dict[str, Any]) -> None:
        """Append message and summarize/prune as needed.

        Expected message keys: {"user": str} or {"assistant": str} or both.
        Additional metadata fields are preserved.
        """
        if not isinstance(message, dict):
            raise TypeError("message must be a dict")
        if not any(k in message for k in ("user", "assistant", "summary")):
            raise ValueError("message must contain at least 'user' or 'assistant' or 'summary'")

        with self._lock:
            history = self.load(identity)
            history.append(message)

            # Summarize when exceeding max_messages
            history = self._summarize_if_needed(history)

            # Hard safety: if file is too large, drop oldest concrete messages into summary
            history = self._enforce_file_size(identity, history)

            # Persist JSON
            _write_json(self._json_path(identity), history)

            # Optional JSONL append-only log
            if self.use_jsonl:
                self._append_jsonl(identity, message)

    # --------- convenience ----------
    def get_context(self, identity: str, k: int = 6) -> List[Dict[str, Any]]:
        """Return up to k most recent concrete messages (ignores the summary blob)."""
        hist = self.load(identity)
        if not hist:
            return []
        # exclude leading summary item
        items = hist[1:] if hist and isinstance(hist[0], dict) and "summary" in hist[0] else hist
        return items[-max(0, k):]

    def clear(self, identity: str) -> None:
        """Delete conversation for identity."""
        with self._lock:
            try:
                self._json_path(identity).unlink(missing_ok=True)
            except Exception:
                pass
            if self.use_jsonl:
                try:
                    self._jsonl_path(identity).unlink(missing_ok=True)
                except Exception:
                    pass

    def list_identities(self) -> List[str]:
        """Return all identities with stored memory."""
        out: List[str] = []
        for p in self.root.glob("*.json"):
            out.append(p.stem)
        return sorted(out)

    def delete(self, identity: str) -> bool:
        """Alias for clear() with boolean result."""
        before = self._json_path(identity).exists() or (self.use_jsonl and self._jsonl_path(identity).exists())
        self.clear(identity)
        after = self._json_path(identity).exists() or (self.use_jsonl and self._jsonl_path(identity).exists())
        return before and not after

    def export_text(self, identity: str, limit_chars: int = 8000) -> str:
        """Export a human-readable text of the conversation (summary + recent turns)."""
        hist = self.load(identity)
        buf = io.StringIO()
        if hist and "summary" in hist[0]:
            buf.write("=== SUMMARY ===\n")
            buf.write((hist[0].get("summary") or "").strip() + "\n\n")
            hist = hist[1:]
        for m in hist:
            u = (m.get("user") or "").strip()
            a = (m.get("assistant") or "").strip()
            if u:
                buf.write(f"user: {u}\n")
            if a:
                buf.write(f"assistant: {a}\n")
        out = buf.getvalue()
        return out[:limit_chars]

    # --------- internals ----------
    def _append_jsonl(self, identity: str, message: Dict[str, Any]) -> None:
        try:
            path = self._jsonl_path(identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
        except Exception:
            # Non-fatal
            pass

    def _summarize_if_needed(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(history)
        p = self.policy
        if n <= p.max_messages:
            # If a summary exists and we have more than keep+1, fold middle into summary.
            if history and "summary" in history[0] and len(history) > p.summary_keep + 1:
                existing = history[0].get("summary") or ""
                middle = history[1:-p.summary_keep]  # messages to fold into summary
                if middle:
                    folded = self._summarise(middle, existing_prefix=existing)
                    return [{"summary": folded}] + history[-p.summary_keep:]
            return history

        # First time exceeding threshold: summarize everything but last K
        to_summarize = history[:-p.summary_keep]
        recent = history[-p.summary_keep:]
        summary_text = self._summarise(to_summarize)
        return [{"summary": summary_text}] + recent

    def _enforce_file_size(self, identity: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """If file grows beyond policy.max_file_mb, prune more aggressively."""
        path = self._json_path(identity)
        try:
            size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        except Exception:
            size_mb = 0.0

        if size_mb <= self.policy.max_file_mb:
            return history

        # Aggressive prune: rebuild summary using all but last summary_keep messages.
        recent = history[-self.policy.summary_keep:] if len(history) > self.policy.summary_keep else history
        folded = self._summarise(history[:-len(recent)])
        return [{"summary": folded}] + recent

    def _summarise(self, messages: List[Dict[str, Any]], *, existing_prefix: str = "") -> str:
        """Role-aware compaction with bounded size and whitespace cleanup."""
        lines: List[str] = []
        if existing_prefix:
            lines.append(existing_prefix.strip())

        for m in messages:
            # Keep any custom metadata minimally if present.
            if "summary" in m:
                # Merge nested summaries cautiously
                s = str(m.get("summary") or "").strip()
                if s:
                    lines.append(s)
                continue

            user = (m.get("user") or "").strip()
            if user:
                # compress internal whitespace to keep it compact
                user = re.sub(r"\s+", " ", user)
                lines.append(f"user: {user}")

            assistant = (m.get("assistant") or "").strip()
            if assistant:
                assistant = re.sub(r"\s+", " ", assistant)
                lines.append(f"assistant: {assistant}")

        joined = "\n".join(lines).strip()
        # Bound the total characters to summary_chars.
        if len(joined) > self.policy.summary_chars:
            head = self.policy.summary_chars - 200  # leave room for trailer
            head = max(head, 0)
            joined = joined[:head].rstrip() + "\n… [summary truncated]"
        return joined