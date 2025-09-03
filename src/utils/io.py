from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any, Generator, Union

PathLike = Union[str, Path]


def ensure_dir(p: PathLike) -> Path:
    """Ensure that a directory exists, returning it as a Path."""
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {p}: {e}") from e
    return p


def append_jsonl(path: PathLike, item: Dict[str, Any]) -> None:
    """Append a JSON-serializable dict as one line to a JSONL file."""
    try:
        line = json.dumps(item, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to serialize item to JSON: {e}") from e

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        raise OSError(f"Failed to write to {path}: {e}") from e


def read_jsonl(path: PathLike, *, stream: bool = False) -> Iterable[Dict[str, Any]]:
    """Read a JSONL file into memory or stream it line by line.

    Parameters
    ----------
    path : str | Path
        The JSONL file path.
    stream : bool
        If True, yield entries lazily (generator).
        If False, return a full list of entries.

    Returns
    -------
    Iterable[Dict[str, Any]]
    """
    p = Path(path)
    if not p.exists():
        return [] if not stream else iter(())

    def _iter() -> Generator[Dict[str, Any], None, None]:
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip corrupt lines but log context
                    print(f"[io.read_jsonl] Skipping corrupt line {line_no} in {p}: {e}")
                    continue

    return _iter() if stream else list(_iter())


def atomic_write_json(path: PathLike, data: Any) -> None:
    """Safely write a JSON file atomically to avoid corruption."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception as e:
        raise OSError(f"Atomic write failed for {p}: {e}") from e