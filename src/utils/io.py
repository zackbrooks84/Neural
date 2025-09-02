from pathlib import Path
import json
from typing import Iterable, Dict, Any


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, item: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except OSError as e:
        raise OSError(f"Failed to write to {path}: {e}") from e


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except OSError:
        # If the file can't be read due to I/O limitations, return empty list
        return []
