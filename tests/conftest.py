"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Ensure src/ is on the import path (for local imports without installing as package)
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Default config.yaml path (falls back to repo root/config.yaml)."""
    return project_root / "config.yaml"


@pytest.fixture(scope="function")
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for memory / embeddings during tests."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="function")
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure tests run with a clean environment (no leftover vars)."""
    for var in ["CHAT_SERVER_CONFIG", "OPENAI_API_KEY", "HF_HOME"]:
        monkeypatch.delenv(var, raising=False)
    yield