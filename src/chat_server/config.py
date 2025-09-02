"""Configuration loading for the chat server."""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load YAML configuration for the chat server.

    Parameters
    ----------
    path: str | None
        Optional path to a configuration file. If not provided, the
        environment variable ``CHAT_SERVER_CONFIG`` is consulted. As a
        last resort ``config/default.yaml`` is used.
    """
    if path is None:
        path = os.environ.get("CHAT_SERVER_CONFIG", "config/default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
