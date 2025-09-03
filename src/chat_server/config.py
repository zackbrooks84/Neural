"""Configuration loading utilities for the chat server.

This module handles layered configuration:
1. Explicit path argument (highest precedence)
2. Environment variable CHAT_SERVER_CONFIG
3. Fallback to "config/default.yaml"

It also supports optional overrides from environment variables with prefix
``CHAT_SERVER__`` (e.g., CHAT_SERVER__MODEL__MODEL_PATH=/tmp/model.gguf).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides with prefix CHAT_SERVER__."""
    prefix = "CHAT_SERVER__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        # e.g., CHAT_SERVER__MODEL__MODEL_PATH -> cfg["model"]["model_path"]
        parts = key[len(prefix):].lower().split("__")
        sub = cfg
        for p in parts[:-1]:
            if p not in sub or not isinstance(sub[p], dict):
                sub[p] = {}
            sub = sub[p]
        leaf = parts[-1]
        # Attempt to parse simple types (bool, int, float)
        if value.lower() in {"true", "false"}:
            sub[leaf] = value.lower() == "true"
        else:
            try:
                if "." in value:
                    sub[leaf] = float(value)
                else:
                    sub[leaf] = int(value)
            except ValueError:
                sub[leaf] = value
    return cfg


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load YAML configuration for the chat server.

    Parameters
    ----------
    path : str | None
        Optional path to a configuration file. If not provided, the
        environment variable ``CHAT_SERVER_CONFIG`` is consulted. As a
        last resort ``config/default.yaml`` is used.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary with environment overrides applied.
    """
    # Resolve path precedence
    if path is None:
        path = os.environ.get("CHAT_SERVER_CONFIG", "config/default.yaml")

    path_obj = Path(path)
    if not path_obj.exists():
        # Provide a safe empty config with warning
        print(f"[config] Warning: config file not found at {path_obj}. Using defaults.")
        cfg: Dict[str, Any] = {
            "model": {"model_path": "models/model.gguf"},
            "memory": {"data_dir": "data"},
        }
        return _apply_env_overrides(cfg)

    with path_obj.open("r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse config file {path_obj}: {e}")

    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config format in {path_obj}, expected dict.")

    return _apply_env_overrides(cfg)