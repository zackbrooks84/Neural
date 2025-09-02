"""Wrapper for loading a GGUF model via llama.cpp."""
from __future__ import annotations

from typing import Any


class GGUFModel:
    """Thin wrapper around :mod:`llama_cpp` to generate text."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        # Import here so tests can run without the dependency installed.
        from llama_cpp import Llama  # type: ignore

        self.llama = Llama(model_path=model_path, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate text given a prompt."""
        output = self.llama(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"]
