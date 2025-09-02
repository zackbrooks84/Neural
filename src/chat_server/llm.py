"""Wrapper for loading a GGUF model via llama.cpp."""
from __future__ import annotations

import os
from typing import Any


class GGUFModel:
    """Thin wrapper around :mod:`llama_cpp` to generate text."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        # Import here so tests can run without the dependency installed.
        from llama_cpp import Llama, llama_supports_gpu_offload  # type: ignore

        threads = kwargs.get("n_threads")
        if threads is None or threads <= 0:
            kwargs["n_threads"] = os.cpu_count() or 1

        if "n_gpu_layers" not in kwargs:
            kwargs["n_gpu_layers"] = -1 if llama_supports_gpu_offload() else 0

        use_mmap = kwargs.get("use_mmap", True)
        try:
            self.llama = Llama(model_path=model_path, use_mmap=use_mmap, **kwargs)
        except OSError:
            if use_mmap:
                kwargs["use_mmap"] = False
                self.llama = Llama(model_path=model_path, **kwargs)
            else:
                raise

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate text given a prompt."""
        output = self.llama(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"]
