from __future__ import annotations
from pathlib import Path
import os
import logging
from llama_cpp import Llama, llama_supports_gpu_offload
import yaml

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, cfg_path: str = "config.yaml") -> None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        model_file = Path(m["model_dir"]) / m["model_path"]
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Determine CPU threads dynamically if not specified
        threads = m.get("n_threads")
        if threads is None or threads <= 0:
            threads = os.cpu_count() or 1

        # Determine GPU layer offloading dynamically
        gpu_layers = m.get("n_gpu_layers")
        if gpu_layers is None:
            gpu_layers = -1 if llama_supports_gpu_offload() else 0

        llm_kwargs = dict(
            model_path=str(model_file),
            n_ctx=m.get("n_ctx", 4096),
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
            use_mmap=m.get("use_mmap", True),
        )

        # Retry without memory mapping if disk I/O is limited
        try:
            self.llm = Llama(**llm_kwargs)
        except OSError as e:
            if llm_kwargs.get("use_mmap", True):
                logger.warning("Loading model with mmap failed, retrying without mmap: %s", e)
                llm_kwargs["use_mmap"] = False
                self.llm = Llama(**llm_kwargs)
            else:
                logger.exception("Failed to load model: %s", e)
                raise
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            raise
        self.temp = float(m.get("temperature", 0.7))
        self.top_p = float(m.get("top_p", 0.95))

    def chat(self, system: str, user: str) -> str:
        prompt = self._format(system, user)
        try:
            out = self.llm.create_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=self.temp,
                top_p=self.top_p,
                stop=["</assistant>"]
            )
        except Exception as e:
            logger.exception("LLM completion failed: %s", e)
            raise
        text = out["choices"][0]["text"].strip()
        return text

    def _format(self, system: str, user: str) -> str:
        return (
            f"<system>\n{system}\n</system>\n"
            f"<user>\n{user}\n</user>\n"
            f"<assistant>"
        )
