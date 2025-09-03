from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Sampling:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = False


class LLMEngine:
    """
    Thin, resilient wrapper around llama.cpp (GGUF) with chat-style prompting.

    Usage:
        engine = LLMEngine("config.yaml")
        text = engine.chat(system, user, history=[("hi", "hello!")])

    Notes:
        - Supports both `llama.create_completion(...)` and `llama(prompt, ...)` APIs.
        - Auto-threads and GPU offload detection.
        - Retries without mmap if the filesystem rejects memory-mapping.
    """

    def __init__(self, cfg_path: str = "config.yaml") -> None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Support your nested layout: model: { model_dir, model_path, ... }
        m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        model_dir = m.get("model_dir", "models")
        model_path = m.get("model_path")
        model_file = Path(model_path if Path(model_path or "").is_absolute()
                          else Path(model_dir) / str(model_path or ""))

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Import llama lazily to avoid hard dependency for tests
        from llama_cpp import Llama, llama_supports_gpu_offload  # type: ignore

        # Threads (auto by default)
        threads = m.get("n_threads")
        if threads is None or int(threads) <= 0:
            threads = os.cpu_count() or 1

        # GPU offload
        gpu_layers = m.get("n_gpu_layers")
        if gpu_layers is None:
            try:
                gpu_layers = -1 if llama_supports_gpu_offload() else 0
            except Exception:
                gpu_layers = 0

        # Build kwargs for llama
        llm_kwargs: Dict[str, Any] = dict(
            model_path=str(model_file),
            n_ctx=m.get("n_ctx", 4096),
            n_threads=int(threads),
            n_gpu_layers=int(gpu_layers),
            verbose=False,
            use_mmap=bool(m.get("use_mmap", True)),
        )
        # Drop any None entries (older builds are picky)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        # Init with mmap retry
        try:
            self.llm = Llama(**llm_kwargs)
        except OSError as e:
            if llm_kwargs.get("use_mmap", True):
                logger.warning("mmap load failed, retrying without mmap: %s", e)
                llm_kwargs["use_mmap"] = False
                self.llm = Llama(**llm_kwargs)
            else:
                logger.exception("Failed to load model: %s", e)
                raise
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            raise

        # Default sampling (from config if present)
        self.sampling = Sampling(
            max_new_tokens=int(m.get("max_new_tokens", 512)),
            temperature=float(m.get("temperature", 0.7)),
            top_p=float(m.get("top_p", 0.95)),
            top_k=int(m.get("top_k", 50)),
            repeat_penalty=float(m.get("repeat_penalty", 1.1)),
            stop=None,  # set below
            stream=False,
        )
        # Common safe stops; you can tweak in config later if needed
        self.default_stops = ["</s>", "</assistant>", "###", "User:", "Assistant:"]
        # Some builds expose an official chat template helper
        self._supports_chat_template = hasattr(self.llm, "apply_chat_template")

    # ----------------------------
    # Public API
    # ----------------------------
    def chat(
        self,
        system: str,
        user: str,
        *,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Optional[bool] = None,
        on_token: Optional[Any] = None,  # placeholder for future true streaming
    ) -> str:
        """
        Chat-style generation with optional history.

        history: list of (user, assistant) tuples in chronological order.
        """
        messages = self._build_messages(system, user, history or [])
        prompt = self._render_chat(messages)

        cfg = self._merge_sampling(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=stream,
        )

        # Call llama.cpp in a version-tolerant way
        return self._complete(prompt, cfg)

    # ----------------------------
    # Internals
    # ----------------------------
    def _build_messages(
        self, system: str, user: str, history: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": (system or "").strip()}]
        for u, a in history:
            if u:
                msgs.append({"role": "user", "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": (user or "").strip()})
        return msgs

    def _render_chat(self, messages: List[Dict[str, str]]) -> str:
        # Prefer official chat template if available
        if self._supports_chat_template:
            try:
                tpl = self.llm.apply_chat_template(messages, add_generation_prompt=True)  # type: ignore[attr-defined]
                return tpl.decode("utf-8", "ignore") if isinstance(tpl, (bytes, bytearray)) else str(tpl)
            except Exception:
                pass

        # Robust fallback template (instruct-style)
        lines: List[str] = []
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
        if sys:
            lines.append("### System\n" + sys + "\n")
        for m in messages:
            if m["role"] == "user":
                lines.append("### User\n" + m["content"].strip() + "\n")
            elif m["role"] == "assistant":
                lines.append("### Assistant\n" + m["content"].strip() + "\n")
        lines.append("### Assistant\n")
        return "\n".join(lines)

    def _merge_sampling(
        self,
        *,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
        stop: Optional[Sequence[str]],
        stream: Optional[bool],
    ) -> Sampling:
        return Sampling(
            max_new_tokens=max_new_tokens or self.sampling.max_new_tokens,
            temperature=self.sampling.temperature if temperature is None else float(temperature),
            top_p=self.sampling.top_p if top_p is None else float(top_p),
            top_k=self.sampling.top_k if top_k is None else int(top_k),
            repeat_penalty=self.sampling.repeat_penalty if repeat_penalty is None else float(repeat_penalty),
            stop=list(stop) if stop is not None else (self.sampling.stop or self.default_stops),
            stream=self.sampling.stream if stream is None else bool(stream),
        )

    def _complete(self, prompt: str, cfg: Sampling) -> str:
        """
        Version-tolerant call into llama.cpp:
        - Prefer `create_completion(prompt=...)`
        - Else fallback to callable: `llm(prompt, ...)`
        """
        args = dict(
            max_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            top_k=int(cfg.top_k),
            repeat_penalty=float(cfg.repeat_penalty),
            stop=cfg.stop,
            stream=bool(cfg.stream),
        )

        # Streaming note:
        # llama-cpp streams token deltas via the return iterator when stream=True,
        # but mixing that with this synchronous API requires a queue/thread. For now
        # we emulate chunked output by running non-streaming and splitting later.
        if cfg.stream:
            # Emulated streaming: run non-streaming and return full text.
            args["stream"] = False

        try:
            if hasattr(self.llm, "create_completion"):
                out = self.llm.create_completion(prompt=prompt, **args)  # type: ignore[attr-defined]
                text = out["choices"][0]["text"]
            else:
                # Newer versions often use the callable interface
                out = self.llm(prompt, **args)  # type: ignore[call-arg]
                text = out["choices"][0]["text"]
        except Exception as e:
            logger.exception("LLM completion failed: %s", e)
            raise

        return (text or "").strip()