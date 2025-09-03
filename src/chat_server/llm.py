"""Wrapper for loading a GGUF model via llama.cpp with chat-style helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# -----------------------------
# Types & defaults
# -----------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = False


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


# -----------------------------
# GGUF wrapper
# -----------------------------

class GGUFModel:
    """Thin wrapper around :mod:`llama_cpp` to generate and chat."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        model_path : str
            Path to .gguf weights.
        kwargs : Any
            Passed to llama_cpp.Llama with some smart defaults:
              - n_threads: defaults to os.cpu_count()
              - n_gpu_layers: auto if gpu offload supported; else 0
              - use_mmap: default True, with fallback retry if OSError
              - n_ctx: respected if provided (context window)
        """
        # Lazy import so unit tests pass without the dep.
        from llama_cpp import Llama, llama_supports_gpu_offload  # type: ignore

        # Threads
        threads = kwargs.get("n_threads")
        if threads is None or int(threads) <= 0:
            kwargs["n_threads"] = os.cpu_count() or 1

        # GPU offload
        if "n_gpu_layers" not in kwargs or kwargs["n_gpu_layers"] is None:
            kwargs["n_gpu_layers"] = -1 if llama_supports_gpu_offload() else 0

        # mmap retry
        use_mmap = _bool(kwargs.get("use_mmap", True), True)
        kwargs["use_mmap"] = use_mmap

        # Some builds support vocab only mmap; keep minimal args clean
        # Allow caller to pass n_ctx / rope_freq_base / etc. untouched.
        try:
            self._llama = Llama(model_path=model_path, **kwargs)
        except OSError:
            if use_mmap:
                # Retry without mmap on network filesystems / Windows oddities.
                kwargs["use_mmap"] = False
                self._llama = Llama(model_path=model_path, **kwargs)
            else:
                raise

        # Cache whether this build supports chat_format (newer llama.cpp)
        self._supports_chat_template = hasattr(self._llama, "apply_chat_template")

        # Default sampling from env or sane constants
        self._gen_cfg = GenerationConfig(
            max_new_tokens=int(os.environ.get("LLM_MAX_NEW", "256")),
            temperature=float(os.environ.get("LLM_TEMP", "0.7")),
            top_p=float(os.environ.get("LLM_TOP_P", "0.95")),
            top_k=int(os.environ.get("LLM_TOP_K", "50")),
            repeat_penalty=float(os.environ.get("LLM_REPEAT_PEN", "1.1")),
            stop=None,
            stream=False,
        )

        # Common stop tokens for instruction models
        self._default_stops = ["</s>", "###", "User:", "Assistant:"]

    # -------------------------
    # Low-level text completion
    # -------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional<float] = None,  # type: ignore[valid-type]
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Optional[bool] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Text completion on a raw prompt."""
        cfg = self._merge_cfg(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=stream,
        )
        # llama.cpp API returns tokens via iterator when stream=True
        kwargs = dict(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop or self._default_stops,
            stream=cfg.stream,
        )

        if cfg.stream:
            out_text = []
            for part in self._llama(prompt, **kwargs):
                # part is a delta dict; extract token text
                token = part.get("choices", [{}])[0].get("text", "")
                if token:
                    if on_token:
                        on_token(token)
                    out_text.append(token)
            return "".join(out_text)

        # Non-streaming
        result = self._llama(prompt, **kwargs)
        return result["choices"][0]["text"]

    # -------------------------
    # Chat helper
    # -------------------------
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Optional[bool] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Chat-style generation using either a chat template or a simple fallback.

        Parameters
        ----------
        system_prompt : str
            The system instruction (identity, tools, memory context).
        user_message : str
            The current user query.
        history : list[tuple[str, str]] | None
            Optional prior (user, assistant) turns.
        """
        messages = self._build_messages(system_prompt, user_message, history or [])
        prompt = self._render_chat(messages)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=stream,
            on_token=on_token,
        )

    # -------------------------
    # Internals
    # -------------------------
    def _merge_cfg(
        self,
        *,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
        stop: Optional[Sequence[str]],
        stream: Optional[bool],
    ) -> GenerationConfig:
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens or self._gen_cfg.max_new_tokens,
            temperature=self._gen_cfg.temperature if temperature is None else float(temperature),
            top_p=self._gen_cfg.top_p if top_p is None else float(top_p),
            top_k=self._gen_cfg.top_k if top_k is None else int(top_k),
            repeat_penalty=self._gen_cfg.repeat_penalty if repeat_penalty is None else float(repeat_penalty),
            stop=list(stop) if stop is not None else self._gen_cfg.stop,
            stream=self._gen_cfg.stream if stream is None else bool(stream),
        )
        return cfg

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for u, a in history:
            if u:
                msgs.append({"role": "user", "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _render_chat(self, messages: List[Dict[str, str]]) -> str:
        """Render chat messages to a prompt string.

        Uses llama.cpp chat template if available; otherwise falls back
        to a simple, robust instruction-style format.
        """
        # Newer llama-cpp exposes apply_chat_template similar to HF
        if self._supports_chat_template:
            try:
                # Some builds use: self._llama.apply_chat_template(msgs, add_generation_prompt=True)
                tpl = self._llama.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                if isinstance(tpl, bytes):
                    return tpl.decode("utf-8", errors="ignore")
                return str(tpl)
            except Exception:
                pass  # fall through to manual template

        # Fallback: generic instruction template
        # This works with most instruct-tuned LLaMA-style weights.
        lines: List[str] = []
        sys_lines = [m["content"] for m in messages if m["role"] == "system"]
        if sys_lines:
            lines.append("### System\n" + "\n".join(sys_lines).strip() + "\n")

        # prior turns
        for m in messages:
            if m["role"] == "user":
                lines.append("### User\n" + m["content"].strip() + "\n")
            elif m["role"] == "assistant":
                lines.append("### Assistant\n" + m["content"].strip() + "\n")

        # Generation cue
        lines.append("### Assistant\n")
        return "\n".join(lines)


# -----------------------------
# Convenience factory
# -----------------------------

def create_from_config(cfg: Dict[str, Any]) -> GGUFModel:
    """Create GGUFModel from a config dict (e.g., loaded YAML)."""
    model_cfg = (cfg or {}).get("model", {}) if isinstance(cfg, dict) else {}
    model_dir = model_cfg.get("model_dir")
    model_path = model_cfg.get("model_path")
    if model_dir and model_path and not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, model_path)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path!r}")

    params = {
        "n_ctx": model_cfg.get("n_ctx", 4096),
        "n_threads": model_cfg.get("n_threads"),
        "n_gpu_layers": model_cfg.get("n_gpu_layers"),
        "use_mmap": model_cfg.get("use_mmap", True),
    }
    # Remove None entries (llama.cpp is picky)
    params = {k: v for k, v in params.items() if v is not None}

    return GGUFModel(model_path=model_path, **params)