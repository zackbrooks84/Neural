from __future__ import annotations
from pathlib import Path
from llama_cpp import Llama
import yaml


class LLMEngine:
    def __init__(self, cfg_path: str = "config.yaml") -> None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        model_file = Path(m["model_dir"]) / m["model_path"]
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        self.llm = Llama(
            model_path=str(model_file),
            n_ctx=m.get("n_ctx", 4096),
            n_threads=m.get("n_threads", 4),
            n_gpu_layers=m.get("n_gpu_layers", 0),
            verbose=False,
        )
        self.temp = float(m.get("temperature", 0.7))
        self.top_p = float(m.get("top_p", 0.95))

    def chat(self, system: str, user: str) -> str:
        prompt = self._format(system, user)
        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=self.temp,
            top_p=self.top_p,
            stop=["</assistant>"]
        )
        text = out["choices"][0]["text"].strip()
        return text

    def _format(self, system: str, user: str) -> str:
        return (
            f"<system>\n{system}\n</system>\n"
            f"<user>\n{user}\n</user>\n"
            f"<assistant>"
        )
