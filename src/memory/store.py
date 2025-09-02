from __future__ import annotations
from pathlib import Path
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .typing import Message
from utils.io import ensure_dir, append_jsonl, read_jsonl


class MemoryStore:
    def __init__(self, data_dir: str, embed_model: str) -> None:
        self.dir = Path(data_dir)
        ensure_dir(self.dir)
        self.mem_path = self.dir / "memory.jsonl"
        self.index_path = self.dir / "faiss.index"
        self.model = SentenceTransformer(embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    def add(self, role: str, content: str) -> None:
        emb = self._embed(content)
        self.index.add(np.array([emb]).astype("float32"))
        append_jsonl(self.mem_path, {"role": role, "content": content})
        self._save_index()

    def search(self, query: str, k: int) -> List[Message]:
        entries: List[Message] = read_jsonl(self.mem_path)  # type: ignore
        if not entries:
            return []
        emb = self._embed(query)
        D, I = self.index.search(np.array([emb]).astype("float32"), min(k, len(entries)))
        results: List[Message] = []
        for idx in I[0]:
            if idx == -1:
                continue
            results.append(entries[idx])  # type: ignore
        return results

    def _embed(self, text: str):
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.astype("float32")
