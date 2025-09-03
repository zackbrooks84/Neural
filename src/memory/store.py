from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .typing import Message  # your existing lightweight type alias/dict
from utils.io import ensure_dir, append_jsonl, read_jsonl


# -----------------------------
# Internal types & helpers
# -----------------------------
class _Row(TypedDict, total=False):
    role: str
    content: str
    ts: str
    id: str
    identity: str


@dataclass
class _Config:
    # hard limits / behaviors
    max_content_chars: int = 4000
    dedup_min_chars: int = 40
    dedup_sim_threshold: float = 0.995  # cosine sim after L2-normalization
    max_rebuild_batch: int = 1024       # vectors per FAISS add() during rebuild
    mmr_lambda: float = 0.5             # tradeoff for MMR (relevance vs novelty)
    mmr_candidates: int = 64            # how many FAISS neighbors to pull before MMR
    save_every_adds: int = 1            # persist FAISS after every N adds
    namespace_key: str = "identity"     # field used for namespacing rows


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize(v: np.ndarray) -> np.ndarray:
    # Normalize rows to unit length for cosine via inner product
    if v.ndim == 1:
        v = v[None, :]
    faiss.normalize_L2(v)
    return v


# -----------------------------
# Memory Store
# -----------------------------
class MemoryStore:
    """
    Resilient, vector-backed conversation memory.

    - Persists rows to JSONL (append-only) at `data/memory.jsonl`
    - Stores embeddings in FAISS `IndexFlatIP` at `data/faiss.index`
    - Auto-rebuilds FAISS from JSONL when out of sync
    - Supports simple identities (namespaces) via an optional `identity` field

    Public API (unchanged):
        add(role: str, content: str) -> None
        search(query: str, k: int) -> List[Message]
    """

    def __init__(self, data_dir: str, embed_model: str) -> None:
        self.dir = Path(data_dir)
        ensure_dir(self.dir)
        self.mem_path = self.dir / "memory.jsonl"
        self.index_path = self.dir / "faiss.index"
        self.model_name = embed_model
        self._model: Optional[SentenceTransformer] = None
        self._dim: Optional[int] = None
        self._index: Optional[faiss.Index] = None
        self._cfg = _Config()
        self._lock = threading.RLock()
        self._add_counter = 0  # to control save cadence

        # Try to load existing index; if it fails, we’ll rebuild lazily
        self._load_index_or_mark_rebuild()

    # ----------------- lazy components -----------------
    def _model_ensure(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _index_ensure(self) -> faiss.Index:
        if self._index is None:
            dim = self._dim or self._model_ensure().get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)
        return self._index

    # ----------------- persistence -----------------
    def _load_index_or_mark_rebuild(self) -> None:
        with self._lock:
            if self.index_path.exists():
                try:
                    self._index = faiss.read_index(str(self.index_path))
                    # If we load the index, make sure we also know dim
                    if self._dim is None:
                        self._dim = self._index.d
                except Exception:
                    # Bad/corrupt index -> start empty and rebuild on first use
                    self._index = None

    def _save_index(self) -> None:
        # Atomic save: write to temp, then replace
        tmp = self.index_path.with_suffix(".index.tmp")
        faiss.write_index(self._index_ensure(), str(tmp))
        tmp.replace(self.index_path)

    # ----------------- core embedding ops -----------------
    def _embed_one(self, text: str) -> np.ndarray:
        # truncate giant inputs to keep embeddings fast/relevant
        text = text[: self._cfg.max_content_chars]
        model = self._model_ensure()
        v = model.encode([text], normalize_embeddings=True)[0]
        return v.astype("float32")

    def _embed_many(self, texts: List[str]) -> np.ndarray:
        model = self._model_ensure()
        texts = [t[: self._cfg.max_content_chars] for t in texts]
        v = model.encode(texts, normalize_embeddings=True)
        return v.astype("float32")

    # ----------------- rebuild logic -----------------
    def _maybe_rebuild_from_disk(self) -> None:
        """
        Rebuild FAISS if index size != row count in JSONL (coarse heuristic).
        """
        with self._lock:
            rows: List[_Row] = read_jsonl(self.mem_path) or []
            row_count = len(rows)
            idx = self._index_ensure()
            if idx.ntotal == row_count:
                return  # already in sync

            # Rebuild from scratch
            dim = self._dim or self._model_ensure().get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)
            if not rows:
                self._save_index()
                return

            # Batch to avoid huge memory spikes
            batch = self._cfg.max_rebuild_batch
            for i in range(0, row_count, batch):
                batch_rows = rows[i : i + batch]
                vecs = self._embed_many([r.get("content", "") for r in batch_rows])
                self._index.add(vecs)
            self._save_index()

    # ----------------- public API -----------------
    def add(self, role: str, content: str, *, identity: Optional[str] = None) -> None:
        """Add a row to memory; dedup very similar content; keep FAISS in sync."""
        role = (role or "").strip()
        content = (content or "").strip()
        if not role or not content:
            return

        row: _Row = {
            "role": role,
            "content": content[: self._cfg.max_content_chars],
            "ts": _utc_iso(),
            "id": _hash_text(f"{role}|{content[:256]}"),
        }
        if identity:
            row[self._cfg.namespace_key] = identity

        with self._lock:
            # Ensure index in a good state
            self._maybe_rebuild_from_disk()

            # Deduplicate if content is long enough to compare
            if len(content) >= self._cfg.dedup_min_chars and self._index_ensure().ntotal > 0:
                q = self._embed_one(content)
                D, I = self._index.search(q[None, :], k=min(8, self._index.ntotal))
                # If we have an almost-identical vector, skip adding
                if float(D[0, 0]) >= self._cfg.dedup_sim_threshold:
                    return

            # Append to JSONL
            append_jsonl(self.mem_path, row)  # your existing helper

            # Add to FAISS
            vec = self._embed_one(content)
            self._index_ensure().add(_normalize(vec))

            # Persist index periodically (or every time by default)
            self._add_counter += 1
            if (self._add_counter % max(1, self._cfg.save_every_adds)) == 0:
                self._save_index()

    def search(self, query: str, k: int, *, identity: Optional[str] = None) -> List[Message]:
        """
        Vector search with MMR re-ranking for diversity.
        - If `identity` provided, results are filtered to that namespace.
        - Returns up to `k` Message dicts (compatible with your app).
        """
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            self._maybe_rebuild_from_disk()
            rows: List[_Row] = read_jsonl(self.mem_path) or []
            if not rows or self._index_ensure().ntotal == 0:
                return []

            # Pre-filter indices by identity if present (but still search globally for speed)
            # We'll filter after retrieval to keep FAISS simple.
            q_vec = self._embed_one(query)
            candidate_k = min(max(k * 3, self._cfg.mmr_candidates), max(1, self._index.ntotal))
            D, I = self._index.search(_normalize(q_vec[None, :]), candidate_k)

            # Gather candidate rows with optional namespace filter
            cand: List[Tuple[int, float]] = []
            for idx, score in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx >= len(rows):
                    continue
                r = rows[idx]
                if identity and r.get(self._cfg.namespace_key) not in (None, identity):
                    continue
                cand.append((idx, float(score)))

            if not cand:
                return []

            # Prepare vectors for MMR
            # (re-embed candidate contents in a small batch to get exact normalized vectors)
            cand_indices = [c[0] for c in cand]
            cand_texts = [rows[i].get("content", "") for i in cand_indices]
            cand_vecs = _normalize(self._embed_many(cand_texts))
            q = _normalize(q_vec[None, :])[0]

            selected = self._mmr_select(q, cand_vecs, k=min(k, len(cand_indices)), lambda_=self._cfg.mmr_lambda)
            out: List[Message] = []
            for pos in selected:
                row = rows[cand_indices[pos]]
                out.append({"role": row.get("role", "user"), "content": row.get("content", "")})  # type: ignore
            return out

    # ----------------- MMR (diversity) -----------------
    @staticmethod
    def _mmr_select(q: np.ndarray, X: np.ndarray, k: int, lambda_: float = 0.5) -> List[int]:
        """
        Maximal Marginal Relevance selection on normalized vectors.
        q: (d,) query vector (normalized)
        X: (n,d) candidate vectors (normalized)
        returns: indices of selected rows in X
        """
        n = X.shape[0]
        if n == 0 or k <= 0:
            return []

        # Relevance scores (cosine since normalized)
        rel = X @ q  # (n,)

        selected: List[int] = []
        remaining = set(range(n))

        # precompute pairwise sims for novelty term
        simXX = X @ X.T  # (n,n)

        for _ in range(min(k, n)):
            if not remaining:
                break
            if not selected:
                # pick best relevance first
                best = max(remaining, key=lambda i: float(rel[i]))
                selected.append(best)
                remaining.remove(best)
                continue

            def score(i: int) -> float:
                # novelty = min similarity to any selected (lower is better)
                max_sim = max(float(simXX[i, j]) for j in selected)
                return lambda_ * float(rel[i]) - (1.0 - lambda_) * max_sim

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)

        return selected