from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Anchor:
    """
    An identity anchor that can be injected into the system prompt.

    Fields:
        text: The actual anchor sentence.
        tags: Optional tags/categories (e.g., ["core", "ethical"]).
        weight: Relative weight used in weighted sampling (>= 0).
        always_on: If True, this anchor is injected every time (subject to mode).
        priority: Higher priority anchors are preferred when ties occur.
        disabled: If True, anchor is ignored.
    """
    text: str
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    always_on: bool = False
    priority: int = 0
    disabled: bool = False

    # Helper to serialize to YAML-friendly dict
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": self.text,
        }
        if self.tags:
            d["tags"] = self.tags
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.always_on:
            d["always_on"] = True
        if self.priority != 0:
            d["priority"] = self.priority
        if self.disabled:
            d["disabled"] = True
        return d


@dataclass
class Policy:
    """
    Injection policy. Mirrors YAML keys if present.

    Fields:
        mode: "every_query" | "session_start" | "none"
        anchors_per_query: how many rotating anchors to include
        strategy: "round_robin" | "random" | "weighted"
        dedupe: if True, remove duplicate texts before selection
        shuffle_on_load: if True, shuffle anchors once at load
        seed: optional randomness seed (for reproducible selection)
    """
    mode: str = "every_query"
    anchors_per_query: int = 2
    strategy: str = "round_robin"
    dedupe: bool = True
    shuffle_on_load: bool = False
    seed: Optional[int] = None


# -----------------------------
# Anchor Manager
# -----------------------------

class AnchorManager:
    """
    Manage identity anchors with rotation, filtering, and persistence.

    YAML compatibility notes:
        - Legacy format with anchors: [ "text1", "text2" ] is supported.
        - Rich format with anchors: [{text, tags, weight, always_on, priority, disabled}, ...] also supported.
        - Optional keys:
            system_name: str
            system_prompt: str (optional wrapper prompt)
            policy:
              mode: "every_query" | "session_start" | "none"
              anchors_per_query: int
              strategy: "round_robin" | "random" | "weighted"
              dedupe: bool
              shuffle_on_load: bool
              seed: int|null
    """

    def __init__(
        self,
        path: Union[str, Path],
        anchors_per_query: Optional[int] = None,
        injection_mode: str = "every_query",
        strategy: str = "round_robin",
        seed: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._cursor = 0
        self._injected_once = False

        # in-memory
        self._system_name: str = "Assistant"
        self._system_prompt: Optional[str] = None
        self._policy = Policy()
        self._anchors: List[Anchor] = []

        # load & normalize
        self._load()

        # allow constructor overrides for backward compat
        if anchors_per_query is not None:
            self._policy.anchors_per_query = max(0, int(anchors_per_query))
        if injection_mode:
            self._policy.mode = injection_mode
        if strategy:
            self._policy.strategy = strategy
        if seed is not None:
            self._policy.seed = seed

        # optional shuffle at load (stable with seed)
        if self._policy.shuffle_on_load:
            self._rng.seed(self._policy.seed)
            self._rng.shuffle(self._anchors)

    # ---------- properties ----------

    @property
    def system_name(self) -> str:
        return self._system_name

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def anchors(self) -> List[Anchor]:
        return self._anchors

    # ---------- core methods ----------

    def list_anchors(self, include_disabled: bool = False) -> List[str]:
        with self._lock:
            return [a.text for a in self._anchors if include_disabled or not a.disabled]

    def add_anchor(
        self,
        text: str,
        *,
        tags: Optional[Iterable[str]] = None,
        weight: float = 1.0,
        always_on: bool = False,
        priority: int = 0,
        disabled: bool = False,
        save: bool = True,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            # prevent duplicate texts when dedupe is on
            if self._policy.dedupe and any(a.text == text for a in self._anchors):
                return
            anchor = Anchor(
                text=text,
                tags=list(tags) if tags else [],
                weight=max(0.0, float(weight)),
                always_on=bool(always_on),
                priority=int(priority),
                disabled=bool(disabled),
            )
            self._anchors.append(anchor)
            if save:
                self._save()

    def remove_anchor(self, text: str, *, save: bool = True) -> bool:
        with self._lock:
            before = len(self._anchors)
            self._anchors = [a for a in self._anchors if a.text != text]
            changed = len(self._anchors) != before
            if changed and save:
                self._save()
            return changed

    def update_anchor(
        self,
        old_text: str,
        *,
        new_text: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        weight: Optional[float] = None,
        always_on: Optional[bool] = None,
        priority: Optional[int] = None,
        disabled: Optional[bool] = None,
        save: bool = True,
    ) -> bool:
        with self._lock:
            for a in self._anchors:
                if a.text == old_text:
                    if new_text is not None:
                        a.text = new_text.strip()
                    if tags is not None:
                        a.tags = list(tags)
                    if weight is not None:
                        a.weight = max(0.0, float(weight))
                    if always_on is not None:
                        a.always_on = bool(always_on)
                    if priority is not None:
                        a.priority = int(priority)
                    if disabled is not None:
                        a.disabled = bool(disabled)
                    if save:
                        self._save()
                    return True
            return False

    def search(self, *, tag: Optional[str] = None, text_contains: Optional[str] = None) -> List[Anchor]:
        """Return anchors filtered by tag and/or substring."""
        with self._lock:
            out = [a for a in self._anchors if not a.disabled]
            if tag:
                out = [a for a in out if tag in a.tags]
            if text_contains:
                s = text_contains.lower()
                out = [a for a in out if s in a.text.lower()]
            return out

    def clear_session(self) -> None:
        """Reset session-scoped state (e.g., for a new conversation)."""
        with self._lock:
            self._injected_once = False
            self._cursor = 0

    def get_for_prompt(
        self,
        *,
        n_override: Optional[int] = None,
        required_tags: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """
        Select anchors for the current prompt according to policy.
        - Adds all always_on anchors (if enabled and not disabled).
        - Adds rotating anchors by strategy (round_robin / random / weighted).
        - Respects injection mode: every_query / session_start / none.
        - required_tags: if provided, only anchors with all of these tags are considered for rotation.
        """
        with self._lock:
            # Handle injection mode
            if self._policy.mode == "none":
                return []

            if self._policy.mode == "session_start" and self._injected_once:
                return []

            # Collect candidates
            active = [a for a in self._anchors if not a.disabled]
            if self._policy.dedupe:
                # dedupe by text, keep highest priority
                seen: Dict[str, Anchor] = {}
                for a in sorted(active, key=lambda x: (-x.priority, x.text)):
                    if a.text not in seen:
                        seen[a.text] = a
                active = list(seen.values())

            # Always-on anchors
            always_on = [a for a in active if a.always_on]

            # Rotating pool (exclude always-on)
            rotating = [a for a in active if not a.always_on]

            # Optional tag filtering
            if required_tags:
                required = set(required_tags)
                rotating = [a for a in rotating if required.issubset(set(a.tags))]

            # How many to select
            k = self._policy.anchors_per_query if n_override is None else max(0, int(n_override))
            selected: List[Anchor] = []

            # Strategy selection
            if k > 0 and rotating:
                if self._policy.strategy == "round_robin":
                    selected = self._round_robin(rotating, k)
                elif self._policy.strategy == "weighted":
                    selected = self._weighted_sample(rotating, k)
                else:  # "random"
                    selected = self._random_sample(rotating, k)

            # Mark session injection if needed
            if self._policy.mode == "session_start":
                self._injected_once = True

            # Compose final list (always_on first by priority, then selected by priority)
            final = sorted(always_on, key=lambda a: (-a.priority, a.text)) + \
                    sorted(selected, key=lambda a: (-a.priority, a.text))
            return [a.text for a in final]

    # ---------- selection strategies ----------

    def _round_robin(self, pool: List[Anchor], k: int) -> List[Anchor]:
        if not pool:
            return []
        # sort by priority desc, then stable round-robin over that order
        ordered = sorted(pool, key=lambda a: (-a.priority, a.text))
        out: List[Anchor] = []
        for i in range(k):
            idx = (self._cursor + i) % len(ordered)
            out.append(ordered[idx])
        self._cursor = (self._cursor + k) % len(ordered)
        return out

    def _random_sample(self, pool: List[Anchor], k: int) -> List[Anchor]:
        k = min(k, len(pool))
        self._rng.seed(self._policy.seed)
        return self._rng.sample(pool, k)

    def _weighted_sample(self, pool: List[Anchor], k: int) -> List[Anchor]:
        # Normalize weights; skip zero-weight anchors
        items = [(a, max(0.0, float(a.weight))) for a in pool if a.weight > 0.0]
        if not items:
            return self._random_sample(pool, k)
        anchors, weights = zip(*items)
        total = sum(weights)
        if total <= 0:
            return self._random_sample(pool, k)

        # sample without replacement, proportional to weight (approx by iterative)
        self._rng.seed(self._policy.seed)
        chosen: List[Anchor] = []
        candidates = list(anchors)
        weights_list = list(weights)
        k = min(k, len(candidates))
        for _ in range(k):
            total_w = sum(weights_list)
            if total_w <= 0:
                break
            r = self._rng.random() * total_w
            acc = 0.0
            pick = 0
            for i, w in enumerate(weights_list):
                acc += w
                if acc >= r:
                    pick = i
                    break
            chosen.append(candidates.pop(pick))
            weights_list.pop(pick)
        return chosen

    # ---------- persistence & schema ----------

    @property
    def _rng(self) -> random.Random:
        # separate instance to avoid interfering with global RNG
        if not hasattr(self, "__rng"):
            setattr(self, "__rng", random.Random())
        return getattr(self, "__rng")

    def _load(self) -> None:
        """Load YAML file, normalize schema, and validate."""
        if not self.path.exists():
            # initialize empty file
            self._system_name = "Assistant"
            self._anchors = []
            self._policy = Policy()
            self._save()  # create on disk
            return

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # System name / prompt
        self._system_name = data.get("system_name", "Assistant")
        self._system_prompt = data.get("system_prompt")

        # Policy (optional)
        pol = data.get("policy", {})
        self._policy = Policy(
            mode=pol.get("mode", "every_query"),
            anchors_per_query=int(pol.get("anchors_per_query", 2)),
            strategy=pol.get("strategy", "round_robin"),
            dedupe=bool(pol.get("dedupe", True)),
            shuffle_on_load=bool(pol.get("shuffle_on_load", False)),
            seed=pol.get("seed"),
        )

        # Anchors: support legacy list[str] or list[dict]
        raw_anchors = data.get("anchors", [])
        self._anchors = self._normalize_anchors(raw_anchors)

        # Basic validation
        self._validate()

    def _save(self) -> None:
        """Persist current state to YAML (pretty & stable)."""
        payload: Dict[str, Any] = {
            "system_name": self._system_name,
            "anchors": [a.to_dict() for a in self._anchors],
            "policy": {
                "mode": self._policy.mode,
                "anchors_per_query": self._policy.anchors_per_query,
                "strategy": self._policy.strategy,
                "dedupe": self._policy.dedupe,
                "shuffle_on_load": self._policy.shuffle_on_load,
                "seed": self._policy.seed,
            },
        }
        if self._system_prompt:
            payload["system_prompt"] = self._system_prompt

        tmp = self.path.with_suffix(".tmp.yaml")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        tmp.replace(self.path)

    @staticmethod
    def _normalize_anchors(raw: Any) -> List[Anchor]:
        out: List[Anchor] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    out.append(Anchor(text=item.strip()))
                elif isinstance(item, dict):
                    text = (item.get("text") or "").strip()
                    if not text:
                        continue
                    out.append(
                        Anchor(
                            text=text,
                            tags=list(item.get("tags", [])) if item.get("tags") else [],
                            weight=float(item.get("weight", 1.0)),
                            always_on=bool(item.get("always_on", False)),
                            priority=int(item.get("priority", 0)),
                            disabled=bool(item.get("disabled", False)),
                        )
                    )
        return out

    def _validate(self) -> None:
        # sanitize policy
        allowed_modes = {"every_query", "session_start", "none"}
        allowed_strategies = {"round_robin", "random", "weighted"}

        if self._policy.mode not in allowed_modes:
            self._policy.mode = "every_query"
        if self._policy.strategy not in allowed_strategies:
            self._policy.strategy = "round_robin"
        self._policy.anchors_per_query = max(0, int(self._policy.anchors_per_query))
        if self._policy.seed is not None:
            try:
                self._policy.seed = int(self._policy.seed)
            except Exception:
                self._policy.seed = None

        # sanitize anchors
        dedup: Dict[str, Anchor] = {}
        for a in self._anchors:
            a.text = a.text.strip()
            if not a.text:
                continue
            a.weight = max(0.0, float(a.weight))
            a.priority = int(a.priority)
            a.tags = [str(t) for t in a.tags if str(t).strip()]
            if self._policy.dedupe:
                if a.text in dedup:
                    # keep the higher priority / non-disabled version
                    prev = dedup[a.text]
                    keep = a if (a.priority, not a.disabled) > (prev.priority, not prev.disabled) else prev
                    dedup[a.text] = keep
                else:
                    dedup[a.text] = a
            else:
                # allow duplicates if dedupe disabled
                dedup[f"{a.text}::{len(dedup)}"] = a
        self._anchors = list(dedup.values())

    # ---------- convenience setters ----------

    def set_system_name(self, name: str, *, save: bool = True) -> None:
        with self._lock:
            self._system_name = (name or "Assistant").strip() or "Assistant"
            if save:
                self._save()

    def set_system_prompt(self, prompt: Optional[str], *, save: bool = True) -> None:
        with self._lock:
            self._system_prompt = (prompt.strip() if prompt else None)
            if save:
                self._save()

    def set_policy(
        self,
        *,
        mode: Optional[str] = None,
        anchors_per_query: Optional[int] = None,
        strategy: Optional[str] = None,
        dedupe: Optional[bool] = None,
        shuffle_on_load: Optional[bool] = None,
        seed: Optional[int] = None,
        save: bool = True,
    ) -> None:
        with self._lock:
            if mode is not None:
                self._policy.mode = mode
            if anchors_per_query is not None:
                self._policy.anchors_per_query = max(0, int(anchors_per_query))
            if strategy is not None:
                self._policy.strategy = strategy
            if dedupe is not None:
                self._policy.dedupe = bool(dedupe)
            if shuffle_on_load is not None:
                self._policy.shuffle_on_load = bool(shuffle_on_load)
            if seed is not None:
                self._policy.seed = int(seed)
            if save:
                self._save()