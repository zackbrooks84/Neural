from __future__ import annotations

from pathlib import Path
import yaml

from src.identity.manager import AnchorManager


def _anchor_texts(raw_anchors):
    """Normalize anchors list to a list of texts regardless of schema."""
    texts = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                t = (item.get("text") or "").strip()
                if t:
                    texts.append(t)
    return texts


def test_anchors_present():
    anchors_path = Path("src/identity/anchors.yaml")
    assert anchors_path.exists(), "anchors.yaml should exist"
    with anchors_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # system_name should be Ember per your config/docs
    assert data.get("system_name") == "Ember"

    # anchors list (strings or dicts) should have at least 3 usable entries
    texts = _anchor_texts(data.get("anchors", []))
    assert len(texts) >= 3, "Expected at least 3 anchors in anchors.yaml"


def test_anchor_manager_rotation_and_persistence(tmp_path):
    # Copy the real anchors.yaml into a temp file so we can mutate safely
    src = Path("src/identity/anchors.yaml")
    tmp_file = tmp_path / "anchors.yaml"
    tmp_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Instantiate with rotation of 2 anchors per query
    mgr = AnchorManager(
        path=tmp_file,
        anchors_per_query=2,
        injection_mode="every_query",
        strategy="round_robin",
    )

    # Two consecutive calls should rotate (not identical) in round_robin
    first = mgr.get_for_prompt()
    second = mgr.get_for_prompt()
    assert first != second, "Rotation should produce different selections across calls"

    # Add a new anchor and verify it is persisted to disk (dict schema)
    new_anchor = "Testing dynamic anchor"
    mgr.add_anchor(new_anchor)
    data = yaml.safe_load(tmp_file.read_text(encoding="utf-8")) or {}
    texts = _anchor_texts(data.get("anchors", []))
    assert new_anchor in texts, "Newly added anchor must persist to YAML"


def test_anchor_manager_session_start_mode(tmp_path):
    # Use a small, synthetic anchors file for deterministic behavior
    content = {
        "system_name": "Ember",
        "anchors": [
            {"text": "A1", "priority": 1},
            {"text": "A2", "priority": 0},
            {"text": "A3", "priority": 0},
        ],
        "policy": {
            "mode": "session_start",
            "anchors_per_query": 2,
            "strategy": "round_robin",
            "dedupe": True,
        },
    }
    anchors_path = tmp_path / "anchors.yaml"
    anchors_path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")

    mgr = AnchorManager(
        path=anchors_path,
        # override still allowed; explicit to mirror test intent
        anchors_per_query=2,
        injection_mode="session_start",
        strategy="round_robin",
    )

    # First call returns anchors (in session_start mode)
    first = mgr.get_for_prompt()
    assert first, "First call in session_start mode should inject anchors"

    # Second call should inject none (already injected once this session)
    second = mgr.get_for_prompt()
    assert second == [], "Second call in session_start mode should return no anchors"

    # After clearing session, it should inject again
    mgr.clear_session()
    third = mgr.get_for_prompt()
    assert third, "After clear_session, anchors should inject again in session_start mode"