def test_anchors_present():
    import yaml
    with open("src/identity/anchors.yaml", "r", encoding="utf-8") as f:
        anchors = yaml.safe_load(f)
    assert anchors.get("system_name") == "Ember"
    assert len(anchors.get("anchors", [])) >= 3


def test_anchor_manager(tmp_path):
    from pathlib import Path
    import sys
    import yaml

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.identity.manager import AnchorManager

    src = Path("src/identity/anchors.yaml")
    tmp_file = tmp_path / "anchors.yaml"
    tmp_file.write_text(src.read_text(), encoding="utf-8")

    mgr = AnchorManager(tmp_file, anchors_per_query=2)
    first = mgr.get_for_prompt()
    second = mgr.get_for_prompt()
    assert first != second  # rotation occurs

    mgr.add_anchor("Testing dynamic anchor")
    data = yaml.safe_load(tmp_file.read_text())
    assert "Testing dynamic anchor" in data["anchors"]

    mgr_session = AnchorManager(tmp_file, injection_mode="session_start")
    assert mgr_session.get_for_prompt()  # first call returns anchors
    assert mgr_session.get_for_prompt() == []  # second call none
