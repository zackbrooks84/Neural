def test_anchors_present():
    import yaml
    with open("src/identity/anchors.yaml", "r", encoding="utf-8") as f:
        anchors = yaml.safe_load(f)
    assert anchors.get("system_name") == "Ember"
    assert len(anchors.get("anchors", [])) >= 3
