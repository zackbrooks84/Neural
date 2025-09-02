from chat_server.memory import DiskMemory

def test_disk_memory_summarises(tmp_path):
    mem = DiskMemory(str(tmp_path), max_messages=5, summary_keep=2)
    identity = "id"
    for i in range(7):
        mem.append(identity, {"user": f"u{i}", "assistant": f"a{i}"})
    history = mem.load(identity)
    # Expect one summary entry plus the last two interactions
    assert len(history) == 3
    assert "summary" in history[0]
    assert history[1]["user"] == "u5"
    assert history[2]["assistant"] == "a6"
