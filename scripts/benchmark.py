"""Simple performance benchmark for the local LLM engine."""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

# Ensure src/ is on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from llm.engine import LLMEngine


def main() -> None:
    """Run a single chat request and report time and memory usage."""
    try:
        engine = LLMEngine("config.yaml")
    except Exception as e:
        print(f"LLM initialization failed: {e}")
        return

    tracemalloc.start()
    start = time.perf_counter()
    reply = engine.chat("Benchmarking run", "Hello")
    duration = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Reply: {reply}")
    print(f"Time: {duration:.2f}s")
    print(f"Peak memory: {peak / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
