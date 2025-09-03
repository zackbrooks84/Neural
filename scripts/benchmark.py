#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ember / Neural — LLM Engine micro-benchmark

Features
- Warmup iterations (excluded from stats)
- Multiple timed runs with percentile summary (p50/p90/p95/p99)
- Optional concurrent runs (simple thread fan-out)
- Tracks:
  - wall time per request
  - chars/s and ~tokens/s (naive token proxy)
  - peak Python heap via tracemalloc
  - RSS (process resident set) if psutil is available
  - CUDA VRAM (if torch + GPU available)
- Saves results to CSV and/or JSON
- Friendly CLI with defaults that "just work"

Usage
-----
python scripts/benchmark.py --config config.yaml --repeat 10 --warmup 2
python scripts/benchmark.py --repeat 20 --concurrency 4 --csv bench.csv
python scripts/benchmark.py --prompt "Tell me a story about Ember." --json bench.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is on path for direct execution
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# Local engine
from llm.engine import LLMEngine  # type: ignore

# Optional deps
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def now() -> float:
    return time.perf_counter()


def approx_token_count(text: str) -> int:
    """
    Cheap token proxy: whitespace-split words * 1.3 (English-ish).
    Replace with a real tokenizer if desired.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def get_process_mem_mb() -> float:
    if psutil is None:
        return float("nan")
    try:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def get_cuda_mem_mb() -> Tuple[float, float]:
    """(allocated_MB, reserved_MB) if torch+CUDA, else (nan, nan)."""
    if torch is None or not torch.cuda.is_available():
        return (float("nan"), float("nan"))
    try:
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return (alloc, reserved)
    except Exception:
        return (float("nan"), float("nan"))


def percentiles(values: List[float], ps: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    vs = sorted(values)
    out = {}
    for p in ps:
        k = max(0, min(len(vs) - 1, int(round((len(vs) - 1) * p))))
        out[f"p{int(p*100)}"] = vs[k]
    return out


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Sample:
    idx: int
    duration_s: float
    reply_chars: int
    reply_tokens_est: int
    chars_per_s: float
    tokens_per_s: float
    heap_peak_mb: float
    rss_mb: float
    cuda_alloc_mb: float
    cuda_reserved_mb: float
    error: Optional[str] = None

    @staticmethod
    def from_result(idx: int, start: float, reply: Optional[str], peak_bytes: int, error: Optional[str]) -> "Sample":
        dur = now() - start
        text = reply or ""
        rc = len(text)
        tk = approx_token_count(text)
        return Sample(
            idx=idx,
            duration_s=dur,
            reply_chars=rc,
            reply_tokens_est=tk,
            chars_per_s=(rc / dur) if dur > 0 else 0.0,
            tokens_per_s=(tk / dur) if dur > 0 else 0.0,
            heap_peak_mb=peak_bytes / 1_000_000,
            rss_mb=get_process_mem_mb(),
            cuda_alloc_mb=get_cuda_mem_mb()[0],
            cuda_reserved_mb=get_cuda_mem_mb()[1],
            error=error,
        )


# -----------------------------
# Runner
# -----------------------------
def run_once(engine: LLMEngine, system_prompt: str, user_prompt: str, idx: int) -> Sample:
    tracemalloc.start()
    heap_before, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    start = now()
    reply: Optional[str] = None
    err: Optional[str] = None
    try:
        reply = engine.chat(system_prompt, user_prompt)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return Sample.from_result(idx, start, reply, peak, err)


def fan_out(
    engine_factory,
    system_prompt: str,
    user_prompt: str,
    n: int,
    threads: int,
) -> List[Sample]:
    """
    Simple threaded fan-out. If LLMEngine is NOT thread-safe,
    pass a factory that returns a NEW engine per thread.
    """
    results: List[Sample] = []
    lock = threading.Lock()
    counter = {"i": 0}

    def worker(tid: int):
        # Either share engine or create per-thread engine
        local_engine = engine_factory() if engine_factory.__name__ != "<lambda>" else engine_factory()
        while True:
            with lock:
                i = counter["i"]
                if i >= n:
                    return
                counter["i"] += 1
            s = run_once(local_engine, system_prompt, user_prompt, i)
            with lock:
                results.append(s)

    threads_list: List[threading.Thread] = []
    for t in range(threads):
        th = threading.Thread(target=worker, args=(t,), daemon=True)
        th.start()
        threads_list.append(th)
    for th in threads_list:
        th.join()

    # sort by idx
    results.sort(key=lambda s: s.idx)
    return results


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember / Neural — LLM micro-benchmark")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config for LLMEngine")
    p.add_argument("--repeat", type=int, default=10, help="Number of measured runs")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs (excluded from stats)")
    p.add_argument("--concurrency", type=int, default=1, help="Thread count (1 = sequential)")
    p.add_argument("--system", type=str, default="You are Ember, a concise and helpful assistant.", help="System prompt")
    p.add_argument("--prompt", type=str, default="Briefly introduce yourself.", help="User prompt")
    p.add_argument("--prompt-file", type=str, default=None, help="File to read the user prompt from")
    p.add_argument("--csv", type=str, default=None, help="Write per-sample rows to CSV")
    p.add_argument("--json", type=str, default=None, help="Write summary + samples to JSON")
    p.add_argument("--print-replies", action="store_true", help="Print model replies inline")
    p.add_argument("--engine-per-thread", action="store_true", help="Construct a new engine per thread (safer)")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    # Load prompt from file if provided
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            print(f"[error] prompt-file not found: {path}")
            sys.exit(2)
        args.prompt = path.read_text(encoding="utf-8")

    # Engine factory
    def make_engine() -> LLMEngine:
        return LLMEngine(args.config)

    # Initialize a primary engine once (for warmups, sequential, or as template)
    try:
        primary_engine = make_engine()
    except Exception as e:
        print(f"[fatal] LLM initialization failed: {e}")
        sys.exit(1)

    # Warmup
    if args.warmup > 0:
        print(f"[warmup] running {args.warmup} iterations (excluded from stats)…")
        for i in range(args.warmup):
            s = run_once(primary_engine, args.system, args.prompt, i)
            if args.print_replies and not s.error:
                print(f"[warmup #{i}] {s.reply_chars} chars in {s.duration_s:.2f}s")
        print("[warmup] done.\n")

    # Measured runs
    n = max(1, args.repeat)
    print(f"[run] repeat={n} concurrency={args.concurrency} engine_per_thread={args.engine_per_thread}")

    if args.concurrency <= 1:
        # sequential, reuse primary engine
        samples: List[Sample] = []
        for i in range(n):
            s = run_once(primary_engine, args.system, args.prompt, i)
            samples.append(s)
            label = "ok" if not s.error else f"ERR: {s.error}"
            print(f"[{i+1:03d}/{n}] {label} | {s.duration_s:.3f}s | ~{s.tokens_per_s:.1f} tok/s | {s.heap_peak_mb:.1f}MB heap | {s.rss_mb:.1f}MB rss")
            if args.print_replies and not s.error:
                print(f"  └─ reply[{s.reply_chars} chars]:\n{('-'*60)}\n{('' if s.reply_chars < 2000 else '[truncated] ')}")
        results = samples
    else:
        # concurrent
        if args.engine_per_thread:
            engine_factory = make_engine
        else:
            # share one engine across threads — only if thread-safe
            shared = primary_engine
            engine_factory = lambda: shared  # noqa: E731

        results = fan_out(engine_factory, args.system, args.prompt, n, threads=args.concurrency)
        for s in results:
            label = "ok" if not s.error else f"ERR: {s.error}"
            print(f"[{s.idx+1:03d}/{n}] {label} | {s.duration_s:.3f}s | ~{s.tokens_per_s:.1f} tok/s | {s.heap_peak_mb:.1f}MB heap | {s.rss_mb:.1f}MB rss")

    # Summary (exclude errored)
    ok = [s for s in results if not s.error]
    errs = [s for s in results if s.error]
    if not ok:
        print("\n[summary] no successful runs.")
        if errs:
            print(f"[summary] errors: {len(errs)}")
        sys.exit(3)

    durs = [s.duration_s for s in ok]
    tokps = [s.tokens_per_s for s in ok]
    chps = [s.chars_per_s for s in ok]

    summary = {
        "runs": len(results),
        "ok": len(ok),
        "errors": len(errs),
        "duration_avg_s": stats.mean(durs),
        "duration_min_s": min(durs),
        "duration_max_s": max(durs),
        **percentiles(durs, [0.50, 0.90, 0.95, 0.99]),
        "tokens_per_s_avg": stats.mean(tokps),
        "chars_per_s_avg": stats.mean(chps),
        "rss_mb_last": ok[-1].rss_mb,
        "cuda_alloc_mb_last": ok[-1].cuda_alloc_mb,
        "cuda_reserved_mb_last": ok[-1].cuda_reserved_mb,
    }

    print("\n[summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.3f}")
        else:
            print(f"- {k}: {v}")

    # Save CSV/JSON if requested
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(asdict(results[0]).keys()),
            )
            w.writeheader()
            for s in results:
                w.writerow(asdict(s))
        print(f"[write] CSV -> {args.csv}")

    if args.json:
        payload = {
            "summary": summary,
            "samples": [asdict(s) for s in results],
            "config": {
                "repeat": args.repeat,
                "warmup": args.warmup,
                "concurrency": args.concurrency,
                "engine_per_thread": args.engine_per_thread,
                "system": args.system,
                "prompt": (args.prompt[:200] + "…") if len(args.prompt) > 200 else args.prompt,
                "config_path": args.config,
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[write] JSON -> {args.json}")


if __name__ == "__main__":
  main()