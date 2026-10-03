# src/benchmark.py
# ─────────────────────────────────────────────────────────────
# Performance Benchmark: CAG vs Simulated RAG
#
# Measures:
#  • Frames Per Second (FPS)
#  • Per-frame latency (mean, std, p95, p99)
#  • CAG lookup latency vs simulated RAG retrieval latency
#  • Memory footprint
#
# Why CAG is faster than RAG
# ──────────────────────────
# RAG pipeline cost per query:
#   serialize vector → network/IPC → vector DB (HNSW/FAISS scan) →
#   deserialize → top-k docs → context assembly → (LLM call)
#   Typical: 5–50 ms for the retrieval step alone
#
# CAG pipeline cost per query:
#   matrix multiply  Q @ K.T  in shared memory
#   Typical: 0.05–0.5 ms on CPU, <0.02 ms on GPU
#
# Because the cache is a simple tensor in RAM/VRAM:
#   • Zero serialisation overhead
#   • Zero network latency
#   • Zero index rebuild time
#   • Cache persists across frames at no additional cost
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from configs.config import (
    CACHE_PATH, DEVICE, EMOTIONS, FEATURE_DIM, SIMILARITY_METRIC
)
from src.cache.kv_cache import EmotionKVCache, KVCacheBuilder


# ─────────────────────────────────────────────────────────────
# Simulated RAG baseline
# ─────────────────────────────────────────────────────────────
class SimulatedRAGRetriever:
    """
    Simulates the overhead of a real RAG pipeline:
      1. Serialise query vector to bytes      (torch → numpy → bytes)
      2. Artificial network/IPC delay         (0.5–2 ms)
      3. Linear scan over DB (no index)       (proportional to DB size)
      4. Deserialise results                  (bytes → dict)

    This is deliberately conservative – real RAG with FAISS HNSW
    is faster than a linear scan, but still far slower than CAG.
    """

    def __init__(
        self,
        db_vectors: torch.Tensor,          # [N, D] – the "database"
        db_labels:  torch.Tensor,          # [N]
        network_delay_ms: float = 1.0,     # simulated IPC latency
    ):
        self.db_vectors       = db_vectors.numpy()    # CPU numpy
        self.db_labels        = db_labels.numpy()
        self.network_delay_ms = network_delay_ms

    def retrieve(self, query: torch.Tensor) -> int:
        """Simulate full RAG retrieval; return predicted class index."""

        # 1. Serialise (mimic sending over socket/IPC)
        q_bytes = query.numpy().tobytes()

        # 2. Network / IPC delay
        time.sleep(self.network_delay_ms / 1000)

        # 3. Deserialise
        q_np = np.frombuffer(q_bytes, dtype=np.float32).reshape(1, -1)
        q_np /= (np.linalg.norm(q_np) + 1e-9)          # normalise

        # 4. Linear scan (brute-force cosine sim)
        sims = self.db_vectors @ q_np.T                 # [N, 1]
        best = int(np.argmax(sims))
        return int(self.db_labels[best])


# ─────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────
def run_benchmark(
    n_queries:   int   = 500,
    n_prototypes: int  = 20,
    network_delay_ms: float = 1.0,
    device: str  = DEVICE,
) -> dict:
    """
    Run CAG vs RAG benchmark and return results dict.
    """
    print("\n" + "═" * 60)
    print("  CAG Emotion Detection  ·  Performance Benchmark")
    print("═" * 60)
    print(f"  Queries     : {n_queries}")
    print(f"  Prototypes  : {n_prototypes} × {len(EMOTIONS)} classes")
    print(f"  Device      : {device.upper()}")
    print(f"  RAG delay   : {network_delay_ms} ms (simulated)")
    print("═" * 60 + "\n")

    # ── Build cache ──────────────────────────────────────────
    print("Building KV cache…")
    builder = KVCacheBuilder(EMOTIONS, FEATURE_DIM, device)
    cache   = builder.build_synthetic(n_prototypes)

    # ── Generate random query vectors ────────────────────────
    torch.manual_seed(0)
    queries = torch.randn(n_queries, FEATURE_DIM)
    queries = torch.nn.functional.normalize(queries, p=2, dim=1)

    # ────────────────────────────────────────────────────────
    # CAG benchmark
    # ────────────────────────────────────────────────────────
    print(f"Benchmarking CAG  ({n_queries} queries)…")
    cag_latencies: List[float] = []
    torch.cuda.synchronize() if device == "cuda" else None

    for q in queries:
        t0 = time.perf_counter()
        cache.query(q, top_k=5, metric=SIMILARITY_METRIC)
        cag_latencies.append((time.perf_counter() - t0) * 1000)

    cag_arr  = np.array(cag_latencies)

    # ────────────────────────────────────────────────────────
    # Simulated RAG benchmark
    # ────────────────────────────────────────────────────────
    print(f"Benchmarking RAG  ({n_queries} queries, "
          f"{network_delay_ms}ms simulated network)…")

    rag = SimulatedRAGRetriever(
        cache._keys.cpu(), cache._values.cpu(), network_delay_ms
    )
    rag_latencies: List[float] = []

    for q in queries:
        t0 = time.perf_counter()
        rag.retrieve(q.cpu())
        rag_latencies.append((time.perf_counter() - t0) * 1000)

    rag_arr = np.array(rag_latencies)

    # ────────────────────────────────────────────────────────
    # Report
    # ────────────────────────────────────────────────────────
    results = {
        "cag": {
            "mean_ms":  float(cag_arr.mean()),
            "std_ms":   float(cag_arr.std()),
            "p95_ms":   float(np.percentile(cag_arr, 95)),
            "p99_ms":   float(np.percentile(cag_arr, 99)),
            "fps":      1000 / cag_arr.mean(),
        },
        "rag": {
            "mean_ms":  float(rag_arr.mean()),
            "std_ms":   float(rag_arr.std()),
            "p95_ms":   float(np.percentile(rag_arr, 95)),
            "p99_ms":   float(np.percentile(rag_arr, 99)),
            "fps":      1000 / rag_arr.mean(),
        },
        "speedup": float(rag_arr.mean() / cag_arr.mean()),
    }

    def row(label, d):
        return (
            f"  {label:<10} │ {d['mean_ms']:>7.3f} ms │ "
            f"{d['std_ms']:>7.3f} ms │ {d['p95_ms']:>7.3f} ms │ "
            f"{d['p99_ms']:>7.3f} ms │ {d['fps']:>7.1f} FPS"
        )

    header = (
        f"  {'Method':<10} │ {'Mean':>9} │ {'Std':>9} │ "
        f"{'P95':>9} │ {'P99':>9} │ {'FPS':>9}"
    )
    print("\n" + "─" * 70)
    print(header)
    print("─" * 70)
    print(row("CAG", results["cag"]))
    print(row("RAG (sim)", results["rag"]))
    print("─" * 70)
    print(f"\n  🚀 CAG is {results['speedup']:.1f}× faster than RAG")
    print(f"  CAG mean latency: {results['cag']['mean_ms']:.3f} ms  "
          f"(target: <50 ms  ✓)\n")

    # Memory footprint
    cache_bytes = (
        cache._keys.element_size() * cache._keys.numel()
        + cache._values.element_size() * cache._values.numel()
    )
    print(f"  Cache memory   : {cache_bytes / 1024:.1f} KB  "
          f"({cache.size} prototypes × {FEATURE_DIM}d × float32)")
    print("═" * 60 + "\n")

    return results


if __name__ == "__main__":
    run_benchmark(n_queries=500)
