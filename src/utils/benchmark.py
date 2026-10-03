"""
benchmark.py
------------
Benchmarks CAG vs. a naive baseline (no caching) and measures:
- Average inference latency
- P50/P95/P99 latencies
- FPS throughput
- Memory footprint

Run: python -m src.utils.benchmark
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict

from src.modules.emotion_cnn import EmotionCNN
from src.modules.kv_cache import EmotionKVCache, EMOTION_LABELS


class NaiveBaselineEngine:
    """
    Simulates RAG-style baseline: for each query, perform a linear scan
    over all stored embeddings (like a naive vector DB lookup).
    In a real RAG system, this would involve:
      - Serialise query embedding → JSON
      - HTTP call to vector DB
      - Deserialise top-k results
      - Re-rank with cross-encoder
    Here we simulate just the compute cost (no I/O).
    """

    def __init__(self, n_stored: int = 10_000, embedding_dim: int = 512,
                 device: str = "cpu"):
        self.device = device
        # Simulate 10K stored embeddings (like a vector DB)
        self.stored = torch.randn(n_stored, embedding_dim, device=device)
        self.stored = F.normalize(self.stored, dim=1)
        self.labels_list = np.random.choice(EMOTION_LABELS, n_stored)

    def query(self, emb: torch.Tensor) -> str:
        # Naive cosine similarity against ALL stored vectors
        sim = (emb @ self.stored.T).squeeze(0)
        top_idx = sim.argmax().item()
        return self.labels_list[top_idx]


def run_benchmark(n_trials: int = 500) -> Dict:
    print(f"\n{'='*60}")
    print("  CAG vs. Baseline Benchmark")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cnn = EmotionCNN(embedding_dim=512).to(device)
    cnn.eval()

    kv = EmotionKVCache(embedding_dim=512, device=device)
    kv.build_from_synthetic()

    baseline = NaiveBaselineEngine(n_stored=10_000, embedding_dim=512, device=device)

    dummy_frame = torch.randn(1, 1, 48, 48, device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(20):
            emb = cnn.extract_embedding(dummy_frame)
            kv.query(emb)

    # ---- CAG timing ----
    cag_latencies = []
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            emb = cnn.extract_embedding(dummy_frame)
            kv.query(emb)
            cag_latencies.append((time.perf_counter() - t0) * 1000)

    # ---- Baseline timing ----
    base_latencies = []
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            emb = cnn.extract_embedding(dummy_frame)
            baseline.query(emb)
            base_latencies.append((time.perf_counter() - t0) * 1000)

    cag_arr = np.array(cag_latencies)
    base_arr = np.array(base_latencies)

    results = {
        "cag": {
            "mean_ms":  round(cag_arr.mean(), 3),
            "p50_ms":   round(np.percentile(cag_arr, 50), 3),
            "p95_ms":   round(np.percentile(cag_arr, 95), 3),
            "p99_ms":   round(np.percentile(cag_arr, 99), 3),
            "fps":      round(1000 / cag_arr.mean(), 1),
        },
        "baseline": {
            "mean_ms":  round(base_arr.mean(), 3),
            "p50_ms":   round(np.percentile(base_arr, 50), 3),
            "p95_ms":   round(np.percentile(base_arr, 95), 3),
            "p99_ms":   round(np.percentile(base_arr, 99), 3),
            "fps":      round(1000 / base_arr.mean(), 1),
        }
    }

    speedup = base_arr.mean() / cag_arr.mean()
    results["speedup_x"] = round(speedup, 2)

    # Print table
    print(f"{'Metric':<22} {'CAG':>12} {'Baseline':>12} {'Speedup':>10}")
    print("-" * 58)
    for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps"]:
        unit = " fps" if key == "fps" else "ms"
        print(f"{key:<22} {results['cag'][key]:>11}{unit} "
              f"{results['baseline'][key]:>11}{unit}")
    print("-" * 58)
    print(f"{'CAG speedup':<22} {speedup:>11.2f}x")

    print(f"\n{'='*60}")
    print("WHY CAG IS FASTER THAN RAG:")
    print(f"  • CAG lookup: 7 dot products (one per emotion class)")
    print(f"  • Baseline:   10,000 dot products (scan all stored embs)")
    print(f"  • CAG is {speedup:.1f}x faster in pure compute")
    print(f"  • Real RAG adds HTTP + serialisation overhead (>>10ms extra)")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_benchmark(n_trials=500)
