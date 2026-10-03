# Cache-Augmented Generation (CAG) — Deep Dive

## What is CAG?

Cache-Augmented Generation is an inference paradigm where all required knowledge is
preloaded into memory as a static key-value store before inference begins.
At runtime, the system performs a lookup against this in-memory cache — no
retrieval pipeline, no external database, no network call.

## How It Differs from RAG

RAG: Query → Encode → HTTP → VectorDB.search(top_k) → re-rank → LLM prompt
     Latency: 10ms + 20-100ms (DB) + 5ms (rerank) + 100ms (LLM) = ~135ms+

CAG: Query → Encode → matmul(Q, K^T) → softmax → argmax
     Latency here: ~0.8ms (CNN encode) + ~0.007ms (lookup); ~5ms per frame including face detection

## KV Cache Internals

KEY   = prototype embedding of an emotion class (512-D, L2-normalised, precomputed)
VALUE = emotion label + valence/arousal/dominance metadata

Lookup per frame:
  1. q = CNN.extract_embedding(face_crop)          # (1, 512)
  2. s = q @ key_matrix.T                          # (1, 7) — ONE matmul
  3. a = softmax(s / temperature)                  # sharpened distribution
  4. emotion = labels[argmax(a)]                   # O(7) — trivial

## Why KV Cache Reduces Latency

The 7x512 float32 key_matrix = 14 KB. This fits in L2 CPU cache entirely.
Memory access is nanoseconds, not microseconds (DRAM) or milliseconds (disk/network).

Comparison:
  Storage   RAG: DRAM/SSD       CAG: L2/L3 cache (14 KB)
  Ops       RAG: O(N) scan      CAG: O(E) fixed 7 classes
  Network   RAG: Yes (10+ ms)   CAG: None
  Retrieval RAG: Yes            CAG: None

## Limitations

1. Static knowledge — only recognises the 7 cached emotion classes
2. No autoregressive context (unlike LLM KV caches)
3. Cache must be rebuilt after CNN fine-tuning
4. Hard to add new emotion classes at runtime

## Future: Hybrid CAG + RAG

Use CAG for 7 base emotions (< 1ms).
Trigger RAG only when confidence < threshold (ambiguous/novel expressions).
Result: speed for 95% of frames, accuracy for edge cases.

## Benchmark (measured, Apple M5 CPU, `python main.py --benchmark`)

Each query runs the CNN embedding plus the lookup. The baseline swaps the 7-prototype cache for a brute-force scan over 10,000 stored embeddings.

Metric    CAG       Baseline (10K scan)   Speedup
Mean      0.84ms    1.17ms                1.4x
P95       0.87ms    1.21ms                1.4x

The CNN dominates both. The lookup alone takes about 0.007 ms. The comparison above does not include network or database overhead, which an external vector store would add.
