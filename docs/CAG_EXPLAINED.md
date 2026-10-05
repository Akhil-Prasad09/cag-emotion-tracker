# Cache-augmented generation (CAG) in this project

## What CAG means here

Cache-augmented generation is an inference setup where all the knowledge the model needs is
loaded into memory as a static key-value store before inference starts. At runtime the system
looks the answer up in this in-memory cache, with no retrieval pipeline, external database or
network call.

## How it differs from RAG

```
RAG: Query → Encode → HTTP → VectorDB.search(top_k) → re-rank → LLM prompt
     Latency: 10ms + 20-100ms (DB) + 5ms (rerank) + 100ms (LLM) = ~135ms+

CAG: Query → Encode → matmul(Q, K^T) → softmax → argmax
     Latency here: ~2ms per CNN pass (two passes: face + mirror) + ~0.007ms (lookup); ~8ms per frame including face detection
```

## Inside the KV cache

```
KEY   = prototype embedding of an emotion class (512-D, L2-normalised, precomputed)
VALUE = emotion label + valence/arousal/dominance metadata

Lookup per frame:
  1. q = CNN.extract_embedding(face_crop)          # (1, 512)
  2. s = q @ key_matrix.T                          # (1, 7), one matmul
  3. a = softmax(s / temperature)                  # sharpened distribution
  4. emotion = labels[argmax(a)]                   # O(7), trivial
```

## Why the lookup is fast

The 7x512 float32 key_matrix is 14 KB, small enough to fit entirely in the CPU's L2 cache.
Reading it takes nanoseconds, compared with microseconds for DRAM or milliseconds for disk or network.

```
Comparison:
  Storage   RAG: DRAM/SSD       CAG: L2/L3 cache (14 KB)
  Ops       RAG: O(N) scan      CAG: O(E) fixed 7 classes
  Network   RAG: Yes (10+ ms)   CAG: None
  Retrieval RAG: Yes            CAG: None
```

## Limitations

1. The knowledge is static: it only recognises the 7 cached emotion classes.
2. There is no autoregressive context (unlike LLM KV caches).
3. The cache has to be rebuilt after the CNN is fine-tuned.
4. Adding a new emotion class at runtime is hard.

## Possible extension: hybrid CAG + RAG

Use CAG for the 7 base emotions (< 1ms), and call RAG only when confidence is below a threshold
(ambiguous or unusual expressions). The aim would be cache speed for 95% of frames and better
accuracy on the edge cases.

## Benchmark (measured, Apple M5 CPU, `python main.py --benchmark`)

Each query runs the CNN embedding plus the lookup. The baseline replaces the 7-prototype cache
with a brute-force scan over 10,000 stored embeddings.

```
Metric    CAG       Baseline (10K scan)   Speedup
Mean      1.99ms    2.25ms                1.1x
P95       2.21ms    2.46ms                1.1x
```

The CNN dominates both. The lookup alone takes about 0.007 ms. The comparison leaves out the
network and database overhead that an external vector store would add.
