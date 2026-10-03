"""
build_cache.py
--------------
Standalone script to build / rebuild the KV cache.
Run this once before starting main.py:
  python -m src.build_cache
"""

from src.modules.kv_cache import EmotionKVCache
import torch.nn.functional as F
import torch

def main():
    print("Building emotion KV cache...")
    cache = EmotionKVCache(embedding_dim=512)
    cache.build_from_synthetic(seed=42)
    cache.save("cache/emotion_cache.pt")
    print("Cache built successfully.")
    print(f"Stats: {cache.cache_stats()}")

    # Verify by running a query
    q = F.normalize(torch.randn(1, 512), dim=1)
    emotion, conf, scores = cache.query(q)
    print(f"\nSample query → emotion='{emotion}', confidence={conf:.3f}")
    print(f"All scores: { {k: round(v,3) for k,v in scores.items()} }")

if __name__ == "__main__":
    main()
