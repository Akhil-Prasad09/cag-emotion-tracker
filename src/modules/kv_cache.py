"""
kv_cache.py
-----------
KV Cache implementation for Cache-Augmented Generation (CAG).

WHAT IS A KV CACHE?
In transformers, the KV cache stores pre-computed Key-Value pairs so the
attention mechanism doesn't re-compute them on every forward pass.

Here we adapt this concept for emotion recognition:
  KEY   = prototype embedding of an emotion class (precomputed, frozen)
  VALUE = emotion label + metadata (confidence weights, description)

During inference:
  1. Query = live face embedding from CNN (just computed)
  2. Similarity(Query, Keys) → attention scores
  3. Weighted sum of Values → predicted emotion

This is O(E) lookup (E = num emotions), not O(N) retrieval like RAG.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Semantic descriptions — used to initialise prototype embeddings
EMOTION_DESCRIPTORS = {
    "angry":    {"valence": -0.9, "arousal":  0.9, "dominance":  0.7,
                 "au_weights": [4, 5, 7, 23, 24]},   # brow lowerer, lid tightener
    "disgust":  {"valence": -0.8, "arousal":  0.2, "dominance":  0.4,
                 "au_weights": [9, 15, 16, 17, 26]},  # nose wrinkle, lip corner pull
    "fear":     {"valence": -0.8, "arousal":  0.9, "dominance": -0.7,
                 "au_weights": [1, 2, 4, 5, 7, 20, 26]},
    "happy":    {"valence":  0.9, "arousal":  0.6, "dominance":  0.5,
                 "au_weights": [6, 12, 25]},          # cheek raiser, lip corner puller
    "neutral":  {"valence":  0.0, "arousal":  0.0, "dominance":  0.0,
                 "au_weights": []},
    "sad":      {"valence": -0.7, "arousal": -0.5, "dominance": -0.5,
                 "au_weights": [1, 4, 15, 17, 54]},
    "surprise": {"valence":  0.1, "arousal":  0.9, "dominance": -0.3,
                 "au_weights": [1, 2, 5, 26, 27]},
}


@dataclass
class CacheEntry:
    """A single K-V pair in the emotion cache."""
    emotion:     str
    key:         torch.Tensor      # prototype embedding (512-D, L2-normalised)
    valence:     float             # positive/negative dimension (-1 to 1)
    arousal:     float             # calm/excited dimension (-1 to 1)
    dominance:   float             # submissive/dominant (-1 to 1)
    hit_count:   int = 0           # how many frames matched this entry
    last_hit_ts: float = 0.0


class EmotionKVCache:
    """
    In-memory Key-Value cache for emotion embeddings.

    Internal layout (all in one contiguous tensor for fast batch similarity):
      self.key_matrix : (E, D) — E emotions × D embedding dims
      self.labels     : list[str] of length E

    Cache build → .build_from_synthetic()  or  .load()
    Inference    → .query(embedding) returns (emotion, confidence, all_scores)
    """

    def __init__(self, embedding_dim: int = 512, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.labels: List[str] = []
        self.key_matrix: Optional[torch.Tensor] = None   # (E, D)
        self.entries: Dict[str, CacheEntry] = {}
        self._built = False
        self._build_ts: float = 0.0

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------
    def build_from_synthetic(self, seed: int = 42) -> None:
        """
        Build prototype embeddings from emotion semantic descriptors.

        Strategy:
          1. Initialise a base vector per emotion using VAD (valence-arousal-dominance)
             features — psychologically grounded anchor in a 3-D subspace.
          2. Expand to full embedding_dim using structured noise seeded per emotion
             (reproducible — same seed → same cache).
          3. L2-normalise each prototype so cosine similarity is just a dot product.

        In a production system, these would be the MEAN of all training embeddings
        per class (cluster centroids), updated periodically.
        """
        rng = np.random.RandomState(seed)
        prototypes = []

        for emotion in EMOTION_LABELS:
            desc = EMOTION_DESCRIPTORS[emotion]

            # 3-D semantic anchor
            vad = np.array([desc["valence"], desc["arousal"], desc["dominance"]])

            # Project VAD into embedding space using a fixed random projection matrix
            proj_rng = np.random.RandomState(hash(emotion) % (2**31))
            proj = proj_rng.randn(3, self.embedding_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

            base = vad @ proj   # (D,)

            # Add structured noise to separate classes in high-dim space
            noise = rng.randn(self.embedding_dim).astype(np.float32) * 0.3
            proto = base + noise

            # L2 normalise
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes.append(proto)

            self.entries[emotion] = CacheEntry(
                emotion=emotion,
                key=torch.tensor(proto, dtype=torch.float32),
                valence=desc["valence"],
                arousal=desc["arousal"],
                dominance=desc["dominance"],
            )

        self.labels = list(EMOTION_LABELS)
        self.key_matrix = torch.tensor(
            np.stack(prototypes, axis=0), dtype=torch.float32
        ).to(self.device)   # (E, D)

        self._built = True
        self._build_ts = time.time()
        print(f"[KVCache] Built {len(self.labels)}-entry cache "
              f"(dim={self.embedding_dim}) on {self.device}")

    def update_prototypes_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: List[int],
        momentum: float = 0.9
    ) -> None:
        """
        EMA update of prototype vectors after supervised training.
        momentum=0.9 means 90% old + 10% new per batch.
        """
        if not self._built:
            raise RuntimeError("Cache not built yet.")
        for cls_idx, emotion in enumerate(self.labels):
            mask = [i for i, l in enumerate(labels) if l == cls_idx]
            if not mask:
                continue
            batch_mean = embeddings[mask].mean(0)  # (D,)
            batch_mean = F.normalize(batch_mean.unsqueeze(0), dim=1).squeeze(0)
            # EMA
            old = self.key_matrix[cls_idx]
            new = momentum * old + (1 - momentum) * batch_mean
            self.key_matrix[cls_idx] = F.normalize(new.unsqueeze(0), dim=1).squeeze(0)
            self.entries[emotion].key = self.key_matrix[cls_idx].cpu()

    # ------------------------------------------------------------------
    # SERIALISE / DESERIALISE
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist cache to disk as a .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key_matrix": self.key_matrix.cpu(),
            "labels": self.labels,
            "embedding_dim": self.embedding_dim,
            "build_ts": self._build_ts,
            "metadata": {
                e: {
                    "valence": self.entries[e].valence,
                    "arousal": self.entries[e].arousal,
                    "dominance": self.entries[e].dominance,
                }
                for e in self.labels
            }
        }
        torch.save(payload, path)
        print(f"[KVCache] Saved to {path}")

    def load(self, path: str) -> None:
        """Load persisted cache from disk."""
        payload = torch.load(path, map_location="cpu")
        self.labels = payload["labels"]
        self.embedding_dim = payload["embedding_dim"]
        self._build_ts = payload.get("build_ts", 0.0)
        self.key_matrix = payload["key_matrix"].to(self.device)
        meta = payload.get("metadata", {})
        for i, emotion in enumerate(self.labels):
            m = meta.get(emotion, {})
            self.entries[emotion] = CacheEntry(
                emotion=emotion,
                key=self.key_matrix[i].cpu(),
                valence=m.get("valence", 0.0),
                arousal=m.get("arousal", 0.0),
                dominance=m.get("dominance", 0.0),
            )
        self._built = True
        print(f"[KVCache] Loaded {len(self.labels)}-entry cache from {path}")

    # ------------------------------------------------------------------
    # INFERENCE — the core CAG lookup
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: torch.Tensor,  # (1, D) or (D,) — L2 normalised
        temperature: float = 0.1,
        top_k: int = 3
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        CAG inference step.

        Computes attention scores = softmax(Q · K^T / temperature).
        Returns the argmax emotion, its confidence, and all emotion scores.

        temperature: lower → sharper (more confident) distribution
        top_k: number of top emotions returned in detail

        Complexity: O(E × D) single matmul — no retrieval, no DB query.
        """
        if not self._built:
            raise RuntimeError("Cache not initialised. Call build_from_synthetic() or load().")

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (1, D)

        q = query_embedding.to(self.device)    # (1, D)
        K = self.key_matrix                    # (E, D)

        # Cosine similarity (both normalised → just dot product)
        sim = (q @ K.T).squeeze(0)             # (E,)

        # Softmax with temperature → attention weights
        attn = F.softmax(sim / temperature, dim=0)   # (E,)

        # Argmax prediction
        best_idx = attn.argmax().item()
        emotion = self.labels[best_idx]
        confidence = attn[best_idx].item()

        # All scores as dict
        scores = {self.labels[i]: attn[i].item() for i in range(len(self.labels))}

        # Update hit stats
        self.entries[emotion].hit_count += 1
        self.entries[emotion].last_hit_ts = time.time()

        return emotion, confidence, scores

    def reset_hit_counts(self) -> None:
        for e in self.entries.values():
            e.hit_count = 0

    def cache_stats(self) -> Dict:
        return {
            "entries": len(self.labels),
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "built": self._built,
            "hit_distribution": {e: self.entries[e].hit_count for e in self.labels},
        }


if __name__ == "__main__":
    cache = EmotionKVCache(embedding_dim=512)
    cache.build_from_synthetic()
    cache.save("cache/emotion_cache.pt")

    # Simulate a query
    fake_q = F.normalize(torch.randn(1, 512), dim=1)
    emotion, conf, scores = cache.query(fake_q)
    print(f"Predicted: {emotion}  Confidence: {conf:.3f}")
    print(f"All scores: { {k: round(v, 3) for k, v in scores.items()} }")
