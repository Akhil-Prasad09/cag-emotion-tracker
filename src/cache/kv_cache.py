# src/cache/kv_cache.py
# ─────────────────────────────────────────────────────────────
# KV Cache for Cache-Augmented Generation (CAG)
#
# What is a KV Cache in this context?
# ────────────────────────────────────
# In Transformer LLMs, the KV cache stores pre-computed Key and
# Value matrices so the attention mechanism doesn't recompute them
# on every token.  We adopt the same philosophy for emotion
# detection:
#
#   KEY   = prototype feature vector  (256-d) for each known emotion
#   VALUE = emotion class label + metadata (confidence prior, etc.)
#
# At inference time:
#   1. CNN extracts a live feature vector Q  (the "query")
#   2. We compute  sim(Q, K_i)  for every prototype K_i in the cache
#   3. The K_i with highest similarity "fires" → returns its VALUE
#
# Zero retrieval pipeline, zero database round-trip.
# The entire knowledge base lives in a single tensor in RAM/VRAM.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# EmotionKVCache
# ─────────────────────────────────────────────────────────────
class EmotionKVCache:
    """
    Stores prototype emotion embeddings as a flat tensor pair:
        keys   → [N_total, D]   (L2-normalised feature vectors)
        values → [N_total]      (integer class indices)

    where N_total = num_emotions × num_prototypes_per_emotion.

    Layout example (3 emotions, 2 prototypes each):
        keys[0], keys[1]  → happy
        keys[2], keys[3]  → sad
        keys[4], keys[5]  → angry
        values = [0, 0, 1, 1, 2, 2]

    Why flat?
        A single batched matrix multiply  Q @ K.T  computes all
        similarities in one GPU/CPU kernel – O(1) latency regardless
        of how many prototypes we store (up to memory limits).
    """

    def __init__(
        self,
        emotions: List[str],
        feature_dim: int = 256,
        device: str = "cpu",
    ):
        self.emotions    = emotions
        self.num_classes = len(emotions)
        self.feature_dim = feature_dim
        self.device      = device

        # Allocated lazily after build()
        self._keys:   Optional[torch.Tensor] = None  # [N, D] float32
        self._values: Optional[torch.Tensor] = None  # [N]   int64
        self._class_counts: Dict[int, int]   = {}

        # Metadata
        self.built_at: Optional[float] = None
        self.num_prototypes_per_class: int = 0

    # ── Build / preload ───────────────────────────────────────
    def build(
        self,
        prototype_dict: Dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> None:
        """
        Populate the cache from a dictionary
            { emotion_name → tensor [n_protos, feature_dim] }

        Steps
        ─────
        1. Validate that all keys are known emotion names.
        2. Stack all prototype tensors → single [N_total, D] matrix.
        3. L2-normalise rows (optional but required for cosine sim).
        4. Build a corresponding integer label tensor.
        5. Move both tensors to the target device.
        """
        key_list: List[torch.Tensor] = []
        val_list: List[torch.Tensor] = []

        for cls_idx, emotion in enumerate(self.emotions):
            if emotion not in prototype_dict:
                raise ValueError(
                    f"Missing prototypes for emotion '{emotion}'. "
                    f"Got keys: {list(prototype_dict.keys())}"
                )
            protos = prototype_dict[emotion].float()  # [n, D]
            assert protos.ndim == 2 and protos.shape[1] == self.feature_dim, (
                f"Prototype tensor for '{emotion}' must be [n, {self.feature_dim}], "
                f"got {protos.shape}"
            )
            n = protos.shape[0]
            self._class_counts[cls_idx] = n
            key_list.append(protos)
            val_list.append(torch.full((n,), cls_idx, dtype=torch.long))

        keys = torch.cat(key_list, dim=0)   # [N_total, D]
        vals = torch.cat(val_list, dim=0)   # [N_total]

        if normalize:
            keys = F.normalize(keys, p=2, dim=1)

        self._keys   = keys.to(self.device)
        self._values = vals.to(self.device)
        self.num_prototypes_per_class = key_list[0].shape[0]
        self.built_at = time.time()

        print(
            f"[KVCache] Built  — {len(self._keys)} prototypes "
            f"({self.num_prototypes_per_class}×{self.num_classes} classes) "
            f"on {self.device.upper()}  |  "
            f"mem ≈ {self._keys.element_size() * self._keys.numel() / 1024:.1f} KB"
        )

    # ── Persistence ───────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialise the cache to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys":          self._keys.cpu(),
                "values":        self._values.cpu(),
                "emotions":      self.emotions,
                "feature_dim":   self.feature_dim,
                "built_at":      self.built_at,
                "class_counts":  self._class_counts,
            },
            path,
        )
        print(f"[KVCache] Saved  → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EmotionKVCache":
        """Deserialise the cache from disk – O(1) amortised startup."""
        ckpt   = torch.load(path, map_location="cpu", weights_only=False)
        cache  = cls(
            emotions    = ckpt["emotions"],
            feature_dim = ckpt["feature_dim"],
            device      = device,
        )
        cache._keys          = ckpt["keys"].to(device)
        cache._values        = ckpt["values"].to(device)
        cache._class_counts  = ckpt["class_counts"]
        cache.built_at       = ckpt["built_at"]
        cache.num_prototypes_per_class = (
            ckpt["keys"].shape[0] // len(ckpt["emotions"])
        )
        age_h = (time.time() - cache.built_at) / 3600
        print(
            f"[KVCache] Loaded ← {path}  "
            f"({len(cache._keys)} protos, age {age_h:.1f}h, device={device.upper()})"
        )
        return cache

    # ── Core inference operation ──────────────────────────────
    @torch.no_grad()
    def query(
        self,
        query_vec: torch.Tensor,        # [D] or [B, D]
        top_k: int = 5,
        metric: str = "cosine",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The single CAG inference step.

        Parameters
        ──────────
        query_vec : live feature vector from the CNN extractor
        top_k     : how many nearest-cache-entries to aggregate
        metric    : "cosine" (default) or "dot"

        Returns
        ───────
        class_probs  [num_classes]   – soft probability over emotions
        pred_class   scalar int      – argmax emotion index
        top_sims     [top_k]         – raw similarity scores (debug)

        How it works (CAG mechanics)
        ────────────────────────────
        1.  Q  = L2-normalise(query_vec)          shape [D]
        2.  S  = Q @ K.T                          shape [N_total]
            This is a single BLAS gemv — extremely fast.
        3.  Take top-k entries of S.
        4.  For each retrieved entry we know its emotion class (value).
        5.  We aggregate similarity scores per class via softmax-weighted
            voting → class probability distribution.

        Unlike RAG, there is NO vector DB call, NO network hop,
        NO tokenisation.  The entire lookup is a matrix multiply
        in shared memory → microsecond latency.
        """
        assert self._keys is not None, "Cache not built. Call build() or load() first."

        # Ensure [1, D] for batch-compatible ops
        q = query_vec.to(self.device).float()
        if q.ndim == 1:
            q = q.unsqueeze(0)                          # [1, D]

        if metric == "cosine":
            q = F.normalize(q, p=2, dim=1)              # unit vector
            # Keys already normalised → dot product = cosine sim
            sims = (q @ self._keys.T).squeeze(0)        # [N_total]
        elif metric == "dot":
            sims = (q @ self._keys.T).squeeze(0)
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        # ── Top-k aggregation ─────────────────────────────────
        k = min(top_k, len(sims))
        top_vals, top_idx = torch.topk(sims, k)

        # Weighted vote: accumulate similarity scores per class
        class_scores = torch.zeros(self.num_classes, device=self.device)
        for val, idx in zip(top_vals, top_idx):
            class_label = self._values[idx].item()
            class_scores[class_label] += val.item()

        class_probs = F.softmax(class_scores * 5.0, dim=0)  # temperature=0.2
        pred_class  = class_probs.argmax()

        return class_probs, pred_class, top_vals

    # ── Accessors ─────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self._keys is not None

    @property
    def size(self) -> int:
        return 0 if self._keys is None else len(self._keys)

    def class_name(self, idx: int) -> str:
        return self.emotions[int(idx)]

    def stats(self) -> dict:
        if not self.is_ready:
            return {"status": "empty"}
        return {
            "status":        "ready",
            "total_protos":  self.size,
            "classes":       self.num_classes,
            "feature_dim":   self.feature_dim,
            "device":        self.device,
            "mem_kb":        round(
                self._keys.element_size() * self._keys.numel() / 1024, 2
            ),
        }


# ─────────────────────────────────────────────────────────────
# KVCacheBuilder — synthetic prototype generation
# ─────────────────────────────────────────────────────────────
class KVCacheBuilder:
    """
    Builds the KV cache from:
      (a) A trained EmotionCNN + image dataset  [preferred]
      (b) Synthetically generated Gaussian clusters [fallback]

    The synthetic path lets the system run without any labelled
    facial image dataset, which is useful for testing / demos.
    """

    def __init__(
        self,
        emotions: List[str],
        feature_dim: int = 256,
        device: str = "cpu",
    ):
        self.emotions    = emotions
        self.feature_dim = feature_dim
        self.device      = device

    # ── Path A: extract from real images via model ─────────────
    def build_from_model(
        self,
        model: torch.nn.Module,
        image_loader,          # DataLoader yielding (imgs, labels)
        num_prototypes: int = 20,
    ) -> EmotionKVCache:
        """
        Pass labelled face patches through the CNN and average
        the resulting feature vectors into class prototypes.
        This is the proper, production path.
        """
        import torch

        model.eval().to(self.device)
        accum: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(len(self.emotions))
        }

        print("[KVCacheBuilder] Extracting features from dataset…")
        with torch.no_grad():
            for imgs, labels in image_loader:
                imgs   = imgs.to(self.device)
                feats  = model.extract_features(imgs)          # [B, D]
                for feat, lbl in zip(feats, labels):
                    accum[int(lbl)].append(feat.cpu())
                    if all(len(v) >= num_prototypes * 10
                           for v in accum.values()):
                        break   # enough samples collected

        prototype_dict: Dict[str, torch.Tensor] = {}
        for cls_idx, emotion in enumerate(self.emotions):
            vecs = torch.stack(accum[cls_idx])               # [n, D]
            # K-means-lite: pick `num_prototypes` cluster centres via
            # farthest-point sampling for maximum coverage
            protos = self._fps(vecs, num_prototypes)
            prototype_dict[emotion] = protos

        cache = EmotionKVCache(self.emotions, self.feature_dim, self.device)
        cache.build(prototype_dict)
        return cache

    # ── Path B: synthetic Gaussian clusters ──────────────────
    def build_synthetic(
        self,
        num_prototypes: int = 20,
        seed: int = 42,
    ) -> EmotionKVCache:
        """
        Creates deterministic Gaussian clusters in feature space.

        Each emotion class is centred at a unique random mean
        with intra-class variance = 0.15.  The clusters are
        well-separated (different means) so cosine similarity
        lookups work correctly.

        This makes the system fully runnable without a GPU or
        facial image dataset – perfect for CI/demo/testing.
        """
        rng = np.random.default_rng(seed)

        # Generate well-separated class centroids
        centroids = rng.standard_normal((len(self.emotions), self.feature_dim))
        # Orthogonalise via QR for maximum class separation
        if len(self.emotions) <= self.feature_dim:
            centroids, _ = np.linalg.qr(centroids.T)
            centroids    = centroids.T[:len(self.emotions)]

        prototype_dict: Dict[str, torch.Tensor] = {}
        for cls_idx, emotion in enumerate(self.emotions):
            centre = centroids[cls_idx]
            noise  = rng.standard_normal((num_prototypes, self.feature_dim)) * 0.15
            protos = torch.tensor(centre + noise, dtype=torch.float32)
            prototype_dict[emotion] = protos

        print(f"[KVCacheBuilder] Synthetic cache: "
              f"{len(self.emotions)} classes × {num_prototypes} prototypes")

        cache = EmotionKVCache(self.emotions, self.feature_dim, self.device)
        cache.build(prototype_dict)
        return cache

    # ── Farthest-point sampling ────────────────────────────────
    @staticmethod
    def _fps(points: torch.Tensor, k: int) -> torch.Tensor:
        """Select k maximally spread points from a point cloud."""
        n = len(points)
        if n <= k:
            return points
        selected = [0]
        dists    = torch.full((n,), float("inf"))
        for _ in range(k - 1):
            last  = points[selected[-1]]
            d     = torch.norm(points - last, dim=1)
            dists = torch.minimum(dists, d)
            selected.append(int(dists.argmax()))
        return points[selected]
