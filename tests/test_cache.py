"""
test_cache.py — Unit tests for the KV cache module.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pytest
import tempfile
import os

from src.modules.kv_cache import EmotionKVCache, EMOTION_LABELS


@pytest.fixture
def cache():
    c = EmotionKVCache(embedding_dim=512)
    c.build_from_synthetic()
    return c


def test_build_creates_correct_number_of_entries(cache):
    assert len(cache.labels) == len(EMOTION_LABELS)
    assert cache.key_matrix.shape == (len(EMOTION_LABELS), 512)


def test_prototypes_are_unit_normalised(cache):
    norms = torch.norm(cache.key_matrix, dim=1)
    assert torch.allclose(norms, torch.ones(len(EMOTION_LABELS)), atol=1e-5)


def test_query_returns_valid_emotion(cache):
    q = F.normalize(torch.randn(1, 512), dim=1)
    emotion, conf, scores = cache.query(q)
    assert emotion in EMOTION_LABELS
    assert 0.0 <= conf <= 1.0


def test_scores_sum_to_one(cache):
    q = F.normalize(torch.randn(1, 512), dim=1)
    _, _, scores = cache.query(q)
    total = sum(scores.values())
    assert abs(total - 1.0) < 1e-5


def test_known_prototype_query_returns_correct_emotion(cache):
    """Querying with the prototype itself should return that emotion."""
    for i, emotion in enumerate(EMOTION_LABELS):
        proto = cache.key_matrix[i].unsqueeze(0)  # (1, 512)
        pred, conf, _ = cache.query(proto, temperature=0.01)
        assert pred == emotion, f"Expected {emotion}, got {pred}"
        assert conf > 0.5


def test_save_and_load_roundtrip(cache):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cache.pt")
        cache.save(path)
        loaded = EmotionKVCache(embedding_dim=512)
        loaded.load(path)
        assert loaded.labels == cache.labels
        assert torch.allclose(loaded.key_matrix, cache.key_matrix, atol=1e-6)


def test_hit_count_increments(cache):
    q = F.normalize(torch.randn(1, 512), dim=1)
    emotion, _, _ = cache.query(q)
    assert cache.entries[emotion].hit_count == 1


def test_query_with_1d_input(cache):
    q = F.normalize(torch.randn(512), dim=0)
    emotion, conf, scores = cache.query(q)
    assert emotion in EMOTION_LABELS


def test_temperature_sharpens_distribution(cache):
    q = F.normalize(torch.randn(1, 512), dim=1)
    _, _, scores_low_temp = cache.query(q, temperature=0.01)
    cache.reset_hit_counts()
    _, _, scores_high_temp = cache.query(q, temperature=1.0)
    max_low = max(scores_low_temp.values())
    max_high = max(scores_high_temp.values())
    assert max_low >= max_high, "Lower temperature should give sharper distribution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
