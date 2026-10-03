# tests/test_kv_cache.py
# ─────────────────────────────────────────────────────────────
# Unit tests for the KV cache and CAG inference engine.
# Run with:  python -m pytest tests/ -v
# ─────────────────────────────────────────────────────────────

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn.functional as F

from configs.config import EMOTIONS, FEATURE_DIM
from src.cache.kv_cache import EmotionKVCache, KVCacheBuilder
from src.cag_engine import TemporalSmoother
from src.models.emotion_cnn import EmotionCNN


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_cache() -> EmotionKVCache:
    builder = KVCacheBuilder(EMOTIONS, FEATURE_DIM, "cpu")
    return builder.build_synthetic(num_prototypes=10, seed=0)


@pytest.fixture
def emotion_model() -> EmotionCNN:
    return EmotionCNN(num_classes=len(EMOTIONS), feature_dim=FEATURE_DIM)


# ─────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────
class TestEmotionCNN:
    def test_output_shapes(self, emotion_model):
        x = torch.randn(4, 1, 48, 48)
        logits, feats = emotion_model(x)
        assert logits.shape == (4, len(EMOTIONS))
        assert feats.shape  == (4, FEATURE_DIM)

    def test_feature_normalisation(self, emotion_model):
        x    = torch.randn(8, 1, 48, 48)
        feats = emotion_model.extract_features(x)
        norms = feats.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
            "Features must be L2-normalised"

    def test_deterministic_eval(self, emotion_model):
        emotion_model.eval()
        x = torch.randn(1, 1, 48, 48)
        with torch.no_grad():
            f1 = emotion_model.extract_features(x)
            f2 = emotion_model.extract_features(x)
        assert torch.allclose(f1, f2), "Eval mode must be deterministic"


# ─────────────────────────────────────────────────────────────
# KV Cache tests
# ─────────────────────────────────────────────────────────────
class TestKVCache:
    def test_build_synthetic(self, synthetic_cache):
        assert synthetic_cache.is_ready
        assert synthetic_cache.size == len(EMOTIONS) * 10

    def test_query_returns_valid_class(self, synthetic_cache):
        q = F.normalize(torch.randn(FEATURE_DIM), p=2, dim=0)
        probs, pred, sims = synthetic_cache.query(q)
        assert 0 <= int(pred) < len(EMOTIONS)
        assert abs(float(probs.sum()) - 1.0) < 1e-4, "Probs must sum to 1"

    def test_query_correct_class(self, synthetic_cache):
        """Querying a stored prototype should return its own class."""
        # Pull the first key for class 'happy' (index 3)
        happy_idx = EMOTIONS.index("happy")
        # Find first key belonging to happy
        for i, v in enumerate(synthetic_cache._values):
            if v.item() == happy_idx:
                prototype = synthetic_cache._keys[i].clone()
                break
        probs, pred, _ = synthetic_cache.query(prototype)
        assert int(pred) == happy_idx, \
            f"Query on own prototype should return same class; got {int(pred)}"

    def test_save_load_roundtrip(self, synthetic_cache):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "cache.pt")
            synthetic_cache.save(path)
            loaded = EmotionKVCache.load(path, "cpu")
            assert loaded.size == synthetic_cache.size
            assert loaded.emotions == synthetic_cache.emotions
            assert torch.allclose(loaded._keys, synthetic_cache._keys)

    def test_query_batch_stability(self, synthetic_cache):
        """100 random queries should not crash."""
        for _ in range(100):
            q = F.normalize(torch.randn(FEATURE_DIM), p=2, dim=0)
            probs, pred, _ = synthetic_cache.query(q)
            assert probs.shape == (len(EMOTIONS),)
            assert 0 <= int(pred) < len(EMOTIONS)


# ─────────────────────────────────────────────────────────────
# Temporal smoother tests
# ─────────────────────────────────────────────────────────────
class TestTemporalSmoother:
    def test_stable_emotion_stays_stable(self):
        smoother = TemporalSmoother(num_classes=7, window_size=5, alpha=0.7)
        # Feed 'happy' (class 3) repeatedly
        probs = torch.zeros(7); probs[3] = 1.0
        for _ in range(10):
            _, cls = smoother.update(probs)
        assert cls == 3

    def test_reset_clears_state(self):
        smoother = TemporalSmoother(num_classes=7)
        probs = torch.zeros(7); probs[0] = 1.0
        smoother.update(probs)
        smoother.reset()
        assert smoother._ema_probs is None
        assert len(smoother._label_queue) == 0

    def test_ema_bounded(self):
        smoother = TemporalSmoother(num_classes=7, alpha=0.5)
        for _ in range(20):
            probs = torch.rand(7)
            probs = probs / probs.sum()
            ema, _ = smoother.update(probs)
            assert torch.all(ema >= 0), "EMA probs must be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
