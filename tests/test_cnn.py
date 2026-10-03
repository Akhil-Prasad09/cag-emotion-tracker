"""test_cnn.py — Unit tests for EmotionCNN."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from src.modules.emotion_cnn import EmotionCNN


@pytest.fixture
def model():
    return EmotionCNN(embedding_dim=512, num_emotions=7)


def test_output_shapes(model):
    x = torch.randn(4, 1, 48, 48)
    logits, emb = model(x)
    assert logits.shape == (4, 7)
    assert emb.shape == (4, 512)


def test_embeddings_are_unit_normalised(model):
    x = torch.randn(4, 1, 48, 48)
    model.eval()
    with torch.no_grad():
        emb = model.extract_embedding(x)
    norms = torch.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_single_image_inference(model):
    model.eval()
    x = torch.randn(1, 1, 48, 48)
    with torch.no_grad():
        logits, emb = model(x)
    assert logits.shape == (1, 7)


def test_parameter_count(model):
    n = sum(p.numel() for p in model.parameters())
    # Should be under 2M for a lightweight model
    assert n < 2_000_000, f"Model too large: {n:,} params"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
