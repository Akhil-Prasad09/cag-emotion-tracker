"""Smoke test: every dashboard page renders, and the trained engine classifies a real face."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("page", ["🎥 Live Demo", "📊 Benchmark", "🗂️ Cache Inspector", "📖 Architecture"])
def test_page_renders(page, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_benchmark_runs(monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("📊 Benchmark").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4


def _face_frame():
    face = cv2.imread(str(ROOT / "tests/fixtures/happy_face.jpg"))
    return cv2.copyMakeBorder(cv2.resize(face, (240, 240)), 120, 120, 200, 200,
                              cv2.BORDER_CONSTANT, value=(128, 128, 128))


class FakeCamera:
    """Stands in for cv2.VideoCapture: yields a few face frames, then ends."""
    def __init__(self, *_):
        self.left = 5
    def isOpened(self): return True
    def set(self, *_): return True
    def read(self):
        self.left -= 1
        return (True, _face_frame()) if self.left >= 0 else (False, None)
    def release(self): pass


def test_live_demo_streams(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(cv2, "VideoCapture", FakeCamera)
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    at.button[0].click().run()   # ▶ Start Camera
    assert not at.exception, at.exception
    assert "Processed 5 frames" in at.success[0].value


def test_engine_on_real_face():
    from src.modules.sklearn_engine import SklearnEmotionEngine
    engine = SklearnEmotionEngine(model_path=str(ROOT / "models/fer_classifier.pkl"))
    engine.load()
    r = engine.infer(_face_frame())
    assert r["face_found"]
    assert abs(sum(r["smooth_scores"].values()) - 1) < 1e-3
    assert r["latency_ms"] < 200
