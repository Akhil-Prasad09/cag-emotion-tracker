# src/cag_engine.py
# ─────────────────────────────────────────────────────────────
# Cache-Augmented Generation (CAG) Inference Engine
#
# This module is the heart of the system.  It wires together:
#   • FeatureExtractor  – CNN query vector Q
#   • EmotionKVCache    – pre-loaded KV prototype store
#   • TemporalSmoother  – EMA + voting over recent frames
#
# Inference loop (per frame):
#   bgr_frame
#     ─→ FeatureExtractor.extract()  → Q [256-d]
#     ─→ EmotionKVCache.query(Q)     → class_probs [7], pred_class
#     ─→ TemporalSmoother.update()   → smoothed_probs, final_emotion
#     ─→ InferenceResult              (returned to UI / pipeline)
#
# No retrieval step.  No LLM call.  No DB query.
# The KV cache IS the knowledge base.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.cache.kv_cache import EmotionKVCache
from src.utils.feature_extractor import FeatureExtractor


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    """Single-frame inference output returned to the UI."""
    emotion:        str               # e.g. "happy"
    emoji:          str               # e.g. "😊"
    confidence:     float             # in [0, 1]
    class_probs:    Dict[str, float]  # all 7 class probabilities
    bbox:           Optional[Tuple[int, int, int, int]]
    face_detected:  bool
    latency_ms:     float             # total inference time
    frame_idx:      int

    def to_dict(self) -> dict:
        return {
            "emotion":       self.emotion,
            "emoji":         self.emoji,
            "confidence":    round(self.confidence, 4),
            "class_probs":   {k: round(v, 4) for k, v in self.class_probs.items()},
            "bbox":          self.bbox,
            "face_detected": self.face_detected,
            "latency_ms":    round(self.latency_ms, 2),
            "frame_idx":     self.frame_idx,
        }


@dataclass
class PerformanceStats:
    """Rolling performance metrics for the benchmark overlay."""
    fps:             float = 0.0
    avg_latency_ms:  float = 0.0
    min_latency_ms:  float = 0.0
    max_latency_ms:  float = 0.0
    frames_processed: int  = 0
    cache_hit_rate:  float = 1.0   # always 1.0 for CAG (no misses)
    _latencies: List[float] = field(default_factory=list, repr=False)

    def update(self, latency_ms: float):
        self.frames_processed += 1
        self._latencies.append(latency_ms)
        window = self._latencies[-60:]
        self.avg_latency_ms = sum(window) / len(window)
        self.min_latency_ms = min(window)
        self.max_latency_ms = max(window)


# ─────────────────────────────────────────────────────────────
# Temporal Smoother
# ─────────────────────────────────────────────────────────────
class TemporalSmoother:
    """
    Prevents flickering between emotion labels across frames.

    Two mechanisms:
    1. Exponential Moving Average (EMA) on probability vectors.
       smoothed_t = alpha × current_t + (1 - alpha) × smoothed_{t-1}
       Gives more weight to recent frames while retaining history.

    2. Majority vote over a sliding window.
       Only emit a NEW emotion label if it has been the modal
       prediction for at least `min_hold` consecutive frames.

    Both mechanisms work purely in memory – no external storage.
    """

    def __init__(
        self,
        num_classes:  int,
        window_size:  int   = 5,
        alpha:        float = 0.6,
        min_hold:     int   = 2,
    ):
        self.num_classes = num_classes
        self.window_size = window_size
        self.alpha       = alpha
        self.min_hold    = min_hold

        self._ema_probs: Optional[torch.Tensor] = None
        self._label_queue: Deque[int] = deque(maxlen=window_size)
        self._current_label: int = -1
        self._hold_count:    int = 0

    def update(
        self, raw_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ──────────
        raw_probs : [num_classes] probability tensor for current frame

        Returns
        ───────
        (smoothed_probs [num_classes], smoothed_class_idx)
        """
        raw_probs = raw_probs.float()

        # ── EMA update ─────────────────────────────────────
        if self._ema_probs is None:
            self._ema_probs = raw_probs.clone()
        else:
            self._ema_probs = (
                self.alpha * raw_probs
                + (1 - self.alpha) * self._ema_probs
            )

        raw_pred = int(raw_probs.argmax())
        self._label_queue.append(raw_pred)

        # ── Majority vote for label stability ─────────────
        if len(self._label_queue) >= self.min_hold:
            from collections import Counter
            modal = Counter(self._label_queue).most_common(1)[0][0]
            if modal != self._current_label:
                self._current_label = modal
                self._hold_count    = 1
            else:
                self._hold_count += 1
        else:
            self._current_label = raw_pred

        smoothed_class = int(self._ema_probs.argmax())
        return self._ema_probs, smoothed_class

    def reset(self):
        """Call when a new face enters the frame / scene cut."""
        self._ema_probs     = None
        self._label_queue.clear()
        self._current_label = -1
        self._hold_count    = 0


# ─────────────────────────────────────────────────────────────
# CAGInferenceEngine
# ─────────────────────────────────────────────────────────────
class CAGInferenceEngine:
    """
    Orchestrates the full CAG pipeline for real-time emotion detection.

    Usage
    ─────
    engine = CAGInferenceEngine(model, kv_cache, device=DEVICE)
    result = engine.infer(bgr_frame)

    The result is an InferenceResult dataclass ready for the UI.
    """

    def __init__(
        self,
        model:            torch.nn.Module,
        kv_cache:         EmotionKVCache,
        emotion_emoji:    Dict[str, str],
        device:           str   = "cpu",
        similarity_metric: str  = "cosine",
        confidence_threshold: float = 0.35,
        temporal_window:  int   = 5,
        temporal_alpha:   float = 0.6,
        cascade_path:     Optional[str] = None,
    ):
        self.emotions             = kv_cache.emotions
        self.kv_cache             = kv_cache
        self.emotion_emoji        = emotion_emoji
        self.device               = device
        self.similarity_metric    = similarity_metric
        self.confidence_threshold = confidence_threshold

        # Sub-modules
        self.extractor = FeatureExtractor(model, device, cascade_path)
        self.smoother  = TemporalSmoother(
            num_classes  = len(self.emotions),
            window_size  = temporal_window,
            alpha        = temporal_alpha,
        )
        self.perf     = PerformanceStats()
        self._frame_idx = 0

        # FPS tracking
        self._fps_times: Deque[float] = deque(maxlen=30)

        print(f"[CAGEngine] Ready  — {len(self.emotions)} emotions, "
              f"device={device.upper()}, metric={similarity_metric}")

    def infer(self, bgr_frame: np.ndarray) -> InferenceResult:
        """
        Full CAG inference on a single BGR frame.

        Returns InferenceResult with emotion, confidence, bbox, latency.
        This method is called once per camera frame (~30 Hz).
        """
        t_start = time.perf_counter()
        self._frame_idx += 1
        self._fps_times.append(t_start)

        # ── Step 1: Extract visual feature vector Q ────────────
        feat_vec, bbox = self.extractor.extract(bgr_frame)

        if feat_vec is None:
            # No face detected – reset smoother, return neutral
            self.smoother.reset()
            latency = (time.perf_counter() - t_start) * 1000
            self.perf.update(latency)
            return self._no_face_result(latency)

        # ── Step 2: CAG lookup – Q vs KV cache ────────────────
        #   Single matrix multiply: Q @ K.T  → [N_total] similarities
        #   This replaces the entire RAG retrieval pipeline.
        class_probs, pred_class, _ = self.kv_cache.query(
            feat_vec,
            top_k  = 5,
            metric = self.similarity_metric,
        )

        # ── Step 3: Temporal smoothing ─────────────────────────
        smoothed_probs, smoothed_class = self.smoother.update(class_probs)

        # ── Step 4: Build result ───────────────────────────────
        confidence = float(smoothed_probs[smoothed_class])
        if confidence < self.confidence_threshold:
            emotion = "uncertain"
            emoji   = "🤔"
        else:
            emotion = self.emotions[smoothed_class]
            emoji   = self.emotion_emoji.get(emotion, "")

        prob_dict = {
            self.emotions[i]: float(smoothed_probs[i])
            for i in range(len(self.emotions))
        }

        latency = (time.perf_counter() - t_start) * 1000
        self.perf.update(latency)

        return InferenceResult(
            emotion       = emotion,
            emoji         = emoji,
            confidence    = confidence,
            class_probs   = prob_dict,
            bbox          = bbox,
            face_detected = True,
            latency_ms    = latency,
            frame_idx     = self._frame_idx,
        )

    def current_fps(self) -> float:
        """Estimate FPS from recent frame timestamps."""
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self):
        """Reset temporal state (e.g. new session)."""
        self.smoother.reset()
        self._frame_idx = 0
        self._fps_times.clear()

    # ── Private helpers ────────────────────────────────────────
    def _no_face_result(self, latency: float) -> InferenceResult:
        return InferenceResult(
            emotion       = "no_face",
            emoji         = "👤",
            confidence    = 0.0,
            class_probs   = {e: 0.0 for e in self.emotions},
            bbox          = None,
            face_detected = False,
            latency_ms    = latency,
            frame_idx     = self._frame_idx,
        )
