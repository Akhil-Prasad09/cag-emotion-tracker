# src/realtime_display.py
# ─────────────────────────────────────────────────────────────
# Real-Time UI Module (OpenCV window)
#
# Draws:
#  • Webcam feed
#  • Face bounding box (colour-coded per emotion)
#  • Emotion label + emoji + confidence bar
#  • FPS and latency overlay
#  • Mini emotion history timeline (bottom strip)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path
from typing import Deque, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from configs.config import (
    CAMERA_INDEX, CACHE_PATH, DEVICE, EMOTION_EMOJI,
    EMOTIONS, FEATURE_DIM, FRAME_HEIGHT, FRAME_WIDTH,
    MODEL_CHECKPOINT, SIMILARITY_METRIC, TARGET_FPS,
    TEMPORAL_ALPHA, TEMPORAL_WINDOW, CONFIDENCE_THRESHOLD,
)
from src.cache.kv_cache import EmotionKVCache, KVCacheBuilder
from src.cag_engine import CAGInferenceEngine
from src.models.emotion_cnn import EmotionCNN


# ── Colour palette (BGR) ──────────────────────────────────────
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 50),
    "fear":     (130, 0,   130),
    "happy":    (0,   200, 0),
    "neutral":  (180, 180, 180),
    "sad":      (220, 120, 0),
    "surprise": (0,   200, 220),
    "uncertain": (100, 100, 100),
    "no_face":  (50,  50,  50),
}


def draw_confidence_bar(
    frame: np.ndarray,
    x: int, y: int, width: int, height: int,
    value: float,
    color: tuple,
):
    """Draw a horizontal progress bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (60, 60, 60), -1)
    fill = int(width * min(max(value, 0), 1))
    cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (120, 120, 120), 1)


def draw_overlay(
    frame: np.ndarray,
    result,
    fps: float,
    history: Deque,
):
    """Draw all UI elements onto the frame in-place."""
    h, w = frame.shape[:2]

    # ── Bounding box ──────────────────────────────────────────
    if result.face_detected and result.bbox:
        x, y, bw, bh = result.bbox
        color = EMOTION_COLORS.get(result.emotion, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        # Label background
        label = f"{result.emotion.upper()}  {result.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame,
                      (x, y - th - 14), (x + tw + 10, y),
                      color, -1)
        cv2.putText(frame, label, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ── HUD panel (top-right) ─────────────────────────────────
    panel_x = w - 220
    cv2.rectangle(frame, (panel_x - 5, 0), (w, 230),
                  (20, 20, 20), -1)

    cv2.putText(frame, "CAG EMOTION", (panel_x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS   : {fps:5.1f}", (panel_x, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"LATENCY: {result.latency_ms:4.1f}ms",
                (panel_x, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRAME : {result.frame_idx}", (panel_x, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # ── Mini probability bars ──────────────────────────────────
    cv2.putText(frame, "Emotion probabilities:", (panel_x, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (160, 160, 160), 1, cv2.LINE_AA)
    for i, emo in enumerate(EMOTIONS):
        prob  = result.class_probs.get(emo, 0.0)
        color = EMOTION_COLORS.get(emo, (180, 180, 180))
        yy    = 120 + i * 15
        cv2.putText(frame, f"{emo[:7]:<7}", (panel_x, yy + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    color, 1, cv2.LINE_AA)
        draw_confidence_bar(
            frame, panel_x + 65, yy, 120, 10, prob, color)

    # ── Timeline strip (bottom) ────────────────────────────────
    if len(history) > 1:
        strip_h = 18
        strip_y = h - strip_h
        slot_w  = max(1, w // len(history))
        for i, emo in enumerate(history):
            color = EMOTION_COLORS.get(emo, (80, 80, 80))
            cv2.rectangle(frame,
                          (i * slot_w, strip_y),
                          ((i + 1) * slot_w, h),
                          color, -1)
        cv2.putText(frame, "emotion history →",
                    (5, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (230, 230, 230), 1, cv2.LINE_AA)


def run(camera_index: int = CAMERA_INDEX):
    """Open webcam and run the real-time CAG emotion detection loop."""

    # ── 1. Load / build KV cache ──────────────────────────────
    cache_path = Path(CACHE_PATH)
    if cache_path.exists():
        kv_cache = EmotionKVCache.load(str(cache_path), DEVICE)
    else:
        print("[display] Cache not found – building synthetic cache…")
        builder  = KVCacheBuilder(EMOTIONS, FEATURE_DIM, DEVICE)
        kv_cache = builder.build_synthetic()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        kv_cache.save(str(cache_path))

    # ── 2. Load model ─────────────────────────────────────────
    model = EmotionCNN(num_classes=len(EMOTIONS), feature_dim=FEATURE_DIM)
    ckpt  = Path(MODEL_CHECKPOINT)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(str(ckpt), map_location=DEVICE, weights_only=True)
        )
        print(f"[display] Loaded model: {MODEL_CHECKPOINT}")
    else:
        print("[display] No model checkpoint – using random weights.")

    # ── 3. Build CAG engine ───────────────────────────────────
    import torch
    engine = CAGInferenceEngine(
        model                = model,
        kv_cache             = kv_cache,
        emotion_emoji        = EMOTION_EMOJI,
        device               = DEVICE,
        similarity_metric    = SIMILARITY_METRIC,
        confidence_threshold = CONFIDENCE_THRESHOLD,
        temporal_window      = TEMPORAL_WINDOW,
        temporal_alpha       = TEMPORAL_ALPHA,
    )

    # ── 4. Open camera ────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[display] ERROR: Cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("[display] Press 'q' to quit, 'r' to reset temporal state")

    history: Deque[str] = collections.deque(maxlen=FRAME_WIDTH // 3)
    fps_times: Deque[float] = collections.deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference ──────────────────────────────────────────
        result = engine.infer(frame)
        fps_times.append(time.perf_counter())
        fps = engine.current_fps()

        history.append(result.emotion)

        # ── Draw UI ────────────────────────────────────────────
        draw_overlay(frame, result, fps, history)
        cv2.imshow("CAG Emotion Detection  [q=quit  r=reset]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            engine.reset()
            history.clear()
            print("[display] Temporal state reset")

    cap.release()
    cv2.destroyAllWindows()

    # ── Print session summary ─────────────────────────────────
    ps = engine.perf
    print(f"\n[Session Summary]")
    print(f"  Frames processed : {ps.frames_processed}")
    print(f"  Avg latency      : {ps.avg_latency_ms:.2f} ms")
    print(f"  Min latency      : {ps.min_latency_ms:.2f} ms")
    print(f"  Max latency      : {ps.max_latency_ms:.2f} ms")
    print(f"  Approx FPS       : {engine.current_fps():.1f}")


if __name__ == "__main__":
    run()
