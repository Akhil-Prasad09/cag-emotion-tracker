"""
visualiser.py
-------------
Real-time OpenCV overlay renderer.
Draws emotion label, confidence bar, timeline strip, and perf stats.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple

# Emotion → BGR colour mapping
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   128, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   220, 220),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 0),
    "surprise": (0,   180, 255),
}

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


class RealtimeVisualiser:
    """Renders CAG inference results onto a video frame."""

    def __init__(self, history_len: int = 80):
        self.history_len = history_len
        self.emotion_history: deque = deque(maxlen=history_len)
        self.score_history: deque = deque(maxlen=history_len)
        self._frame_idx = 0

    def draw(
        self,
        frame: np.ndarray,
        result: dict,
    ) -> np.ndarray:
        """
        Draw all overlays on frame (in-place + return).
        result = output dict from CAGEngine.infer()
        """
        out = frame.copy()
        h, w = out.shape[:2]

        emotion = result.get("emotion", "neutral")
        confidence = result.get("confidence", 0.0)
        smooth = result.get("smooth_scores", {})
        bbox = result.get("bbox")
        lat = result.get("latency_ms", 0.0)
        fps = result.get("fps", 0.0)
        face_found = result.get("face_found", False)

        # Append to history
        self.emotion_history.append(emotion if face_found else None)
        self.score_history.append(smooth if face_found else None)
        self._frame_idx += 1

        # 1. Face bounding box
        if bbox and face_found:
            self._draw_bbox(out, bbox, emotion)

        # 2. Emotion label + confidence (top-left panel)
        self._draw_label_panel(out, emotion, confidence, face_found)

        # 3. Score bars (right side)
        self._draw_score_bars(out, smooth, emotion)

        # 4. Emotion timeline (bottom strip)
        self._draw_timeline(out)

        # 5. Performance stats (bottom-left)
        self._draw_perf(out, lat, fps)

        return out

    def _draw_bbox(self, frame, bbox, emotion):
        x, y, w, h = bbox
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Corner accents
        L = min(w, h) // 5
        for px, py in [(x, y), (x+w-L, y), (x, y+h-L), (x+w-L, y+h-L)]:
            cv2.rectangle(frame, (px, py), (px+L, py+4), color, -1)
            cv2.rectangle(frame, (px, py), (px+4, py+L), color, -1)

    def _draw_label_panel(self, frame, emotion, confidence, face_found):
        """Semi-transparent top-left panel with emotion + confidence."""
        H, W = frame.shape[:2]
        panel_w, panel_h = 260, 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10+panel_w, 10+panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if not face_found:
            cv2.putText(frame, "No face detected", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
            return

        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        label = emotion.upper()
        cv2.putText(frame, label, (20, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

        # Confidence bar
        bar_x, bar_y = 20, 68
        bar_w = int(220 * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+220, bar_y+12), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+12), color, -1)
        cv2.putText(frame, f"{confidence*100:.1f}%", (bar_x+225, bar_y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    def _draw_score_bars(self, frame, scores: Dict[str, float], top_emotion: str):
        """Vertical bars on the right for each emotion class."""
        if not scores:
            return
        H, W = frame.shape[:2]
        bar_area_w = 110
        x_start = W - bar_area_w - 10
        emotions = sorted(scores.keys())
        bar_max_h = 120
        bar_w = 12
        gap = 4
        y_base = H // 2 + bar_max_h // 2

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 5, y_base - bar_max_h - 20),
                      (W - 5, y_base + 20), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for i, em in enumerate(emotions):
            sc = scores.get(em, 0.0)
            bh = int(sc * bar_max_h)
            bx = x_start + i * (bar_w + gap)
            color = EMOTION_COLORS.get(em, (200, 200, 200))

            # Empty bar outline
            cv2.rectangle(frame, (bx, y_base - bar_max_h), (bx+bar_w, y_base),
                          (60, 60, 60), 1)
            # Filled portion
            if bh > 0:
                cv2.rectangle(frame, (bx, y_base - bh), (bx+bar_w, y_base),
                              color, -1)
            # Label (first 3 chars)
            cv2.putText(frame, em[:3], (bx-1, y_base+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    def _draw_timeline(self, frame):
        """Horizontal strip at the bottom showing emotion history."""
        H, W = frame.shape[:2]
        strip_h = 18
        y = H - strip_h - 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - 2), (W, H), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        history = list(self.emotion_history)
        if not history:
            return

        cell_w = max(1, W // max(len(history), 1))
        for i, em in enumerate(history):
            if em is None:
                continue
            color = EMOTION_COLORS.get(em, (100, 100, 100))
            x = i * cell_w
            cv2.rectangle(frame, (x, y), (x+cell_w, y+strip_h), color, -1)

        # Timeline label
        cv2.putText(frame, "timeline", (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    def _draw_perf(self, frame, latency_ms: float, fps: float):
        """Bottom-left perf stats."""
        H, W = frame.shape[:2]
        lat_color = (0, 220, 0) if latency_ms < 50 else \
                    (0, 165, 255) if latency_ms < 100 else (0, 0, 220)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms",
                    (10, H - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lat_color, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, "CAG Engine",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
