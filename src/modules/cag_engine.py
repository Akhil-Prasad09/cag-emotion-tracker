"""
cag_engine.py
-------------
The CAG (Cache-Augmented Generation) Inference Engine.

HOW CAG WORKS HERE:
  1. PRELOAD  — Build KV cache once from emotion prototype embeddings.
  2. QUERY    — For each frame:
       a. Extract face crop (FaceDetector)
       b. Run CNN forward pass → 512-D query embedding  (~5ms)
       c. Batch cosine similarity against KV cache      (~0.1ms)
       d. Softmax → attention distribution              (~0.01ms)
       e. Apply temporal smoothing (EMA over last N frames)
  3. OUTPUT   — (emotion, confidence, all_scores, latency_ms)

No DB, no vector index, no retrieval pipeline.
Everything lives in GPU VRAM / CPU RAM as pre-allocated tensors.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
from typing import Optional, Dict, Tuple, List

from src.modules.emotion_cnn import EmotionCNN
from src.modules.kv_cache import EmotionKVCache, EMOTION_LABELS
from src.modules.face_detector import FaceDetector


class TemporalSmoother:
    """
    Exponential Moving Average (EMA) smoothing over frame predictions.
    Prevents emotion flickering between frames.

    Formula: smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]
    alpha=0.4 → 40% weight on current frame, 60% on history.
    """

    def __init__(self, num_emotions: int, alpha: float = 0.4):
        self.alpha = alpha
        self.num_emotions = num_emotions
        self.state: Optional[np.ndarray] = None   # (E,) smoothed distribution
        self._frame_count = 0
        self._history: deque = deque(maxlen=30)   # last 30 raw predictions

    def update(self, scores: Dict[str, float], labels: List[str]) -> Dict[str, float]:
        raw = np.array([scores[l] for l in labels], dtype=np.float32)
        if self.state is None:
            self.state = raw.copy()
        else:
            self.state = self.alpha * raw + (1 - self.alpha) * self.state
        self._frame_count += 1
        self._history.append(raw.copy())
        return {labels[i]: float(self.state[i]) for i in range(len(labels))}

    def reset(self) -> None:
        self.state = None
        self._frame_count = 0

    def get_dominant_emotion(self, smoothed_scores: Dict[str, float]) -> Tuple[str, float]:
        best = max(smoothed_scores, key=smoothed_scores.get)
        return best, smoothed_scores[best]


class CAGEngine:
    """
    Main inference engine. Orchestrates all modules.

    Usage:
        engine = CAGEngine()
        engine.load()
        result = engine.infer(frame)   # frame = BGR numpy array from cv2
    """

    def __init__(
        self,
        cache_path: str = "cache/emotion_cache.pt",
        model_path: Optional[str] = None,
        embedding_dim: int = 512,
        device: str = "auto",
        temporal_alpha: float = 0.4,
        cache_temperature: float = 0.08,
        prefer_dnn_detector: bool = False,
    ):
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedding_dim = embedding_dim
        self.cache_path = cache_path
        self.model_path = model_path
        self.cache_temperature = cache_temperature
        self.prefer_dnn_detector = prefer_dnn_detector

        # Sub-modules (initialised in load())
        self.cnn: Optional[EmotionCNN] = None
        self.kv_cache: Optional[EmotionKVCache] = None
        self.face_detector: Optional[FaceDetector] = None
        self.smoother = TemporalSmoother(len(EMOTION_LABELS), alpha=temporal_alpha)

        # Perf tracking
        self._fps_window: deque = deque(maxlen=60)
        self._latency_window: deque = deque(maxlen=60)
        self._last_frame_ts: float = 0.0
        self._frame_count: int = 0
        self._no_face_frames: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def load(self, model_dir: str = "models") -> None:
        """Load all components into memory."""
        t0 = time.perf_counter()

        # 1. CNN model
        self.cnn = EmotionCNN(
            embedding_dim=self.embedding_dim,
            num_emotions=len(EMOTION_LABELS)
        ).to(self.device)
        self.cnn.eval()

        if self.model_path and __import__("os").path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            self.cnn.load_state_dict(state)
            print(f"[CAGEngine] Loaded CNN weights from {self.model_path}")
        else:
            print("[CAGEngine] Using randomly-initialised CNN "
                  "(run train.py for a trained model).")

        # Warm up JIT / CUDA kernels
        _ = self._warmup_cnn()

        # 2. KV Cache
        self.kv_cache = EmotionKVCache(
            embedding_dim=self.embedding_dim,
            device=self.device
        )
        try:
            self.kv_cache.load(self.cache_path)
        except FileNotFoundError:
            print("[CAGEngine] Cache not found, building from synthetic prototypes.")
            self.kv_cache.build_from_synthetic()
            self.kv_cache.save(self.cache_path)

        # 3. Face detector
        self.face_detector = FaceDetector(
            model_dir=model_dir,
            face_size=48,
            prefer_dnn=self.prefer_dnn_detector
        )

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[CAGEngine] Ready in {elapsed:.1f}ms on {self.device}")

    def _warmup_cnn(self) -> None:
        """Run 3 dummy forward passes to warm up CUDA kernels / JIT."""
        dummy = torch.zeros(1, 1, 48, 48, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.cnn.extract_embedding(dummy)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def infer(
        self, frame: np.ndarray
    ) -> dict:
        """
        Full CAG inference pipeline for one frame.

        Returns:
          {
            "emotion":       str,         # top-1 after smoothing
            "confidence":    float,       # 0-1
            "raw_scores":    dict,        # per-emotion before smoothing
            "smooth_scores": dict,        # per-emotion after smoothing
            "bbox":          tuple|None,  # (x,y,w,h)
            "latency_ms":    float,
            "fps":           float,
            "face_found":    bool,
          }
        """
        t_start = time.perf_counter()

        result = {
            "emotion": "neutral",
            "confidence": 0.0,
            "raw_scores": {e: 0.0 for e in EMOTION_LABELS},
            "smooth_scores": {e: 0.0 for e in EMOTION_LABELS},
            "bbox": None,
            "latency_ms": 0.0,
            "fps": 0.0,
            "face_found": False,
        }

        # --- Step 1: Face detection ---
        face_crop, bbox = self.face_detector.process_frame(frame)

        if face_crop is None:
            self._no_face_frames += 1
            # Reset smoother if face absent >30 consecutive frames
            if self._no_face_frames > 30:
                self.smoother.reset()
            result["latency_ms"] = (time.perf_counter() - t_start) * 1000
            result["fps"] = self._calc_fps()
            return result

        self._no_face_frames = 0

        # --- Step 2: CNN embedding extraction ---
        tensor = self._face_to_tensor(face_crop)   # (1, 1, 48, 48)
        with torch.no_grad():
            query_emb = self.cnn.extract_embedding(tensor)   # (1, 512)
            query_emb = query_emb / (query_emb.norm(p=2) + 1e-8)
            query_emb = torch.nan_to_num(query_emb)

        # --- Step 3: CAG lookup (the core — O(E) matmul in KV cache) ---
        raw_emotion, raw_conf, raw_scores = self.kv_cache.query(
            query_emb, temperature=self.cache_temperature
        )

        # --- Step 4: Temporal smoothing ---
        smooth_scores = self.smoother.update(raw_scores, EMOTION_LABELS)
        emotion, confidence = self.smoother.get_dominant_emotion(smooth_scores)

        # --- Finalize ---
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        self._latency_window.append(latency_ms)
        fps = self._calc_fps()

        self._frame_count += 1
        result.update({
            "emotion":       emotion,
            "confidence":    confidence,
            "raw_scores":    raw_scores,
            "smooth_scores": smooth_scores,
            "bbox":          bbox,
            "latency_ms":    latency_ms,
            "fps":           fps,
            "face_found":    True,
        })
        return result

    def _face_to_tensor(self, face_crop: np.ndarray) -> torch.Tensor:
        """Convert (48,48) float32 numpy array → (1,1,48,48) tensor on device."""
        t = torch.from_numpy(face_crop).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t.to(self.device)

    def _calc_fps(self) -> float:
        now = time.perf_counter()
        if self._last_frame_ts > 0:
            self._fps_window.append(1.0 / max(now - self._last_frame_ts, 1e-6))
        self._last_frame_ts = now
        if len(self._fps_window) == 0:
            return 0.0
        return float(np.mean(list(self._fps_window)))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def perf_stats(self) -> dict:
        lats = list(self._latency_window)
        fpss = list(self._fps_window)
        return {
            "frames_processed":  self._frame_count,
            "avg_latency_ms":    round(np.mean(lats), 2) if lats else 0,
            "p95_latency_ms":    round(np.percentile(lats, 95), 2) if lats else 0,
            "avg_fps":           round(np.mean(fpss), 1) if fpss else 0,
            "cache_hits":        self.kv_cache.cache_stats()["hit_distribution"] if self.kv_cache else {},
        }
