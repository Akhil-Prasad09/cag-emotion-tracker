"""
sklearn_engine.py  (v2 — fixes angry/surprise and disgust/angry confusion)
--------------------------------------------------------------------------
Two changes from v1:
  1. extract_features() now returns 191-D instead of 185-D.
     Six new features are appended that are strongly discriminative for
     the confused pairs:
       brow_gy            → surprise (brows shoot up = high vertical gradient)
       brow_eye_gap       → surprise vs angry (large gap vs tight)
       ul_asym            → disgust (asymmetric upper lip raise)
       nose_entropy       → disgust (nose wrinkle = high LBP entropy)
       brow_darkness_ratio→ angry (inner brow crease = darker than outer)
       mouth_dark         → surprise/fear (open mouth = dark interior)

  2. The model (models/fer_classifier.pkl) must be rebuilt with build_fer_model.py
     so it trains on 191-D vectors matching this extractor.

Everything else (face detection, EMA smoothing, run.py) stays the same.
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from collections import deque
from typing import Optional, Tuple
from skimage.feature import local_binary_pattern

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS_BGR = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   128,   0),
    "fear":     (128,   0, 128),
    "happy":    (0,   210, 210),
    "neutral":  (170, 170, 170),
    "sad":      (220, 100,   0),
    "surprise": (0,   180, 255),
}


class SklearnEmotionEngine:
    """Real-time CAG emotion engine — no deep learning framework needed."""

    def __init__(
        self,
        model_path: str = "models/fer_classifier.pkl",
        face_size:  int   = 48,
        temporal_alpha: float = 0.45,
    ):
        self.model_path   = Path(model_path)
        self.face_size    = face_size
        self.alpha        = temporal_alpha
        self.model        = None
        self.haar         = None
        self._smooth_state: Optional[np.ndarray] = None
        self._lat_buf     = deque(maxlen=60)
        self._fps_buf     = deque(maxlen=60)
        self._last_ts     = 0.0
        self._frame_count = 0
        self._no_face_n   = 0

    def load(self) -> None:
        t0 = time.perf_counter()
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade)
        if self.haar.empty():
            raise RuntimeError("Haar cascade XML not found")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model missing: {self.model_path}\n"
                "Run:  python build_fer_model.py"
            )
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        # Warm-up
        dummy = np.zeros((1, 191), dtype=np.float32)
        for _ in range(5):
            self.model.predict_proba(dummy)
        ms = (time.perf_counter() - t0) * 1000
        clf = type(list(self.model.named_steps.values())[-1]).__name__
        print(f"[Engine] Ready in {ms:.0f}ms  |  {clf}  |  191-D features")

    # ------------------------------------------------------------------
    # Feature extraction  (191-D)
    # ------------------------------------------------------------------
    @staticmethod
    def extract_features(img_f: np.ndarray) -> np.ndarray:
        """
        191-D face descriptor from a 48×48 float32 [0,1] grayscale patch.

        Layout:
          [0:10]    LBP uniform histogram            — texture (10)
          [10:22]   Gradient zone mean + std          — edge energy (12)
          [22:32]   Pixel zone mean + std             — brightness (10)
          [32:160]  16×8 downsampled face patch       — global shape (128)
          [160:172] Mouth row means                   — smile/frown curve (12)
          [172:177] Mouth corner geometry             — smile marker (5)
          [177:185] Eye/brow geometry                 — brow/eye openness (8)
          [185:191] NEW discriminative features:
            [185] brow_gy           — brow vertical gradient  (surprise↑)
            [186] brow_eye_gap      — forehead-to-eye strip   (surprise↑, angry↓)
            [187] ul_asym           — upper-lip L/R asymmetry (disgust↑)
            [188] nose_entropy      — nose-bridge LBP entropy (disgust↑)
            [189] brow_darkness_ratio — inner/outer brow dark (angry↑)
            [190] mouth_dark        — open mouth interior     (surprise/fear↑)
        """
        u8 = (img_f * 255).astype(np.uint8)

        # ── Base 185 features ──────────────────────────────────────────

        # LBP texture
        lbp   = local_binary_pattern(u8, P=8, R=1, method="uniform")
        lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

        # Gradient zones
        gx  = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        zm  = [mag[8:16, 8:40], mag[16:24, 8:22], mag[16:24, 26:40],
               mag[30:44, 14:34], mag[8:16, 8:22], mag[22:30, 16:32]]
        gf  = np.array([z.mean() for z in zm] + [z.std() for z in zm])

        # Pixel zones
        zp = [img_f[8:16, 8:40],  img_f[16:24, 8:22], img_f[16:24, 26:40],
              img_f[30:44, 14:34], img_f[22:30, 16:32]]
        pf = np.array([z.mean() for z in zp] + [z.std() for z in zp])

        # Downsampled patch
        small = cv2.resize(u8, (16, 8), interpolation=cv2.INTER_AREA
                           ).flatten().astype(np.float32) / 255.0

        # Mouth shape
        mouth      = img_f[31:43, 14:34]
        mouth_rows = np.array([mouth[r].mean() for r in range(mouth.shape[0])])
        mc_l = img_f[35:40, 14:18].mean()
        mc_r = img_f[35:40, 30:34].mean()
        mc_c = img_f[35:40, 21:27].mean()
        mouth_geom = np.array([mc_l, mc_r, mc_c, mc_l - mc_c, mc_r - mc_c])

        # Eye/brow geometry
        el_t = img_f[16:20, 12:20].mean()
        el_b = img_f[20:24, 12:20].mean()
        er_t = img_f[16:20, 28:36].mean()
        er_b = img_f[20:24, 28:36].mean()
        bl   = img_f[8:14,  10:22].mean()
        br   = img_f[8:14,  26:38].mean()
        eye_geom = np.array([el_t, el_b, er_t, er_b, bl, br, bl - el_t, br - er_t])

        # ── 6 new discriminative features ─────────────────────────────

        # [185] Brow vertical gradient — surprise brows shoot UP = high |gy| in brow row
        brow_gy = np.abs(gy[8:18, 8:40]).mean()

        # [186] Brow-eye gap brightness — surprise: brows high → bright forehead strip
        #       angry: brows low → dark, compressed strip
        brow_eye_gap = img_f[13:17, 10:38].mean()

        # [187] Upper-lip left/right asymmetry — disgust raises one side more
        ul_left  = img_f[30:35, 14:22].mean()
        ul_right = img_f[30:35, 26:34].mean()
        ul_asym  = abs(ul_left - ul_right)

        # [188] Nose-bridge LBP entropy — disgust wrinkle = more varied texture
        nose     = u8[22:32, 17:31]
        nose_lbp = local_binary_pattern(nose, P=8, R=1, method="uniform")
        nose_h, _ = np.histogram(nose_lbp, bins=6, range=(0, 6), density=True)
        nose_h   = np.where(nose_h > 0, nose_h, 1e-9)
        nose_ent = float(-(nose_h * np.log(nose_h)).sum())

        # [189] Inner/outer brow darkness ratio — angry inner brow crease = darker
        inner_brow = img_f[14:19, 17:31].mean()
        outer_brow = img_f[14:19,  8:16].mean()
        brow_dark_ratio = inner_brow / (outer_brow + 0.001)

        # [190] Mouth interior darkness — open mouth (surprise/fear) = dark region
        mouth_dark = 1.0 - img_f[34:42, 18:30].mean()

        extra = np.array([brow_gy, brow_eye_gap, ul_asym, nose_ent,
                          brow_dark_ratio, mouth_dark], dtype=np.float32)

        return np.concatenate([lbp_h, gf, pf, small, mouth_rows,
                               mouth_geom, eye_geom, extra])

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------
    def detect_and_crop(self, frame: np.ndarray
                        ) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        H, W = frame.shape[:2]
        mx, my = int(w * 0.12), int(h * 0.12)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(W, x + w + mx), min(H, y + h + my)
        crop = gray[y1:y2, x1:x2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        crop  = clahe.apply(crop)
        crop  = cv2.resize(crop, (self.face_size, self.face_size),
                           interpolation=cv2.INTER_AREA)
        return crop, (x, y, w, h)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        result = {
            "emotion": "neutral", "confidence": 0.0,
            "raw_scores":    {e: 0.0 for e in EMOTIONS},
            "smooth_scores": {e: 0.0 for e in EMOTIONS},
            "bbox": None, "latency_ms": 0.0,
            "fps": 0.0, "face_found": False,
        }
        face_u8, bbox = self.detect_and_crop(frame)
        if face_u8 is None:
            self._no_face_n += 1
            if self._no_face_n > 30:
                self._smooth_state = None
            result["latency_ms"] = (time.perf_counter() - t0) * 1000
            result["fps"] = self._fps()
            return result
        self._no_face_n = 0

        face_f = face_u8.astype(np.float32) / 255.0
        feat   = self.extract_features(face_f).reshape(1, -1)
        proba  = self.model.predict_proba(feat)[0]
        raw    = {EMOTIONS[i]: float(proba[i]) for i in range(len(EMOTIONS))}

        if self._smooth_state is None:
            self._smooth_state = proba.copy()
        else:
            self._smooth_state = (self.alpha * proba
                                  + (1 - self.alpha) * self._smooth_state)
        s    = self._smooth_state
        best = int(s.argmax())
        smooth = {EMOTIONS[i]: float(s[i]) for i in range(len(EMOTIONS))}

        lat = (time.perf_counter() - t0) * 1000
        self._lat_buf.append(lat)
        self._frame_count += 1
        result.update({
            "emotion": EMOTIONS[best], "confidence": float(s[best]),
            "raw_scores": raw, "smooth_scores": smooth,
            "bbox": bbox, "latency_ms": lat,
            "fps": self._fps(), "face_found": True,
        })
        return result

    def _fps(self) -> float:
        now = time.perf_counter()
        if self._last_ts > 0:
            self._fps_buf.append(1.0 / max(now - self._last_ts, 1e-6))
        self._last_ts = now
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def perf_stats(self) -> dict:
        lats = list(self._lat_buf)
        return {
            "frames": self._frame_count,
            "avg_ms": round(np.mean(lats), 2) if lats else 0,
            "p95_ms": round(np.percentile(lats, 95), 2) if lats else 0,
            "fps":    round(float(np.mean(list(self._fps_buf))), 1)
                      if self._fps_buf else 0,
        }

    def reset_smoother(self):
        self._smooth_state = None
