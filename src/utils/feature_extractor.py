# src/utils/feature_extractor.py
# ─────────────────────────────────────────────────────────────
# Visual Feature Extraction Module
#
# Pipeline per frame:
#   Raw BGR frame
#     → grayscale
#     → face detection (Haar cascade)
#     → crop + align + resize to 48×48
#     → normalise to [-1, 1]
#     → batch tensor  [1, 1, 48, 48]
#     → CNN.extract_features()
#     → L2-normalised 256-d vector  ← fed to CAG engine
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# FaceDetector
# ─────────────────────────────────────────────────────────────
class FaceDetector:
    """
    Wraps OpenCV's Haar cascade for fast frontal face detection.
    Falls back to a DNN-based detector if cascade file is missing.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int  = 5,
        min_size: Tuple[int, int] = (60, 60),
    ):
        self.scale_factor  = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size      = min_size

        # Try loading Haar cascade
        if cascade_path and Path(cascade_path).exists():
            self.cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Use the cascade bundled with OpenCV
            cv2_data = cv2.data.haarcascades
            default  = cv2_data + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(default)

        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade XML. "
                               "Reinstall opencv-python.")

    def detect(
        self,
        gray_frame: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) bounding boxes,
        sorted by area (largest face first).
        """
        faces = self.cascade.detectMultiScale(
            gray_frame,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbors,
            minSize      = self.min_size,
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        # Sort by area descending (take the most prominent face)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return [tuple(f) for f in faces]

    def largest_face(
        self, gray_frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Convenience method – returns the single largest face or None."""
        faces = self.detect(gray_frame)
        return faces[0] if faces else None


# ─────────────────────────────────────────────────────────────
# FacePreprocessor
# ─────────────────────────────────────────────────────────────
class FacePreprocessor:
    """
    Converts a raw face crop into a model-ready tensor.

    Steps:
      1. Crop to slightly expanded bounding box (captures chin / forehead)
      2. Resize to target_size  (default 48×48)
      3. Apply CLAHE for illumination normalisation
      4. Convert to float tensor and normalise to [-1, 1]
    """

    def __init__(self, target_size: Tuple[int, int] = (48, 48)):
        self.target_size = target_size
        # CLAHE (Contrast Limited Adaptive Histogram Equalisation)
        # improves robustness under poor / variable lighting
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def crop_and_preprocess(
        self,
        gray_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        expand: float = 0.15,
    ) -> Optional[torch.Tensor]:
        """
        Parameters
        ──────────
        gray_frame : full grayscale frame
        bbox       : (x, y, w, h) from face detector
        expand     : fractional padding added around bbox

        Returns
        ───────
        Tensor [1, 1, H, W] or None if crop is degenerate.
        """
        x, y, w, h = bbox
        ih, iw = gray_frame.shape

        # Expand bbox symmetrically
        dx = int(w * expand)
        dy = int(h * expand)
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(iw, x + w + dx)
        y2 = min(ih, y + h + dy)

        crop = gray_frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # CLAHE normalisation
        crop = self.clahe.apply(crop)

        # Resize
        crop = cv2.resize(crop, self.target_size,
                          interpolation=cv2.INTER_LINEAR)

        # → float32 tensor, normalise to [-1, 1]
        tensor = torch.from_numpy(crop).float() / 127.5 - 1.0
        return tensor.unsqueeze(0).unsqueeze(0)   # [1, 1, 48, 48]


# ─────────────────────────────────────────────────────────────
# FeatureExtractor  (combines detector + preprocessor + model)
# ─────────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    High-level interface used by the CAG inference engine.

    Owns:
      • FaceDetector
      • FacePreprocessor
      • EmotionCNN (feature-extraction only – no classifier head)

    Call `extract(bgr_frame)` to get the 256-d query vector Q
    that is compared against the KV cache.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        cascade_path: Optional[str] = None,
    ):
        self.device      = device
        self.model       = model.eval().to(device)
        self.detector    = FaceDetector(cascade_path)
        self.preprocessor = FacePreprocessor(target_size=(48, 48))

    @torch.no_grad()
    def extract(
        self,
        bgr_frame: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]]]:
        """
        Parameters
        ──────────
        bgr_frame : raw OpenCV frame (BGR, uint8)

        Returns
        ───────
        (feature_vec [D], bbox) or (None, None) if no face found.

        feature_vec is L2-normalised – ready for cosine sim lookup.
        """
        t0 = time.perf_counter()

        # 1. Convert to grayscale for face detection
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect largest face
        bbox = self.detector.largest_face(gray)
        if bbox is None:
            return None, None

        # 3. Crop + preprocess
        tensor = self.preprocessor.crop_and_preprocess(gray, bbox)
        if tensor is None:
            return None, None

        # 4. CNN feature extraction (only the extractor path)
        tensor = tensor.to(self.device)
        feats  = self.model.extract_features(tensor)  # [1, 256]
        feats  = F.normalize(feats, p=2, dim=1)

        _ = time.perf_counter() - t0   # track internally if needed

        return feats.squeeze(0), bbox   # [256], (x,y,w,h)

    def extract_batch(
        self,
        bgr_frames: List[np.ndarray],
    ) -> List[Tuple[Optional[torch.Tensor], Optional[tuple]]]:
        """Process a list of frames; useful for benchmarking."""
        return [self.extract(f) for f in bgr_frames]
