"""
face_detector.py
----------------
Face detection + facial region extraction using OpenCV.
Supports:
  - Haar Cascade (fast, CPU-only fallback)
  - OpenCV DNN with Caffe model (more accurate)

Outputs a 48×48 grayscale normalised face crop ready for the CNN.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import urllib.request
import os


class FaceDetector:
    """
    Multi-backend face detector with automatic fallback.

    Priority: DNN (if model files exist) → Haar Cascade
    """

    HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Caffe DNN model URLs (OpenCV's pre-trained model)
    DNN_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    DNN_MODEL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(
        self,
        model_dir: str = "models",
        face_size: int = 48,
        min_face_px: int = 40,
        dnn_confidence: float = 0.5,
        prefer_dnn: bool = True
    ):
        self.face_size = face_size
        self.min_face_px = min_face_px
        self.dnn_confidence = dnn_confidence
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.dnn_net = None
        self.haar_cascade = None
        self._backend = "none"

        if prefer_dnn:
            self._try_load_dnn()
        if self._backend != "dnn":
            self._load_haar()

        print(f"[FaceDetector] Using backend: {self._backend}")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _try_load_dnn(self) -> None:
        proto_path = self.model_dir / "deploy.prototxt"
        model_path = self.model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        # Download if missing
        if not proto_path.exists():
            try:
                print("[FaceDetector] Downloading DNN proto...")
                urllib.request.urlretrieve(self.DNN_PROTO, str(proto_path))
            except Exception as e:
                print(f"[FaceDetector] DNN proto download failed: {e}")
                return

        if not model_path.exists():
            try:
                print("[FaceDetector] Downloading DNN weights (~1 MB)...")
                urllib.request.urlretrieve(self.DNN_MODEL, str(model_path))
            except Exception as e:
                print(f"[FaceDetector] DNN model download failed: {e}")
                return

        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(
                str(proto_path), str(model_path)
            )
            self._backend = "dnn"
        except Exception as e:
            print(f"[FaceDetector] DNN load failed: {e}")

    def _load_haar(self) -> None:
        self.haar_cascade = cv2.CascadeClassifier(self.HAAR_PATH)
        if self.haar_cascade.empty():
            raise RuntimeError("Haar cascade failed to load.")
        self._backend = "haar"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.
        Returns list of (x, y, w, h) bounding boxes, sorted by size desc.
        """
        if self._backend == "dnn":
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.dnn_confidence:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            fw, fh = x2 - x1, y2 - y1
            if fw >= self.min_face_px and fh >= self.min_face_px:
                boxes.append((x1, y1, fw, fh))

        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_px, self.min_face_px)
        )
        if len(faces) == 0:
            return []
        return sorted(
            [tuple(f) for f in faces],
            key=lambda b: b[2] * b[3],
            reverse=True
        )

    # ------------------------------------------------------------------
    # Crop & preprocess
    # ------------------------------------------------------------------
    def extract_face_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.15
    ) -> np.ndarray:
        """
        Crop face region with margin, resize to face_size × face_size,
        convert to grayscale, apply CLAHE (contrast enhancement),
        return float32 array in [0, 1].
        """
        h, w = frame.shape[:2]
        x, y, fw, fh = bbox

        # Add margin
        mx = int(fw * margin)
        my = int(fh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.face_size, self.face_size), dtype=np.float32)

        # Grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # CLAHE — boosts local contrast (especially eyebrows, mouth corners)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Resize
        resized = cv2.resize(enhanced, (self.face_size, self.face_size),
                             interpolation=cv2.INTER_AREA)

        # Normalise to [0, 1]
        return resized.astype(np.float32) / 255.0

    def extract_facial_regions(
        self,
        face_crop: np.ndarray
    ) -> dict:
        """
        Extract sub-regions for optional analysis:
          - left_eye, right_eye (top ~35%)
          - mouth (bottom ~35%)
          - eyebrows (top ~20%)
        face_crop: (H, W) float32 [0,1]
        """
        H, W = face_crop.shape
        return {
            "left_eye":   face_crop[int(H * 0.15):int(H * 0.40),
                                    int(W * 0.45):int(W * 0.95)],
            "right_eye":  face_crop[int(H * 0.15):int(H * 0.40),
                                    int(W * 0.05):int(W * 0.55)],
            "mouth":      face_crop[int(H * 0.60):int(H * 0.90),
                                    int(W * 0.20):int(W * 0.80)],
            "eyebrows":   face_crop[int(H * 0.05):int(H * 0.22),
                                    int(W * 0.05):int(W * 0.95)],
        }

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        End-to-end: detect + crop largest face in frame.
        Returns (face_array, bbox) or (None, None) if no face found.
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None, None
        bbox = faces[0]  # largest face
        crop = self.extract_face_crop(frame, bbox)
        return crop, bbox


if __name__ == "__main__":
    detector = FaceDetector(model_dir="models", prefer_dnn=False)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop, bbox = detector.process_frame(frame)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
