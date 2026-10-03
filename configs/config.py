# configs/config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for CAG Emotion Detection System
# ─────────────────────────────────────────────────────────────

import torch

# ── Device ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Emotion Classes ──────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_EMOTIONS = len(EMOTIONS)

# Emotion → emoji mapping for UI
EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# ── Model ────────────────────────────────────────────────────
INPUT_SIZE        = (48, 48)      # grayscale face patch size
FEATURE_DIM       = 256           # CNN feature vector dimension
BACKBONE          = "lightweight" # "lightweight" or "inception"
MODEL_CHECKPOINT  = "models/emotion_cnn.pth"

# ── KV Cache ─────────────────────────────────────────────────
CACHE_PATH           = "cache/emotion_kv_cache.pt"
CACHE_NUM_PROTOTYPES = 20         # prototypes per emotion class
CACHE_EMBED_DIM      = FEATURE_DIM

# ── Inference ────────────────────────────────────────────────
SIMILARITY_METRIC     = "cosine"  # "cosine" or "dot"
CONFIDENCE_THRESHOLD  = 0.35      # below this → "uncertain"
TEMPORAL_WINDOW       = 5         # frames for smoothing
TEMPORAL_ALPHA        = 0.6       # EMA weight for current frame

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX     = 0
TARGET_FPS       = 30
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# ── Face Detection ────────────────────────────────────────────
HAARCASCADE_PATH = "models/haarcascade_frontalface_default.xml"
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE      = (60, 60)

# ── Benchmark ────────────────────────────────────────────────
BENCHMARK_FRAMES = 300   # frames to measure for perf report
