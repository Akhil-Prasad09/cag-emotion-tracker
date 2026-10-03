"""
run.py  —  Main launcher for the CAG Emotion Tracker (sklearn backend)
=======================================================================
Usage:
  python run.py                     # webcam, OpenCV window
  python run.py --source video.mp4  # video file
  python run.py --build             # rebuild classifier and exit
  python run.py --benchmark         # latency benchmark

Controls (OpenCV window):
  Q / ESC  — quit
  R        — reset temporal smoother
  S        — save screenshot
"""

import argparse, sys, os, time
import cv2
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

# Emotion colour palette (BGR)
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   128, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   210, 210),
    "neutral":  (170, 170, 170),
    "sad":      (220, 100, 0),
    "surprise": (0,   180, 255),
}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def draw_overlay(frame, result):
    """Draw emotion overlay on frame."""
    h, w = frame.shape[:2]
    emotion    = result["emotion"]
    confidence = result["confidence"]
    smooth     = result["smooth_scores"]
    bbox       = result["bbox"]
    lat        = result["latency_ms"]
    fps        = result["fps"]
    face_found = result["face_found"]
    color      = EMOTION_COLORS.get(emotion, (200,200,200))

    # ── Face bounding box ──────────────────────────────────────
    if bbox and face_found:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x,y), (x+bw, y+bh), color, 2)
        L = min(bw,bh)//6
        for px,py in [(x,y),(x+bw-L,y),(x,y+bh-L),(x+bw-L,y+bh-L)]:
            cv2.rectangle(frame,(px,py),(px+L,py+4),color,-1)
            cv2.rectangle(frame,(px,py),(px+4,py+L),color,-1)

    # ── Top-left emotion panel ─────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (8,8), (275,100), (18,18,18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if face_found:
        cv2.putText(frame, emotion.upper(), (18,56),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2)
        bx, by = 18, 72
        blen = int(230 * confidence)
        cv2.rectangle(frame, (bx,by), (bx+230,by+14), (45,45,45), -1)
        cv2.rectangle(frame, (bx,by), (bx+blen,by+14), color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (bx+235,by+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    else:
        cv2.putText(frame, "No face", (18,56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 1)

    # ── Right-side score bars ──────────────────────────────────
    if smooth and face_found:
        xr = w - 120
        bar_max = 110
        y_base = h//2 + bar_max//2
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (xr-8, y_base-bar_max-24), (w-4, y_base+22),
                      (18,18,18), -1)
        cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
        bw2, gap = 13, 3
        for i, em in enumerate(EMOTIONS):
            sc   = smooth.get(em, 0.0)
            bh2  = int(sc * bar_max)
            bx2  = xr + i * (bw2+gap)
            c2   = EMOTION_COLORS.get(em, (180,180,180))
            cv2.rectangle(frame,(bx2,y_base-bar_max),(bx2+bw2,y_base),(55,55,55),1)
            if bh2 > 0:
                cv2.rectangle(frame,(bx2,y_base-bh2),(bx2+bw2,y_base),c2,-1)
            cv2.putText(frame, em[:3], (bx2-1,y_base+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, c2, 1)

    # ── Perf overlay ──────────────────────────────────────────
    lat_c = (0,210,0) if lat < 50 else (0,165,255) if lat < 100 else (0,0,220)
    cv2.putText(frame, f"Latency {lat:.1f}ms", (10,h-38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, lat_c, 1)
    cv2.putText(frame, f"FPS {fps:.1f}", (10,h-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, "CAG | MLP", (10,h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70,70,70), 1)
    return frame


def build_model():
    """Run model training script."""
    import subprocess
    script = Path(__file__).parent / "build_fer_model.py"
    if not script.exists():
        # Write it inline
        _write_build_script(script)
    subprocess.run([sys.executable, str(script)], check=True)


def run_opencv(source, no_display=False):
    from src.modules.sklearn_engine import SklearnEmotionEngine

    engine = SklearnEmotionEngine(
        model_path="models/fer_classifier.pkl",
        temporal_alpha=0.45,
    )
    engine.load()

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {src}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    emotion_history = deque(maxlen=100)
    screenshot_idx = 0
    print("Running... Press Q to quit, R to reset smoother, S to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine.infer(frame)
        if result["face_found"]:
            emotion_history.append(result["emotion"])

        if not no_display:
            display = draw_overlay(frame.copy(), result)

            # Emotion timeline strip
            if emotion_history:
                bh2 = frame.shape[0]
                strip_y = bh2 - 26
                cw = max(1, frame.shape[1] // max(len(emotion_history), 1))
                for i, em in enumerate(emotion_history):
                    c = EMOTION_COLORS.get(em, (100,100,100))
                    cv2.rectangle(display, (i*cw, strip_y),
                                  (i*cw+cw, strip_y+18), c, -1)

            cv2.imshow("CAG Emotion Tracker", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                engine._smooth_state = None
                print("Smoother reset.")
            elif key == ord('s'):
                fname = f"screenshot_{screenshot_idx:04d}.png"
                cv2.imwrite(fname, display)
                print(f"Saved {fname}")
                screenshot_idx += 1

    cap.release()
    if not no_display:
        cv2.destroyAllWindows()

    stats = engine.perf_stats()
    print("\n=== Session Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if emotion_history:
        from collections import Counter
        counts = Counter(emotion_history)
        print("\nEmotion distribution:")
        for em, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(cnt / max(counts.values()) * 20)
            print(f"  {em:<10} {bar} {cnt}")


def run_benchmark():
    from src.modules.sklearn_engine import SklearnEmotionEngine
    import numpy as np

    engine = SklearnEmotionEngine(model_path="models/fer_classifier.pkl")
    engine.load()

    print("\nBenchmarking with synthetic frames...")
    N = 200
    dummy = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    latencies = []
    for i in range(N):
        r = engine.infer(dummy)
        latencies.append(r["latency_ms"])
    lats = np.array(latencies)
    print(f"  N={N} frames")
    print(f"  Mean latency:  {lats.mean():.2f}ms")
    print(f"  P50:           {np.percentile(lats,50):.2f}ms")
    print(f"  P95:           {np.percentile(lats,95):.2f}ms")
    print(f"  Max FPS:       {1000/lats.mean():.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    default="0",  help="Webcam index or video path")
    p.add_argument("--build",     action="store_true", help="Rebuild model")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--no_display",action="store_true")
    args = p.parse_args()

    Path("models").mkdir(exist_ok=True)

    if args.build:
        build_model()
        return
    if not Path("models/fer_classifier.pkl").exists():
        print("Model not found. Building now...")
        build_model()
    if args.benchmark:
        run_benchmark()
        return
    run_opencv(args.source, args.no_display)


if __name__ == "__main__":
    main()
