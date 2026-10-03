"""
main.py
-------
Entry point for the real-time CAG Emotion Tracker.

Usage:
  python main.py                  # OpenCV window mode
  python main.py --streamlit      # Launch Streamlit web UI
  python main.py --benchmark      # Run benchmark only
  python main.py --source 0       # Webcam index (default 0)
  python main.py --source video.mp4

Controls (OpenCV mode):
  Q / ESC  — quit
  R        — reset temporal smoother
  B        — toggle score bars
  S        — save screenshot
"""

import argparse
import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="CAG Real-Time Emotion Tracker")
    p.add_argument("--source",     type=str, default="0",
                   help="Webcam index (0,1,...) or video file path")
    p.add_argument("--streamlit",  action="store_true",
                   help="Launch Streamlit dashboard")
    p.add_argument("--benchmark",  action="store_true",
                   help="Run performance benchmark only")
    p.add_argument("--build_cache",action="store_true",
                   help="Rebuild KV cache and exit")
    p.add_argument("--device",     type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--width",      type=int, default=640)
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--no_display", action="store_true",
                   help="Headless mode (for servers)")
    return p.parse_args()


def run_opencv(args):
    """OpenCV window-based real-time loop."""
    from src.modules.cag_engine import CAGEngine
    from src.utils.visualiser import RealtimeVisualiser

    print("[main] Initialising CAG engine...")
    engine = CAGEngine(
        cache_path="cache/emotion_cache.pt",
        model_path="models/emotion_cnn.pt",
        device=args.device,
    )
    engine.load()
    vis = RealtimeVisualiser(history_len=100)

    # Open video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {src}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer lag

    print("[main] Starting real-time loop. Press Q to quit.")
    screenshot_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CAG inference
        result = engine.infer(frame)

        if not args.no_display:
            # Draw overlays
            display = vis.draw(frame, result)
            cv2.imshow("CAG Emotion Tracker", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                engine.smoother.reset()
                print("[main] Temporal smoother reset.")
            elif key == ord("s"):
                fname = f"screenshot_{screenshot_idx:04d}.png"
                cv2.imwrite(fname, display)
                print(f"[main] Saved {fname}")
                screenshot_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Print perf summary
    stats = engine.perf_stats()
    print("\n=== Performance Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def run_streamlit():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)


def main():
    args = parse_args()

    # Create directories
    Path("cache").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if args.build_cache:
        from src.build_cache import main as build
        build()
        return

    if args.benchmark:
        from src.utils.benchmark import run_benchmark
        run_benchmark(n_trials=500)
        return

    if args.streamlit:
        run_streamlit()
        return

    run_opencv(args)


if __name__ == "__main__":
    main()
