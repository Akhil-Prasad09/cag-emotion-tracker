# CAG Emotion Tracker

Real-time facial emotion recognition from a webcam, with a Streamlit dashboard. 70.6% accuracy on FER-2013 at about 8 ms per frame on a laptop CPU.

**[Try the live demo](https://akhil-prasad09.github.io/cag-emotion-tracker/)**: runs entirely in your browser (webcam or a photo), nothing is uploaded.

Detects 7 emotions: `angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

B.Tech mini project (3-person team), 2024–25.

## How it works

```
webcam frame → Haar face detector → 48×48 grayscale crop
            → CNN (4 VGG-style conv stages + squeeze-and-excitation attention, 5M params)
            → 512-D L2-normalised embedding, averaged with the mirrored face's embedding
            → cache lookup: one matmul against a 7 × 512 table of emotion prototypes
            → exponential smoothing across frames (stops the label flickering)
            → overlay / dashboard
```

"Cache-augmented" means the classification step is a lookup into a small cache that sits in memory. Each prototype is the mean embedding of one emotion over the training set, built once after training. At inference, the label is the prototype with the highest cosine similarity: no classifier head, no retrieval index, no I/O. [docs/CAG_EXPLAINED.md](docs/CAG_EXPLAINED.md) has the details.

## Results

FER-2013 test set (7,178 held-out images, never used for training or checkpoint selection). Latency measured on an Apple M5 CPU.

| | Test accuracy | Per-frame latency |
|---|---|---|
| **CNN + prototype cache, mirror-averaged (default)** | **70.6%** | 8.2 ms mean · 8.6 ms p95 |
| CNN + prototype cache, single pass | 69.7% | — |
| CNN + classifier head, mirror-averaged | 70.6% | — |
| Baseline: hand-crafted features + MLP (`run.py`) | 38.2% | 4.8 ms |

For reference, chance is about 14% and human agreement on FER-2013 is about 65%. Per-class recall ranges from 47% (fear) to 89% (happy). Each CNN pass takes about 2 ms, the cache lookup 0.007 ms, and face detection most of the rest.

Training: 80 epochs, AdamW + cosine schedule, label smoothing, augmentation (flip, rotation/shift/scale, brightness/contrast, random erasing), 10% of the train split held out for checkpoint selection. About 47 minutes on an Apple M5 GPU (MPS). Full numbers are in [models/emotion_cnn.json](models/emotion_cnn.json).

## Browser demo

[`web/`](web/) runs the model client-side: MediaPipe face detection, the CNN exported to ONNX (ONNX Runtime Web, WebGPU with a WASM fallback), mirror averaging, the prototype-cache lookup and smoothing, all in plain JavaScript with no build step.

Face framing is the main difference from the desktop app: MediaPipe's face box crops faces differently from how FER-2013 was cropped. So the demo uses a second model trained with stronger shift/zoom augmentation (`python -m src.train --data fer2013 --epochs 100 --batch 128 --strong-jitter`), which handles that better. On a class-balanced sample of 1,305 test faces run through the browser pipeline:

| Model | MediaPipe crop (what the demo sees) | Exact FER crop | Full test set |
|---|---|---|---|
| Default (`models/emotion_cnn.pt`) | 60.5% | 66.0% | 70.6% |
| **Web (`models/emotion_cnn_web.pt`)** | **62.6%** | 66.5% | 70.1% |

Two preprocessing choices were measured, not guessed: area-averaged downscaling like `cv2.INTER_AREA` (the canvas's built-in downscaling cost about 3 points), and a 10% crop margin, picked from 5–15% on training-split faces. `python web/export.py --check-test fer2013/test` re-exports the web model and confirms the export matches PyTorch (max difference under 1e-7) and still scores 70.1%.

## Run it

Needs Python 3.10–3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                   # OpenCV window: Q quit · R reset smoothing · S screenshot
streamlit run dashboard.py       # web dashboard: live demo, benchmark, cache inspector
python main.py --benchmark       # cache lookup vs brute-force scan benchmark
python -m pytest tests -q        # tests, including a headless dashboard check
```

macOS asks for camera permission the first time. Grant it to your terminal.

## Retrain

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and unzip it as `fer2013/train` and `fer2013/test`, then:

```bash
python -m src.train --data fer2013 --epochs 80 --batch 128
```

This writes `models/emotion_cnn.pt`, rebuilds the prototype cache in `cache/emotion_cache.pt`, and saves test accuracy to `models/emotion_cnn.json`. It uses CUDA or Apple MPS when available.

The hand-crafted baseline has its own scripts: `train_on_fer2013.py` trains it, `calibrate.py` fits it to your face from webcam samples, and `python run.py` runs it. `models/fer_classifier.pkl` was saved with scikit-learn 1.7.2, which is why that version is pinned.

## Layout

```
web/                       browser demo (GitHub Pages) + ONNX export script
models/emotion_cnn_web.pt  framing-robust model used by the browser demo
main.py                    OpenCV app (CNN + prototype cache)
dashboard.py               Streamlit dashboard (CNN + prototype cache)
src/train.py               training entry point
src/modules/
  emotion_cnn.py           the CNN
  kv_cache.py              the 7-prototype cache and its lookup
  cag_engine.py            detect → embed → lookup → smooth, per frame
  face_detector.py         Haar / OpenCV DNN face detection
  trainer.py               training loop + prototype rebuild
  sklearn_engine.py        hand-crafted-feature baseline
src/utils/                 benchmark + OpenCV overlay
run.py, train_on_fer2013.py, calibrate.py, build_fer_model.py   baseline scripts
models/                    trained weights + metrics
tests/
```
