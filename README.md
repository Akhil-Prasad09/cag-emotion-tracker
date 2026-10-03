# CAG Emotion Tracker

Real-time facial emotion recognition from a webcam, with a Streamlit dashboard. Runs on a laptop CPU at about 5 ms per frame.

Detects 7 emotions: `angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

B.Tech mini project (3-person team), 2024–25.

## How it works

```
webcam frame → Haar face detector → 48×48 grayscale crop
            → 191-D hand-crafted features (LBP texture, Sobel gradient zones, mouth/brow geometry)
            → MLP classifier (512 → 256), trained on FER-2013
            → exponential smoothing across frames (stops the label flickering)
            → overlay / dashboard
```

The repo also contains an experimental "cache-augmented" path (`main.py`, `src/modules/cag_engine.py`): a small PyTorch CNN produces an embedding, and the emotion comes from one matrix multiply against a cached 7 × 512 table of class prototypes instead of a classifier head or a retrieval step. That CNN ships **untrained**, so that path is for the architecture and the lookup benchmark, not for predictions. [docs/CAG_EXPLAINED.md](docs/CAG_EXPLAINED.md) walks through the idea.

## Results

Measured on the FER-2013 public test split (7,178 images) and an Apple M5 CPU:

| Metric | Value |
|---|---|
| Test accuracy (7 classes) | **38.2%** (chance ≈ 14%; human agreement on FER-2013 is ~65%) |
| Best classes (F1) | happy 0.53 · surprise 0.51 |
| Per-frame latency, face present | 4.8 ms mean · 5.1 ms p95 (detection + features + MLP) |

Hand-crafted features cap accuracy well below CNNs, which reach roughly 65–73% on this dataset. Training the CNN in `src/` is the obvious next step (see below).

## Run it

Needs Python 3.10–3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py                    # OpenCV window: Q quit · R reset smoothing · S screenshot
streamlit run dashboard.py       # web dashboard: live demo, benchmark, cache inspector
python run.py --benchmark        # latency benchmark
python -m pytest tests -q        # tests, including a headless dashboard check
```

macOS asks for camera permission the first time. Grant it to your terminal.

## Retrain

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and unzip it next to the scripts as `fer2013/train` and `fer2013/test`, then:

```bash
python train_on_fer2013.py       # ~3–5 min on CPU, overwrites models/fer_classifier.pkl
python calibrate.py              # optional: fit to your own face from 40 webcam frames per emotion
python -m src.train              # train the experimental CNN for the cache-augmented path
```

`models/fer_classifier.pkl` was saved with scikit-learn 1.7.2, which is why that version is pinned. If you change it, retrain.

## Layout

```
run.py                     OpenCV app (trained model)
dashboard.py               Streamlit dashboard (trained model)
main.py                    experimental CNN + prototype-cache pipeline
train_on_fer2013.py        trains models/fer_classifier.pkl on FER-2013
calibrate.py               per-user calibration from webcam samples
build_fer_model.py         original synthetic-face bootstrap model (superseded by train_on_fer2013.py)
src/modules/
  sklearn_engine.py        feature extraction + MLP inference + smoothing
  cag_engine.py            CNN → prototype-cache inference
  kv_cache.py              the 7-prototype cache
  emotion_cnn.py           depthwise-separable CNN with squeeze-and-excitation
  face_detector.py         Haar / OpenCV DNN face detection
  trainer.py               CNN training loop
src/utils/                 benchmark + OpenCV overlay
tests/
```
