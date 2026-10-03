# CAG Emotion Tracker

Real-time facial emotion recognition from a webcam, with a Streamlit dashboard. 66.5% accuracy on FER-2013 at about 5 ms per frame on a laptop CPU.

Detects 7 emotions: `angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

B.Tech mini project (3-person team), 2024–25.

## How it works

```
webcam frame → Haar face detector → 48×48 grayscale crop
            → CNN (4 VGG-style conv stages + squeeze-and-excitation attention, 1.3M params)
            → 512-D L2-normalised embedding
            → cache lookup: one matmul against a 7 × 512 table of emotion prototypes
            → exponential smoothing across frames (stops the label flickering)
            → overlay / dashboard
```

"Cache-augmented" means the classification step is a lookup into a small cache that sits in memory. Each prototype is the mean embedding of one emotion over the training set, built once after training. At inference, the label is the prototype with the highest cosine similarity: no classifier head, no retrieval index, no I/O. [docs/CAG_EXPLAINED.md](docs/CAG_EXPLAINED.md) has the details.

## Results

FER-2013 test set (7,178 held-out images, never used for training or checkpoint selection). Latency measured on an Apple M5 CPU.

| | Test accuracy | Per-frame latency |
|---|---|---|
| **CNN + prototype cache (default)** | **66.5%** | 4.9 ms mean · 5.2 ms p95 |
| CNN + classifier head | 66.3% | — |
| Baseline: hand-crafted features + MLP (`run.py`) | 38.2% | 4.8 ms |

For reference, chance is about 14% and human agreement on FER-2013 is about 65%. Most of the per-frame latency is face detection: the CNN takes 0.8 ms and the cache lookup 0.007 ms.

Training: 60 epochs, AdamW + cosine schedule, label smoothing, flip/rotation/brightness augmentation, 10% of the train split held out for checkpoint selection. About 20 minutes on an Apple M5 GPU (MPS). Full numbers are in [models/emotion_cnn.json](models/emotion_cnn.json).

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
python -m src.train --data fer2013 --epochs 60 --batch 128
```

This writes `models/emotion_cnn.pt`, rebuilds the prototype cache in `cache/emotion_cache.pt`, and saves test accuracy to `models/emotion_cnn.json`. It uses CUDA or Apple MPS when available.

The hand-crafted baseline has its own scripts: `train_on_fer2013.py` trains it, `calibrate.py` fits it to your face from webcam samples, and `python run.py` runs it. `models/fer_classifier.pkl` was saved with scikit-learn 1.7.2, which is why that version is pinned.

## Layout

```
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
