"""
train_on_fer2013.py
===================
Trains the emotion classifier on the REAL FER2013 dataset.
This replaces the synthetic training and fixes the misclassification problem.

STEP 1 — Download the dataset (free, one-time):
  Go to: https://www.kaggle.com/datasets/msambare/fer2013
  Click "Download" (you need a free Kaggle account)
  You will get a zip containing:
    train/
      angry/    *.jpg   (3995 images)
      disgust/  *.jpg   (436 images)
      fear/     *.jpg   (4097 images)
      happy/    *.jpg   (7215 images)
      neutral/  *.jpg   (4965 images)
      sad/      *.jpg   (4830 images)
      surprise/ *.jpg   (3171 images)
    test/
      (same structure, smaller)

STEP 2 — Place the unzipped folder next to this script:
  your_project/
    train_on_fer2013.py    ← this file
    fer2013/               ← the unzipped dataset
      train/
      test/

STEP 3 — Run:
  python train_on_fer2013.py

  Takes about 3-5 minutes on CPU.
  Overwrites models/fer_classifier.pkl when done.

STEP 4 — Run your project normally:
  python run.py
"""

import cv2
import numpy as np
import pickle
import json
import time
import os
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "fer2013"          # folder containing train/ and test/
OUTPUT_MODEL = "models/fer_classifier.pkl"
MAX_PER_CLASS = 2000              # cap per class so training stays fast on CPU
                                  # increase to 4000+ for higher accuracy

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE  = 48


# ── Feature extractor (191-D, identical to sklearn_engine.py) ─────────────────
def extract_features(img_f: np.ndarray) -> np.ndarray:
    u8 = (img_f * 255).astype(np.uint8)
    lbp = local_binary_pattern(u8, P=8, R=1, method="uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    gx  = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    zm  = [mag[8:16,8:40], mag[16:24,8:22], mag[16:24,26:40],
           mag[30:44,14:34], mag[8:16,8:22], mag[22:30,16:32]]
    gf  = np.array([z.mean() for z in zm] + [z.std() for z in zm])
    zp  = [img_f[8:16,8:40], img_f[16:24,8:22], img_f[16:24,26:40],
           img_f[30:44,14:34], img_f[22:30,16:32]]
    pf  = np.array([z.mean() for z in zp] + [z.std() for z in zp])
    small = cv2.resize(u8, (16,8), interpolation=cv2.INTER_AREA
                      ).flatten().astype(np.float32) / 255.0
    mouth = img_f[31:43, 14:34]
    mouth_rows = np.array([mouth[r].mean() for r in range(mouth.shape[0])])
    mc_l = img_f[35:40,14:18].mean(); mc_r = img_f[35:40,30:34].mean()
    mc_c = img_f[35:40,21:27].mean()
    mouth_geom = np.array([mc_l, mc_r, mc_c, mc_l-mc_c, mc_r-mc_c])
    el_t=img_f[16:20,12:20].mean(); el_b=img_f[20:24,12:20].mean()
    er_t=img_f[16:20,28:36].mean(); er_b=img_f[20:24,28:36].mean()
    bl=img_f[8:14,10:22].mean();    br=img_f[8:14,26:38].mean()
    eye_geom = np.array([el_t,el_b,er_t,er_b,bl,br,bl-el_t,br-er_t])
    brow_gy      = np.abs(gy[8:18,8:40]).mean()
    brow_eye_gap = img_f[13:17,10:38].mean()
    ul_asym      = abs(img_f[30:35,14:22].mean() - img_f[30:35,26:34].mean())
    nose         = u8[22:32,17:31]
    nose_lbp     = local_binary_pattern(nose, P=8, R=1, method="uniform")
    nose_h, _    = np.histogram(nose_lbp, bins=6, range=(0,6), density=True)
    nose_h       = np.where(nose_h > 0, nose_h, 1e-9)
    nose_ent     = float(-(nose_h * np.log(nose_h)).sum())
    brow_dark_r  = img_f[14:19,17:31].mean() / (img_f[14:19,8:16].mean() + 0.001)
    mouth_dark   = 1.0 - img_f[34:42,18:30].mean()
    extra = np.array([brow_gy, brow_eye_gap, ul_asym, nose_ent,
                      brow_dark_r, mouth_dark], dtype=np.float32)
    return np.concatenate([lbp_h, gf, pf, small, mouth_rows,
                           mouth_geom, eye_geom, extra])


def load_split(split_dir: Path, max_per_class: int) -> tuple:
    """Load images from train/ or test/ folder structure."""
    X, y = [], []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    for ci, emotion in enumerate(EMOTIONS):
        emotion_dir = split_dir / emotion
        if not emotion_dir.exists():
            print(f"  WARNING: {emotion_dir} not found, skipping")
            continue

        images = sorted(emotion_dir.glob("*.jpg")) + \
                 sorted(emotion_dir.glob("*.png")) + \
                 sorted(emotion_dir.glob("*.jpeg"))

        # Shuffle for random subset selection
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(images))[:max_per_class]
        images  = [images[i] for i in indices]

        loaded = 0
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = clahe.apply(img)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img_f = img.astype(np.float32) / 255.0
            X.append(extract_features(img_f))
            y.append(ci)
            loaded += 1

        print(f"    {emotion:<10} {loaded:>4} images")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("=" * 54)
    print("  Train Emotion Classifier on FER2013  (real data)")
    print("=" * 54)

    # ── Validate dataset path ──────────────────────────────────
    dataset_path = Path(DATASET_DIR)
    train_dir    = dataset_path / "train"
    test_dir     = dataset_path / "test"

    if not train_dir.exists():
        print(f"\nERROR: Dataset not found at '{DATASET_DIR}/train/'")
        print("\nTo download FER2013:")
        print("  1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Sign in / create a free account")
        print("  3. Click the Download button")
        print("  4. Unzip the file")
        print(f"  5. Place the unzipped folder here as '{DATASET_DIR}/'")
        print("  6. Re-run this script\n")
        return

    # ── Load training data ─────────────────────────────────────
    print(f"\n[1/3] Loading training data (max {MAX_PER_CLASS}/class)...")
    t0 = time.time()
    X_train, y_train = load_split(train_dir, MAX_PER_CLASS)
    print(f"  Loaded {len(X_train)} training samples in {time.time()-t0:.1f}s")
    print(f"  Feature dim: {X_train.shape[1]}")

    # ── Load test data ─────────────────────────────────────────
    X_test, y_test = None, None
    if test_dir.exists():
        print(f"\n  Loading test data...")
        X_test, y_test = load_split(test_dir, 500)
        print(f"  Loaded {len(X_test)} test samples")

    # ── Class distribution ─────────────────────────────────────
    print("\n  Class distribution:")
    for ci, em in enumerate(EMOTIONS):
        n = (y_train == ci).sum()
        bar = "█" * (n // 50)
        print(f"    {em:<10} {n:>4}  {bar}")

    # ── Train ──────────────────────────────────────────────────
    print(f"\n[2/3] Training MLP (191 → 512 → 256 → 7)...")
    t0 = time.time()
    model = Pipeline([
        ("sc",  StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            max_iter=300,
            learning_rate_init=0.001,
            batch_size=128,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=True,
        )),
    ])
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    train_acc = (model.predict(X_train) == y_train).mean()
    print(f"\n  Training time:    {train_time:.1f}s")
    print(f"  Train accuracy:   {train_acc:.3f}")

    if X_test is not None:
        test_acc = (model.predict(X_test) == y_test).mean()
        print(f"  Test accuracy:    {test_acc:.3f}")
        print(f"\n  Note: FER2013 is a hard dataset.")
        print(f"  Human accuracy on FER2013 is ~65%.")
        print(f"  Best deep CNNs reach ~75%.")
        print(f"  HOG+MLP is expected to land around 55-65%.")
    else:
        test_acc = None

    # ── Per-class accuracy on test set ─────────────────────────
    if X_test is not None:
        print("\n  Per-class test accuracy:")
        from collections import Counter
        for ci, em in enumerate(EMOTIONS):
            mask = y_test == ci
            if mask.sum() == 0:
                continue
            preds = model.predict(X_test[mask])
            acc   = (preds == ci).mean()
            bar   = "█" * int(acc * 20)
            print(f"    {em:<10} {acc:.2f}  {bar}")

    # ── Save ───────────────────────────────────────────────────
    print(f"\n[3/3] Saving model...")
    out = Path("models")
    out.mkdir(exist_ok=True)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(model, f, protocol=4)

    meta = {
        "emotions":    EMOTIONS,
        "feature_dim": int(X_train.shape[1]),
        "model_type":  "MLP_512_256_fer2013",
        "train_acc":   float(train_acc),
        "test_acc":    float(test_acc) if test_acc is not None else None,
        "trained_on":  "FER2013_real_data",
        "samples":     int(len(X_train)),
    }
    with open("models/fer_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_kb = os.path.getsize(OUTPUT_MODEL) // 1024
    print(f"  Saved: {OUTPUT_MODEL}  ({size_kb} KB)")
    print(f"\n  Done! Run:  python run.py")


if __name__ == "__main__":
    main()
