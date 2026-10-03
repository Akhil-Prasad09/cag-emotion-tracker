"""
build_fer_model.py  (v2 — 191-D features, fixes angry/surprise + disgust/angry)
--------------------------------------------------------------------------------
Run once to generate models/fer_classifier.pkl:

    python build_fer_model.py

Requires: opencv-python, scikit-learn, scikit-image, numpy
No internet, no GPU, no PyTorch needed.
"""

import numpy as np
import cv2
import pickle
import json
import time
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE  = 48
rng       = np.random.RandomState(42)


# ── Synthetic face generator ──────────────────────────────────────────────────
def make_face_img(emotion: str, rng) -> np.ndarray:
    """
    Generate a 48×48 float32 [0,1] synthetic face with emotion-specific geometry.
    Based on FACS Action Units for each emotion:
      happy    AU6+12  — cheek raise + lip corner pull
      sad      AU1+4+15— inner brow raise + lip corner depress
      angry    AU4+5+7 — brow lowerer + upper lid raiser + lid tightener
      surprise AU1+2+5B— inner+outer brow raise + lid raiser
      fear     AU1+2+4 — combined brow raise and pull
      disgust  AU9+16  — nose wrinkle + upper lip raise (asymmetric)
      neutral  —        baseline, relaxed
    """
    img = np.full((IMG_SIZE, IMG_SIZE), 0.55, dtype=np.float32)
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    img[((X-24)/16)**2 + ((Y-24)/20)**2 < 1] += 0.05  # face oval

    if emotion == "happy":
        img[20:26, 8:15]  += 0.12          # raised cheeks (AU6)
        img[20:26, 33:40] += 0.12
        for i in range(-10, 11):            # U-shaped smile (AU12)
            cy = 36 + int(2.5 * (i/10)**2 * 5)
            if 0 <= cy < 48 and 0 <= 24+i < 48: img[cy, 24+i] = 0.08
        img[17:21, 10:20] = 0.12            # squinted eyes (AU6)
        img[17:21, 28:38] = 0.12
        img[18, 14:18]    = 0.06
        img[18, 30:34]    = 0.06

    elif emotion == "sad":
        for i in range(-9, 10):             # frown — inverted U (AU15)
            cy = 35 - int(2.0 * (i/9)**2 * 4)
            if 0 <= cy < 48 and 0 <= 24+i < 48: img[cy, 24+i] = 0.08
        img[13:16, 12:20] = 0.1             # raised inner brows (AU1)
        img[13:16, 28:36] = 0.1
        img[17:22, 12:22] = 0.12
        img[17:22, 26:36] = 0.12

    elif emotion == "angry":
        img[15:19, 10:22] = 0.08            # lowered brows (AU4)
        img[15:19, 26:38] = 0.08
        img[16:18, 16:22] = 0.04            # brow crease (inner pull)
        img[16:18, 26:30] = 0.04
        img[26:30, 20:28] = 0.04            # nose crease
        img[35:38, 16:32] = 0.08            # tight compressed lips
        img[17:22, 12:21] = 0.12
        img[17:22, 27:36] = 0.12
        # Darker inner brow region (crease shadow)
        img[14:18, 19:29] -= 0.06

    elif emotion == "surprise":
        img[9:13,  10:22] = 0.08            # raised brows high (AU1+2)
        img[9:13,  26:38] = 0.08
        img[17:24, 10:22] = 0.12            # wide open eyes (AU5B)
        img[17:24, 26:38] = 0.12
        img[16, 14:19]    = 0.04
        img[16, 29:34]    = 0.04
        cv2.ellipse(img, (24, 37), (8, 5), 0, 0, 360, 0.1, 2)  # O-mouth
        img[35:42, 18:30] = 0.3             # open mouth interior
        # Bright forehead strip = brows far from eyes
        img[13:17, 10:38] += 0.05

    elif emotion == "fear":
        img[11:15, 12:22] = 0.08            # raised + pulled brows (AU1+2+4)
        img[11:15, 26:36] = 0.08
        img[12:14, 19:29] = 0.04            # inner brow pull together
        img[16:23, 10:22] = 0.12            # wide eyes
        img[16:23, 26:38] = 0.12
        img[16:18, 14:18] = 0.04
        img[16:18, 30:34] = 0.04
        for i in range(-7, 8):              # slightly open mouth
            cy = 36 + int(1.0 * (i/7)**2 * 3)
            if 0 <= cy < 48 and 0 <= 24+i < 48: img[cy, 24+i] = 0.1
        img[37:41, 18:30] = 0.28

    elif emotion == "disgust":
        img[25:30, 19:27] = 0.08            # nose wrinkle (AU9)
        img[25:29, 20:24] = 0.04            # bridge crease
        img[31:34, 14:22] = 0.06            # raised upper lip LEFT (AU16, asymmetric)
        img[31:34, 22:30] = 0.10            # right side less raised = asymmetry
        for i in range(-8, 9):
            cy = 36 + int(1.2 * (i/8)**2 * 3)
            if 0 <= cy < 48 and 0 <= 24+i < 48: img[cy, 24+i] = 0.1
        img[15:20, 12:22] = 0.12
        img[15:20, 26:36] = 0.12

    else:  # neutral
        img[14:17, 11:22] = 0.1
        img[14:17, 26:37] = 0.1
        img[18:23, 12:22] = 0.12
        img[18:23, 26:36] = 0.12
        img[18:20, 15:19] = 0.05
        img[18:20, 29:33] = 0.05
        img[35:37, 16:32] = 0.1

    img += rng.randn(IMG_SIZE, IMG_SIZE).astype(np.float32) * 0.04
    return np.clip(img, 0, 1)


# ── 191-D feature extractor ───────────────────────────────────────────────────
def extract_features(img_f: np.ndarray) -> np.ndarray:
    """
    191-D compact descriptor.  MUST stay in sync with sklearn_engine.py.

    [0:10]   LBP histogram            texture
    [10:22]  Gradient zone stats      edge energy per facial region
    [22:32]  Pixel zone stats         regional brightness
    [32:160] 16×8 downsampled patch   global appearance
    [160:172] Mouth row means         smile/frown shape
    [172:177] Mouth corner geometry   smile marker
    [177:185] Eye/brow geometry       brow height & eye openness
    [185:191] New discriminative:
       brow_gy, brow_eye_gap, ul_asym, nose_entropy,
       brow_dark_ratio, mouth_dark
    """
    u8 = (img_f * 255).astype(np.uint8)

    lbp   = local_binary_pattern(u8, P=8, R=1, method="uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    gx  = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    zm  = [mag[8:16, 8:40], mag[16:24, 8:22], mag[16:24, 26:40],
           mag[30:44, 14:34], mag[8:16, 8:22], mag[22:30, 16:32]]
    gf  = np.array([z.mean() for z in zm] + [z.std() for z in zm])

    zp = [img_f[8:16, 8:40],  img_f[16:24, 8:22], img_f[16:24, 26:40],
          img_f[30:44, 14:34], img_f[22:30, 16:32]]
    pf = np.array([z.mean() for z in zp] + [z.std() for z in zp])

    small = cv2.resize(u8, (16, 8), interpolation=cv2.INTER_AREA
                       ).flatten().astype(np.float32) / 255.0

    mouth      = img_f[31:43, 14:34]
    mouth_rows = np.array([mouth[r].mean() for r in range(mouth.shape[0])])
    mc_l = img_f[35:40, 14:18].mean()
    mc_r = img_f[35:40, 30:34].mean()
    mc_c = img_f[35:40, 21:27].mean()
    mouth_geom = np.array([mc_l, mc_r, mc_c, mc_l - mc_c, mc_r - mc_c])

    el_t = img_f[16:20, 12:20].mean(); el_b = img_f[20:24, 12:20].mean()
    er_t = img_f[16:20, 28:36].mean(); er_b = img_f[20:24, 28:36].mean()
    bl   = img_f[8:14, 10:22].mean();  br   = img_f[8:14, 26:38].mean()
    eye_geom = np.array([el_t, el_b, er_t, er_b, bl, br, bl - el_t, br - er_t])

    # New discriminative features
    brow_gy       = np.abs(gy[8:18, 8:40]).mean()
    brow_eye_gap  = img_f[13:17, 10:38].mean()
    ul_asym       = abs(img_f[30:35, 14:22].mean() - img_f[30:35, 26:34].mean())
    nose          = u8[22:32, 17:31]
    nose_lbp      = local_binary_pattern(nose, P=8, R=1, method="uniform")
    nose_h, _     = np.histogram(nose_lbp, bins=6, range=(0, 6), density=True)
    nose_h        = np.where(nose_h > 0, nose_h, 1e-9)
    nose_ent      = float(-(nose_h * np.log(nose_h)).sum())
    inner_brow    = img_f[14:19, 17:31].mean()
    outer_brow    = img_f[14:19,  8:16].mean()
    brow_dark_r   = inner_brow / (outer_brow + 0.001)
    mouth_dark    = 1.0 - img_f[34:42, 18:30].mean()

    extra = np.array([brow_gy, brow_eye_gap, ul_asym, nose_ent,
                      brow_dark_r, mouth_dark], dtype=np.float32)

    return np.concatenate([lbp_h, gf, pf, small, mouth_rows,
                           mouth_geom, eye_geom, extra])


# ── Main training pipeline ────────────────────────────────────────────────────
def main():
    out = Path("models")
    out.mkdir(exist_ok=True)

    print("=" * 52)
    print("  Building FER Emotion Classifier  (v2, 191-D)")
    print("=" * 52)

    print("\n[1/3] Generating training data  (600 × 7 = 4200 samples)...")
    t0 = time.time()
    X, y = [], []
    for ci, em in enumerate(EMOTIONS):
        for _ in range(600):
            X.append(extract_features(make_face_img(em, rng)))
            y.append(ci)
        print(f"      {em}")
    X, y = np.array(X, dtype=np.float32), np.array(y)
    print(f"      Shape: {X.shape}  in {time.time()-t0:.1f}s")

    print("\n[2/3] Training MLP  (191 → 256 → 128 → 7)...")
    t0 = time.time()
    model = Pipeline([
        ("sc",  StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=500,
            learning_rate_init=0.001,
            batch_size=64,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"      Done in {time.time()-t0:.1f}s  |  train accuracy: {acc:.3f}")

    print("\n[3/3] Saving...")
    pkl_path = out / "fer_classifier.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f, protocol=4)
    import os
    meta = {"emotions": EMOTIONS, "feature_dim": int(X.shape[1]),
            "model_type": "MLP_v2_191feat", "train_acc": float(acc)}
    with open(out / "fer_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      {pkl_path}  ({os.path.getsize(pkl_path)//1024} KB)")

    print("\n── Per-emotion validation (50 samples) ──")
    tr = np.random.RandomState(99)
    from collections import Counter
    total = 0
    for em in EMOTIONS:
        preds = []
        for _ in range(50):
            p = model.predict_proba(extract_features(make_face_img(em, tr)).reshape(1,-1))[0]
            preds.append(EMOTIONS[p.argmax()])
        top = Counter(preds).most_common(1)[0]
        total += top[1]
        ok = "✓" if top[0] == em else "✗"
        print(f"  {em:<10} → {top[0]:<10} {top[1]}/50  {ok}")
    print(f"\n  Overall: {total}/{len(EMOTIONS)*50}  "
          f"({total/(len(EMOTIONS)*50)*100:.0f}%)")
    print("\nDone!  Run:  python run.py")


if __name__ == "__main__":
    main()
