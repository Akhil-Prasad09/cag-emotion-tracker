"""
calibrate.py  —  Webcam-based emotion calibration
==================================================
Captures real face samples from YOUR webcam for each emotion,
extracts features, and saves them as prototypes directly into
a nearest-neighbour classifier.  No synthetic data involved.

Usage:
    python calibrate.py

For each of the 7 emotions you'll see a countdown, hold the
expression for 3 seconds while it captures 30 frames.
Done in ~2 minutes. Overwrites models/fer_classifier.pkl.
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from collections import defaultdict
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
CAPTURE_SECONDS = 3      # hold expression for this long
COUNTDOWN_SECONDS = 4    # prep time before capture starts
FRAMES_PER_EMOTION = 40  # samples to collect per emotion
IMG_SIZE = 48

EMOTION_INSTRUCTIONS = {
    "neutral":  "Relax your face completely. Look straight ahead.",
    "happy":    "Smile as big as you can. Raise your cheeks.",
    "sad":      "Droop your mouth corners. Raise your inner eyebrows.",
    "angry":    "Furrow your brows DOWN and together. Clench your jaw.",
    "surprise": "Raise your eyebrows HIGH. Open your eyes and mouth wide.",
    "fear":     "Raise eyebrows and pull them together. Tense your eyes.",
    "disgust":  "Wrinkle your nose. Raise one side of your upper lip.",
}

COLORS = {
    "neutral": (170,170,170), "happy": (0,210,210), "sad": (220,100,0),
    "angry": (0,0,220), "surprise": (0,180,255), "fear": (128,0,128),
    "disgust": (0,128,0),
}


def extract_features(img_f: np.ndarray) -> np.ndarray:
    """191-D feature vector — identical to sklearn_engine.py v2."""
    u8 = (img_f * 255).astype(np.uint8)
    lbp = local_binary_pattern(u8, P=8, R=1, method="uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    zm  = [mag[8:16,8:40], mag[16:24,8:22], mag[16:24,26:40],
           mag[30:44,14:34], mag[8:16,8:22], mag[22:30,16:32]]
    gf  = np.array([z.mean() for z in zm] + [z.std() for z in zm])
    zp  = [img_f[8:16,8:40], img_f[16:24,8:22], img_f[16:24,26:40],
           img_f[30:44,14:34], img_f[22:30,16:32]]
    pf  = np.array([z.mean() for z in zp] + [z.std() for z in zp])
    small = cv2.resize(u8,(16,8),interpolation=cv2.INTER_AREA).flatten().astype(np.float32)/255.
    mouth = img_f[31:43,14:34]
    mouth_rows = np.array([mouth[r].mean() for r in range(mouth.shape[0])])
    mc_l=img_f[35:40,14:18].mean(); mc_r=img_f[35:40,30:34].mean(); mc_c=img_f[35:40,21:27].mean()
    mouth_geom = np.array([mc_l,mc_r,mc_c,mc_l-mc_c,mc_r-mc_c])
    el_t=img_f[16:20,12:20].mean(); el_b=img_f[20:24,12:20].mean()
    er_t=img_f[16:20,28:36].mean(); er_b=img_f[20:24,28:36].mean()
    bl=img_f[8:14,10:22].mean(); br=img_f[8:14,26:38].mean()
    eye_geom = np.array([el_t,el_b,er_t,er_b,bl,br,bl-el_t,br-er_t])
    brow_gy      = np.abs(gy[8:18,8:40]).mean()
    brow_eye_gap = img_f[13:17,10:38].mean()
    ul_asym      = abs(img_f[30:35,14:22].mean()-img_f[30:35,26:34].mean())
    nose         = u8[22:32,17:31]
    nose_lbp     = local_binary_pattern(nose,P=8,R=1,method='uniform')
    nose_h,_     = np.histogram(nose_lbp,bins=6,range=(0,6),density=True)
    nose_h       = np.where(nose_h>0,nose_h,1e-9)
    nose_ent     = float(-(nose_h*np.log(nose_h)).sum())
    brow_dark_r  = img_f[14:19,17:31].mean() / (img_f[14:19,8:16].mean()+0.001)
    mouth_dark   = 1.0 - img_f[34:42,18:30].mean()
    extra = np.array([brow_gy,brow_eye_gap,ul_asym,nose_ent,brow_dark_r,mouth_dark],np.float32)
    return np.concatenate([lbp_h,gf,pf,small,mouth_rows,mouth_geom,eye_geom,extra])


def detect_face(frame, haar):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return None, None
    x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
    H,W = frame.shape[:2]
    mx,my = int(w*0.12), int(h*0.12)
    crop = gray[max(0,y-my):min(H,y+h+my), max(0,x-mx):min(W,x+w+mx)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    crop = clahe.apply(crop)
    crop = cv2.resize(crop,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
    return crop, (x,y,w,h)


def draw_ui(frame, emotion, phase, countdown, collected, total, instruction):
    h, w = frame.shape[:2]
    color = COLORS.get(emotion, (200,200,200))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 110), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, emotion.upper(), (15,48),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2)
    cv2.putText(frame, instruction, (15,72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    if phase == "countdown":
        msg = f"Get ready... {countdown}"
        cv2.putText(frame, msg, (15,98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
    elif phase == "capturing":
        bar_w = int((collected / total) * (w - 30))
        cv2.rectangle(frame, (15,88), (w-15,104), (40,40,40), -1)
        cv2.rectangle(frame, (15,88), (15+bar_w,104), color, -1)
        cv2.putText(frame, f"Capturing {collected}/{total}", (15,84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    elif phase == "done":
        cv2.putText(frame, f"✓ Captured {total} samples", (15,98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,100), 1)

    # Progress dots
    dot_x = 15
    for i, em in enumerate(EMOTIONS):
        c = (0,200,100) if EMOTIONS.index(emotion) > i else \
            color if EMOTIONS.index(emotion) == i else (60,60,60)
        cv2.circle(frame, (dot_x + i*30, h-15), 8, c, -1)
        cv2.putText(frame, em[0].upper(), (dot_x+i*30-5, h-11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
    return frame


def main():
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot open webcam"); return

    all_features = []
    all_labels   = []
    print("\n=== Emotion Calibration ===")
    print("A window will open. Follow the on-screen instructions.")
    print("Press Q at any time to quit.\n")

    for emotion in EMOTIONS:
        instruction = EMOTION_INSTRUCTIONS[emotion]
        print(f"→ {emotion.upper()}: {instruction}")

        # ── Countdown phase ──────────────────────────────────────
        t_start = time.time()
        while time.time() - t_start < COUNTDOWN_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            _, bbox = detect_face(frame, haar)
            if bbox:
                x,y,w,h = bbox
                cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)
            remaining = max(0, int(COUNTDOWN_SECONDS - (time.time()-t_start)) + 1)
            draw_ui(frame, emotion, "countdown", remaining, 0,
                    FRAMES_PER_EMOTION, instruction)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return

        # ── Capture phase ────────────────────────────────────────
        collected = 0
        t_start   = time.time()
        while collected < FRAMES_PER_EMOTION:
            ret, frame = cap.read()
            if not ret: break
            face, bbox = detect_face(frame, haar)
            color = COLORS[emotion]

            if face is not None and bbox is not None:
                x,y,w,h = bbox
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                feat = extract_features(face.astype(np.float32)/255.0)
                all_features.append(feat)
                all_labels.append(EMOTIONS.index(emotion))
                collected += 1

            draw_ui(frame, emotion, "capturing", 0, collected,
                    FRAMES_PER_EMOTION, instruction)
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
            # Throttle to ~10 fps during capture for variety
            time.sleep(0.08)

        # ── Done flash ───────────────────────────────────────────
        ret, frame = cap.read()
        if ret:
            draw_ui(frame, emotion, "done", 0, collected, FRAMES_PER_EMOTION, instruction)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(600)

        print(f"  ✓ Captured {collected} samples")

    cap.release()
    cv2.destroyAllWindows()

    # ── Train KNN on your real face data ─────────────────────────
    print("\nTraining personalised classifier on your data...")
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    print(f"  Samples: {X.shape[0]}  Features: {X.shape[1]}")

    model = Pipeline([
        ("sc",  StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=7,
            metric="cosine",
            weights="distance",
            algorithm="brute",
        )),
    ])
    model.fit(X, y)
    train_acc = (model.predict(X) == y).mean()
    print(f"  Train accuracy: {train_acc:.3f}")

    # Save — same path run.py uses
    out = Path("models")
    out.mkdir(exist_ok=True)
    with open(out/"fer_classifier.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)

    import json
    with open(out/"fer_classifier_meta.json","w") as f:
        json.dump({"emotions": EMOTIONS, "feature_dim": int(X.shape[1]),
                   "model_type": "KNN_personalised",
                   "samples_per_emotion": FRAMES_PER_EMOTION,
                   "train_acc": float(train_acc)}, f, indent=2)

    print(f"\n✓ Personalised model saved to models/fer_classifier.pkl")
    print("  Run:  python run.py")

    # Quick confusion check
    print("\nPer-emotion self-check:")
    from collections import Counter
    for ci, em in enumerate(EMOTIONS):
        mask  = y == ci
        preds = [EMOTIONS[p] for p in model.predict(X[mask])]
        top   = Counter(preds).most_common(1)[0]
        print(f"  {em:<10} → {top[0]:<10} {top[1]}/{mask.sum()}  "
              f"{'✓' if top[0]==em else '✗'}")


if __name__ == "__main__":
    main()
