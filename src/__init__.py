"""
CAG Emotion Tracker — Source Package

Module layout:
  src/modules/      → unified modules (used by main.py)
    emotion_cnn.py  → EmotionCNN (lightweight DS-Conv + SE attention)
    kv_cache.py     → EmotionKVCache (simplified single-prototype-per-class)
    face_detector.py→ FaceDetector (Haar / DNN OpenCV)
    cag_engine.py   → CAGEngine (full orchestrator)
    sklearn_engine.py → SklearnEmotionEngine (trained FER-2013 model, used by run.py + dashboard)
    trainer.py      → EmotionTrainer

  src/utils/
    visualiser.py   → OpenCV overlay renderer
    benchmark.py    → CAG vs baseline benchmark
"""
