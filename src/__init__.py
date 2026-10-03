"""
CAG Emotion Tracker — Source Package

Module layout:
  src/modules/      → unified modules (used by main.py)
    emotion_cnn.py  → EmotionCNN (lightweight DS-Conv + SE attention)
    kv_cache.py     → EmotionKVCache (simplified single-prototype-per-class)
    face_detector.py→ FaceDetector (Haar / DNN OpenCV)
    cag_engine.py   → CAGEngine (full orchestrator)
    trainer.py      → EmotionTrainer

  src/cache/        → original multi-prototype KV cache
    kv_cache.py     → EmotionKVCache + KVCacheBuilder (multi-prototype)

  src/models/       → original model
    emotion_cnn.py  → EmotionCNN (alternative implementation)

  src/utils/
    visualiser.py   → OpenCV overlay renderer
    benchmark.py    → CAG vs baseline benchmark
    feature_extractor.py → Visual feature extraction pipeline
"""
