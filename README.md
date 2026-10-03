# 🧠 CAG Emotion Tracker
### Cache-Augmented Generation for Real-Time Emotion Detection

A production-level AI system that detects human emotions from webcam input using a custom **Cache-Augmented Generation (CAG)** inference engine — achieving <50ms end-to-end latency without any external API, vector database, or retrieval pipeline.

---

## 📁 Project Structure

```
cag_emotion/
├── main.py                        # Entry point (OpenCV / Streamlit / benchmark)
├── dashboard.py                   # Streamlit web UI
├── requirements.txt
│
├── src/
│   ├── build_cache.py             # Build KV cache (run once)
│   ├── train.py                   # Training entry point
│   │
│   ├── modules/
│   │   ├── emotion_cnn.py         # Lightweight CNN (DS-Conv + SE attention)
│   │   ├── kv_cache.py            # KV Cache — core CAG data structure
│   │   ├── face_detector.py       # OpenCV face detection + crop pipeline
│   │   ├── cag_engine.py          # Full CAG inference engine (orchestrator)
│   │   └── trainer.py             # Training + cache update pipeline
│   │
│   └── utils/
│       ├── visualiser.py          # Real-time OpenCV overlay renderer
│       └── benchmark.py           # CAG vs baseline performance benchmark
│
├── cache/
│   └── emotion_cache.pt           # Serialised KV cache (auto-generated)
│
├── models/
│   └── emotion_cnn.pt             # Trained CNN weights (after training)
│
└── tests/
    ├── test_cache.py
    └── test_cnn.py
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the KV cache (run once)
```bash
python main.py --build_cache
```

### 3. Run real-time emotion detection
```bash
# OpenCV window (default)
python main.py

# Streamlit dashboard (web UI)
python main.py --streamlit

# Run performance benchmark
python main.py --benchmark
```

### 4. (Optional) Train on FER-2013 dataset
```bash
# Download FER-2013 from Kaggle, then:
python -m src.train --fer_csv fer2013.csv --epochs 50
```

---

## 🧠 How CAG Works

### Traditional RAG (what we DON'T use):
```
Query → Encode → HTTP call → Vector DB search → Top-K results → LLM → Answer
```
Each step adds latency. Vector DB alone adds 10–100ms.

### Our CAG approach:
```
Frame → FaceDetector → CNN embedding → dot product with KV cache → argmax → emotion
```
The KV cache is a **7 × 512 tensor pre-loaded into GPU VRAM**.
Inference = one matrix multiplication. No I/O, no retrieval, no network.

### KV Cache internals:
```
KEY   = prototype embedding of emotion class (precomputed, L2-normalised)
VALUE = emotion label + valence/arousal/dominance metadata

Lookup:
  similarity = query_embedding @ key_matrix.T    # (1, E) — one matmul
  attention  = softmax(similarity / temperature) # sharpened distribution
  emotion    = argmax(attention)                  # O(E) — 7 comparisons
```

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| End-to-end latency | ~15–40ms (CPU), ~5–12ms (GPU) |
| Face detection | ~3ms (Haar) / ~8ms (DNN) |
| CNN embedding | ~4–8ms (CPU) |
| KV cache lookup | <0.5ms |
| Temporal smoothing | <0.1ms |
| CAG vs naive baseline | **~200–500× faster lookup** |

---

## 🎭 Emotions Detected
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## 📊 CAG vs RAG Comparison

| Aspect | CAG (this project) | RAG |
|--------|-------------------|-----|
| Knowledge store | In-memory tensor (KB) | External vector DB (GB) |
| Per-query cost | 7 dot products | O(N) scan + HTTP round-trip |
| Latency overhead | <0.5ms | 10–200ms |
| Retrieval step | ❌ None | ✅ Required |
| Dynamic updates | EMA update (offline) | Real-time insertion |
| Scalability | Fixed E classes | Scales to millions |

**When to use CAG**: Fixed, small knowledge space where speed is critical.  
**When to use RAG**: Dynamic, large, heterogeneous knowledge bases.

---

## 🔬 Architecture Details

### EmotionCNN (~800K parameters)
- 4× Depthwise-Separable Conv blocks (MobileNet-style)
- Squeeze-and-Excitation attention (channel attention = implicit facial region focus)
- Global Average Pooling → 256-D
- Linear projection → 512-D L2-normalised embedding
- Separate classification head (training only; bypassed in CAG mode)

### Temporal Smoothing
- EMA with α=0.4: `state[t] = 0.4 × raw[t] + 0.6 × state[t-1]`
- Prevents emotion flickering between frames
- Auto-resets after 30 consecutive frames without a face

---

## 🧪 Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🔭 Future Improvements
1. **Hybrid CAG + RAG**: Use CAG for common emotions, trigger RAG for ambiguous/novel cases
2. **Personalised cache**: Fine-tune prototypes per user using federated learning
3. **Multi-face tracking**: Maintain separate temporal states per detected face
4. **Action Unit integration**: Explicit FACS AU detection as auxiliary features
5. **Context-aware smoothing**: Adaptive α based on prediction confidence variance
