# app.py
# ─────────────────────────────────────────────────────────────
# Streamlit Web UI for CAG Emotion Detection
#
# Run: streamlit run app.py
#
# Features:
#  • Live webcam feed with emotion overlay
#  • Emotion probability bar chart (live updating)
#  • Latency / FPS dashboard
#  • Benchmark tab
#  • Architecture explainer tab
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from configs.config import (
    CACHE_PATH, CONFIDENCE_THRESHOLD, DEVICE, EMOTION_EMOJI,
    EMOTIONS, FEATURE_DIM, MODEL_CHECKPOINT, SIMILARITY_METRIC,
    TEMPORAL_ALPHA, TEMPORAL_WINDOW,
)
from src.cache.kv_cache import EmotionKVCache, KVCacheBuilder
from src.cag_engine import CAGInferenceEngine
from src.models.emotion_cnn import EmotionCNN

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "CAG Emotion Tracker",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
  }
  .big-emotion {
    font-size: 3rem;
    text-align: center;
    margin: 0.5rem 0;
  }
  .stProgress > div > div { border-radius: 4px; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

EMOTION_COLORS_HEX = {
    "angry":    "#dc2626",
    "disgust":  "#16a34a",
    "fear":     "#7c3aed",
    "happy":    "#16a34a",
    "neutral":  "#6b7280",
    "sad":      "#2563eb",
    "surprise": "#0891b2",
    "uncertain": "#4b5563",
    "no_face":  "#374151",
}


# ─────────────────────────────────────────────────────────────
# Cached resource initialisation
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    """Initialise model + cache + engine once per session."""
    # Model
    model = EmotionCNN(num_classes=len(EMOTIONS), feature_dim=FEATURE_DIM)
    ckpt  = Path(MODEL_CHECKPOINT)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(str(ckpt), map_location=DEVICE, weights_only=True)
        )

    # KV Cache
    cache_path = Path(CACHE_PATH)
    if cache_path.exists():
        kv_cache = EmotionKVCache.load(str(cache_path), DEVICE)
    else:
        builder  = KVCacheBuilder(EMOTIONS, FEATURE_DIM, DEVICE)
        kv_cache = builder.build_synthetic()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        kv_cache.save(str(cache_path))

    engine = CAGInferenceEngine(
        model                = model,
        kv_cache             = kv_cache,
        emotion_emoji        = EMOTION_EMOJI,
        device               = DEVICE,
        similarity_metric    = SIMILARITY_METRIC,
        confidence_threshold = CONFIDENCE_THRESHOLD,
        temporal_window      = TEMPORAL_WINDOW,
        temporal_alpha       = TEMPORAL_ALPHA,
    )
    return engine


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
def tab_live(engine: CAGInferenceEngine):
    """Live webcam inference tab."""
    st.markdown("## 📷 Live Emotion Detection")
    col_video, col_stats = st.columns([3, 2])

    with col_stats:
        st.markdown("### Current Emotion")
        emo_placeholder   = st.empty()
        conf_placeholder  = st.empty()
        chart_placeholder = st.empty()
        st.divider()
        st.markdown("### Performance")
        fps_ph       = st.empty()
        latency_ph   = st.empty()
        frames_ph    = st.empty()

    with col_video:
        video_ph = st.image([])
        run_btn  = st.toggle("▶ Start Webcam", key="run_webcam")

    # History buffer for timeline
    history_emotions = []
    history_latencies = []

    if not run_btn:
        st.info("Toggle **Start Webcam** to begin.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Check permissions or camera index.")
        return

    engine.reset()
    stop_btn = st.button("⏹ Stop")

    while run_btn and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("Frame grab failed.")
            break

        result = engine.infer(frame)
        fps    = engine.current_fps()

        # Draw bounding box on frame
        display = frame.copy()
        if result.face_detected and result.bbox:
            x, y, bw, bh = result.bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 120), 2)
            cv2.putText(display,
                        f"{result.emotion} {result.confidence:.0%}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 120), 2, cv2.LINE_AA)

        video_ph.image(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True,
        )

        # Emotion display
        emo_color = EMOTION_COLORS_HEX.get(result.emotion, "#888")
        emo_placeholder.markdown(
            f'<div class="big-emotion">'
            f'{result.emoji} <span style="color:{emo_color}">'
            f'{result.emotion.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        conf_placeholder.progress(
            min(int(result.confidence * 100), 100),
            text=f"Confidence: {result.confidence:.1%}"
        )

        # Prob bar chart
        emos  = list(result.class_probs.keys())
        probs = list(result.class_probs.values())
        colors = [EMOTION_COLORS_HEX.get(e, "#888") for e in emos]
        fig = go.Figure(go.Bar(
            x=probs, y=emos, orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            xaxis=dict(range=[0, 1], showgrid=False),
            yaxis=dict(showgrid=False),
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True,
                                       key=f"chart_{result.frame_idx}")

        fps_ph.metric("FPS",     f"{fps:.1f}")
        latency_ph.metric("Latency", f"{result.latency_ms:.1f} ms")
        frames_ph.metric("Frames",   result.frame_idx)

        history_emotions.append(result.emotion)
        history_latencies.append(result.latency_ms)

    cap.release()

    # Post-session latency chart
    if history_latencies:
        st.markdown("### Session Latency")
        fig2 = go.Figure(go.Scatter(
            y=history_latencies, mode="lines",
            line=dict(color="#89b4fa", width=1.5),
            fill="tozeroy", fillcolor="rgba(137,180,250,0.15)",
        ))
        fig2.add_hline(y=50, line_dash="dash",
                       line_color="#f38ba8",
                       annotation_text="50ms target")
        fig2.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            xaxis_title="Frame", yaxis_title="ms",
        )
        st.plotly_chart(fig2, use_container_width=True)


def tab_benchmark():
    """Benchmark tab."""
    st.markdown("## ⚡ CAG vs RAG Benchmark")
    st.markdown("""
    This tab runs a **headless benchmark** comparing:
    - **CAG** (this system): single matrix multiply in shared memory
    - **Simulated RAG**: serialise → IPC delay → linear scan → deserialise
    """)

    col1, col2 = st.columns(2)
    n_queries  = col1.slider("Number of queries", 100, 2000, 500)
    net_delay  = col2.slider("Simulated network delay (ms)", 0.5, 5.0, 1.0)

    if st.button("▶ Run Benchmark"):
        with st.spinner("Benchmarking…"):
            from src.benchmark import run_benchmark
            results = run_benchmark(
                n_queries=n_queries,
                network_delay_ms=net_delay,
                device=DEVICE,
            )

        c = results["cag"]
        r = results["rag"]

        st.success(f"🚀 CAG is **{results['speedup']:.1f}×** faster than RAG")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("CAG mean latency",  f"{c['mean_ms']:.3f} ms")
        col_b.metric("RAG mean latency",  f"{r['mean_ms']:.3f} ms")
        col_c.metric("Speedup",           f"{results['speedup']:.1f}×")

        # Comparison chart
        fig = go.Figure()
        fig.add_bar(name="CAG",
                    x=["Mean", "P95", "P99"],
                    y=[c["mean_ms"], c["p95_ms"], c["p99_ms"]],
                    marker_color="#a6e3a1")
        fig.add_bar(name="RAG (sim)",
                    x=["Mean", "P95", "P99"],
                    y=[r["mean_ms"], r["p95_ms"], r["p99_ms"]],
                    marker_color="#f38ba8")
        fig.update_layout(
            barmode="group", height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            yaxis_title="Latency (ms)",
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_explainer():
    """Architecture / theory explainer tab."""
    st.markdown("## 🧠 How CAG Works in This Project")
    st.markdown("""
### Cache-Augmented Generation (CAG) vs Retrieval-Augmented Generation (RAG)

| | **RAG** | **CAG (this system)** |
|---|---|---|
| Knowledge storage | External vector DB | In-memory tensor |
| Retrieval step | Yes (FAISS/Chroma/Pinecone) | No |
| Network hop | Yes | No |
| Latency | 5–50 ms | <1 ms |
| Knowledge update | Dynamic | Batch (rebuild cache) |
| Context window | Unlimited (paginated) | Fixed (cache size) |

---

### KV Cache Mechanics

```
KEY  [K_i ∈ ℝ^256]  → L2-normalised prototype feature vector for emotion i
VALUE [V_i ∈ ℤ]     → integer class label (0=angry … 6=surprise)

At inference:
  Q = CNN(face_crop)            # query: live feature vector [256]
  S = Q @ K.T                   # similarities: [N_total]  ← single BLAS op
  top-k = argsort(S)[-k:]       # top-k matching prototypes
  class_probs = softmax(         # aggregate votes by class
      scatter_add(S[top-k], V[top-k])
  )
  prediction = argmax(class_probs)
```

The **entire retrieval** is one matrix multiply.  
No serialisation. No IPC. No index traversal.

---

### Temporal Smoothing

To prevent flickering between frames:
1. **EMA**: `smoothed_t = α × current_t + (1-α) × smoothed_{t-1}`
2. **Majority vote**: only emit a new label if it wins the last N frames

---

### Limitations

- **Static knowledge**: cache must be rebuilt when new emotions are added
- **Fixed context**: cache size bounded by RAM/VRAM
- **No conversational reasoning**: CAG maps features → labels, no generation

---

### Future Work

- **Hybrid CAG + RAG**: use CAG for common emotions, RAG for edge cases
- **Incremental cache update**: online prototype update without full rebuild
- **Multi-face tracking**: maintain separate temporal state per identity
- **Adapter fine-tuning**: few-shot personalisation via cache injection
    """)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
def sidebar(engine: CAGInferenceEngine):
    with st.sidebar:
        st.title("🧠 CAG Emotion")
        st.caption("Cache-Augmented Generation")
        st.divider()

        st.markdown("**System**")
        st.json(engine.kv_cache.stats(), expanded=False)

        st.divider()
        st.markdown("**Settings**")
        st.caption(f"Device: `{DEVICE.upper()}`")
        st.caption(f"Feature dim: `{FEATURE_DIM}`")
        st.caption(f"Temporal α: `{TEMPORAL_ALPHA}`")
        st.caption(f"Similarity: `{SIMILARITY_METRIC}`")

        st.divider()
        if st.button("🔄 Rebuild Cache"):
            Path(CACHE_PATH).unlink(missing_ok=True)
            st.cache_resource.clear()
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    engine = load_engine()
    sidebar(engine)

    tabs = st.tabs(["📷 Live Detection", "⚡ Benchmark", "🧠 How It Works"])
    with tabs[0]:
        tab_live(engine)
    with tabs[1]:
        tab_benchmark()
    with tabs[2]:
        tab_explainer()


if __name__ == "__main__":
    main()
