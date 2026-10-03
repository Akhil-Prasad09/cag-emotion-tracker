"""
dashboard.py
------------
Streamlit web dashboard for the CAG Emotion Tracker.
Provides: live webcam stream, emotion charts, benchmark runner, cache inspector.

Launch: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import time
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="CAG Emotion Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f1117; }
  .emotion-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 14px;
    margin: 4px;
    text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 700; }
  .metric-lbl { font-size: 0.75rem; color: #888; }
  .stProgress > div > div { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

EMOTION_COLORS_HEX = {
    "angry":    "#DC143C",
    "disgust":  "#228B22",
    "fear":     "#8B008B",
    "happy":    "#FFD700",
    "neutral":  "#A9A9A9",
    "sad":      "#4169E1",
    "surprise": "#FF8C00",
}

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


@st.cache_resource
def load_engine():
    # The trained FER-2013 model. The CNN path in src/modules/cag_engine.py
    # has no trained weights, so it would only produce random labels.
    from src.modules.sklearn_engine import SklearnEmotionEngine
    engine = SklearnEmotionEngine(model_path="models/fer_classifier.pkl")
    engine.load()
    return engine


def show_header():
    st.markdown("""
    <h1 style='text-align:center; background: linear-gradient(90deg, #667eea, #764ba2);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.5rem; margin-bottom: 0;'>
      🧠 CAG Emotion Tracker
    </h1>
    <p style='text-align:center; color:#888; margin-top:0;'>
      Cache-Augmented Generation · Real-Time · &lt;50ms Latency
    </p>
    """, unsafe_allow_html=True)


def show_sidebar():
    st.sidebar.title("⚙️ Settings")
    page = st.sidebar.radio(
        "Mode",
        ["🎥 Live Demo", "📊 Benchmark", "🗂️ Cache Inspector",
         "📖 Architecture"]
    )
    st.sidebar.markdown("---")
    alpha = st.sidebar.slider("Temporal Smoothing α", 0.1, 0.9, 0.4, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.caption("CAG Engine — ultra-low latency emotion detection")
    return page, alpha


def show_live_demo(engine, alpha):
    import cv2

    engine.alpha = alpha

    st.subheader("🎥 Live Webcam Feed")
    col1, col2 = st.columns([3, 2])

    with col1:
        img_placeholder = st.empty()
        status_bar = st.empty()

    with col2:
        emotion_placeholder = st.empty()
        scores_placeholder = st.empty()
        perf_placeholder = st.empty()

    start_btn = st.button("▶ Start Camera", type="primary")
    stop_btn = st.button("⏹ Stop")

    if not start_btn:
        st.info("Click 'Start Camera' to begin real-time emotion detection.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Ensure camera permissions are granted.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    emotion_history = deque(maxlen=50)
    frame_count = 0

    while not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine.infer(frame)
        frame_count += 1

        # Draw bounding box
        if result["bbox"]:
            x, y, w, h = result["bbox"]
            em = result["emotion"]
            color_bgr = {
                "angry":(0,0,220),"disgust":(0,128,0),"fear":(128,0,128),
                "happy":(0,220,220),"neutral":(180,180,180),
                "sad":(220,100,0),"surprise":(0,180,255)
            }.get(em, (200,200,200))
            cv2.rectangle(frame, (x,y),(x+w,y+h), color_bgr, 2)

        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_placeholder.image(rgb, channels="RGB", width="stretch")

        # Emotion display
        if result["face_found"]:
            em = result["emotion"]
            conf = result["confidence"]
            color = EMOTION_COLORS_HEX.get(em, "#aaa")
            emoji = EMOTION_EMOJIS.get(em, "")
            emotion_history.append(em)

            emotion_placeholder.markdown(f"""
            <div style='text-align:center; padding:20px;
                        background:#1e2130; border-radius:12px;
                        border: 2px solid {color};'>
              <div style='font-size:3.5rem;'>{emoji}</div>
              <div class='emotion-badge' style='background:{color}22; color:{color};'>
                {em.upper()}
              </div>
              <div style='margin-top:10px; color:#ccc;'>
                Confidence: <b>{conf*100:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Score bars
            smooth = result["smooth_scores"]
            scores_placeholder.text(
                "\n".join(
                    f"{e:<10} {int(s*100):3d}% |{'█'*int(s*20)}"
                    for e, s in sorted(smooth.items(), key=lambda x: -x[1])
                )
            )

        # Perf metrics
        lat = result["latency_ms"]
        fps = result["fps"]
        lat_color = "green" if lat < 50 else "orange" if lat < 100 else "red"
        perf_placeholder.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='color:{lat_color};'>{lat:.1f}<small>ms</small></div>
          <div class='metric-lbl'>Latency</div>
        </div>
        <div class='metric-card'>
          <div class='metric-val'>{fps:.1f}</div>
          <div class='metric-lbl'>FPS</div>
        </div>
        <div class='metric-card'>
          <div class='metric-val'>{frame_count}</div>
          <div class='metric-lbl'>Frames</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.01)  # Small yield so Streamlit can process events

    cap.release()
    st.success(f"Session ended. Processed {frame_count} frames.")


def show_benchmark():
    import plotly.graph_objects as go
    import torch

    st.subheader("📊 CAG vs. Baseline Benchmark")
    st.markdown("""
    This benchmark compares:
    - **CAG**: 7 dot products against preloaded KV cache (one per emotion)
    - **Baseline**: 10,000 dot products simulating a naive vector DB scan
    """)

    if st.button("🚀 Run Benchmark (500 trials)", type="primary"):
        with st.spinner("Benchmarking..."):
            from src.utils.benchmark import run_benchmark
            results = run_benchmark(n_trials=500)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAG Mean", f"{results['cag']['mean_ms']}ms")
        c2.metric("Baseline Mean", f"{results['baseline']['mean_ms']}ms")
        c3.metric("Speedup", f"{results['speedup_x']}x")
        c4.metric("CAG FPS", f"{results['cag']['fps']}")

        # Latency comparison chart
        cats = ["Mean", "P50", "P95", "P99"]
        cag_vals = [results['cag']['mean_ms'], results['cag']['p50_ms'],
                    results['cag']['p95_ms'], results['cag']['p99_ms']]
        base_vals = [results['baseline']['mean_ms'], results['baseline']['p50_ms'],
                     results['baseline']['p95_ms'], results['baseline']['p99_ms']]

        fig = go.Figure(data=[
            go.Bar(name="CAG", x=cats, y=cag_vals,
                   marker_color="#667eea"),
            go.Bar(name="Baseline", x=cats, y=base_vals,
                   marker_color="#e74c3c"),
        ])
        fig.update_layout(
            barmode="group",
            title="Latency Comparison (ms, lower is better)",
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font_color="#ccc",
        )
        st.plotly_chart(fig)

        st.markdown(f"""
        ### Why CAG is faster
        | Aspect | CAG | Baseline (RAG-like) |
        |--------|-----|---------------------|
        | Operations | 7 dot products | 10,000+ dot products |
        | Memory access | L2/L3 cache (tiny) | DRAM / disk (large) |
        | Network I/O | None | HTTP round-trip (10-100ms) |
        | Retrieval step | None | Encode → search → rank |
        | Speedup | 1× (baseline) | {results['speedup_x']}× slower |

        In production, RAG adds **HTTP latency** (10-100ms), **serialisation**,
        and **cross-encoder re-ranking**. CAG avoids all of this.
        """)


def show_cache_inspector():
    import plotly.graph_objects as go

    st.subheader("🗂️ KV Cache Inspector")

    try:
        import torch
        cache_data = torch.load("cache/emotion_cache.pt", map_location="cpu",
                                weights_only=False)  # dict with labels + metadata
        key_matrix = cache_data["key_matrix"].numpy()
        labels = cache_data["labels"]
        meta = cache_data.get("metadata", {})

        st.success(f"Cache loaded: {len(labels)} emotion classes, "
                   f"embedding dim = {key_matrix.shape[1]}")

        # VAD metadata
        if meta:
            import pandas as pd
            rows = []
            for em in labels:
                m = meta.get(em, {})
                rows.append({
                    "Emotion": em,
                    "Valence": m.get("valence", 0),
                    "Arousal": m.get("arousal", 0),
                    "Dominance": m.get("dominance", 0),
                })
            df = pd.DataFrame(rows).set_index("Emotion")
            st.dataframe(df)

        # Inter-prototype cosine similarity heatmap
        import torch.nn.functional as F
        km = torch.tensor(key_matrix)
        sim_matrix = (km @ km.T).numpy()

        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=labels, y=labels,
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=np.round(sim_matrix, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Prototype Cosine Similarity Matrix",
            paper_bgcolor="#0f1117",
            font_color="#ccc",
        )
        st.plotly_chart(fig)

        st.markdown("""
        **Reading the matrix:**
        - Diagonal = 1.0 (each class is identical to itself)
        - Off-diagonal values close to 0 = well-separated classes (good!)
        - High off-diagonal = confusion risk between those emotions
        """)

    except FileNotFoundError:
        st.warning("Cache file not found. Run `python main.py --build_cache` first.")


def show_architecture():
    st.subheader("📖 CAG Architecture")
    st.markdown("""
    ## Cache-Augmented Generation (CAG) for Emotion Detection

    ### Core Concept

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     PRELOAD PHASE (once)                    │
    │                                                             │
    │  Emotion Prototypes                                         │
    │  angry  → [0.12, -0.89, ..., 0.34]  ← KEY₀               │
    │  happy  → [0.78,  0.45, ..., -0.12] ← KEY₁               │
    │  ...                                                        │
    │  Stored as key_matrix (E × D) tensor in GPU VRAM           │
    └─────────────────────────────────────────────────────────────┘
                              │
                    loaded once at startup
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                   INFERENCE PHASE (each frame)              │
    │                                                             │
    │  Webcam → FaceDetector → 48×48 crop                        │
    │         → EmotionCNN  → query_emb (512-D, normalised)      │
    │         → KVCache.query():                                  │
    │              sim = query_emb @ key_matrix.T   (7 dot prods)│
    │              attn = softmax(sim / temp)                     │
    │              emotion = argmax(attn)                         │
    │         → TemporalSmoother (EMA)                           │
    │         → Display overlay                                   │
    └─────────────────────────────────────────────────────────────┘
    ```

    ### Why KV Cache Reduces Latency

    | Step | RAG | CAG |
    |------|-----|-----|
    | Knowledge storage | External vector DB | In-memory tensor |
    | Per-query cost | O(N) scan + HTTP | O(E) matmul |
    | Retrieval step | Yes (encode+search) | No |
    | Typical latency | 20–200ms extra | <0.5ms |

    ### Limitations
    - **Static knowledge**: cache represents fixed emotion prototypes;
      novel expressions not seen during prototype construction may be misclassified.
    - **Cache staleness**: prototypes need periodic retraining if the CNN is fine-tuned.
    - **No contextual chaining**: unlike LLM KV caches, there is no autoregressive
      dependency between frames.

    ### Future: Hybrid CAG + RAG
    - Use CAG for the 7 base emotions (fast, <1ms)
    - Trigger RAG only when confidence < threshold (e.g., ambiguous expression)
    - RAG retrieves similar user-specific expressions from personalised store
    - Best of both: speed for common cases, accuracy for edge cases
    """)


def main():
    show_header()
    page, alpha = show_sidebar()

    if "🎥" in page:
        try:
            engine = load_engine()
            show_live_demo(engine, alpha)
        except Exception as e:
            st.error(f"Engine failed to load: {e}")
            st.info("Retrain the model with: `python train_on_fer2013.py`")
    elif "📊" in page:
        show_benchmark()
    elif "🗂️" in page:
        show_cache_inspector()
    elif "📖" in page:
        show_architecture()


if __name__ == "__main__":
    main()
