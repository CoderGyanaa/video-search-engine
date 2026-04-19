"""
Streamlit UI for Video Search Engine
Run: streamlit run app.py
"""

import base64
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import streamlit as st

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)

sys.path.insert(0, str(Path(__file__).parent))
from src import IndexingPipeline, QueryEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoSearch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e6f0;
    }

    .main-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #7c3aed, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #6b7280;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .result-card {
        background: #13131a;
        border: 1px solid #1f1f2e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }

    .result-card:hover {
        border-color: #7c3aed;
    }

    .score-badge {
        background: linear-gradient(135deg, #7c3aed22, #06b6d422);
        border: 1px solid #7c3aed55;
        border-radius: 6px;
        padding: 3px 10px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #a78bfa;
        display: inline-block;
    }

    .timestamp-badge {
        background: #06b6d411;
        border: 1px solid #06b6d433;
        border-radius: 6px;
        padding: 3px 10px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #22d3ee;
        display: inline-block;
        margin-left: 6px;
    }

    .bench-card {
        background: #0f0f1a;
        border: 1px solid #1f1f30;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 8px;
    }

    .bench-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #10b981;
    }

    .bench-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    div[data-testid="stSidebarContent"] {
        background: #0d0d14;
        border-right: 1px solid #1a1a28;
    }

    .stTextInput > div > div > input {
        background: #13131a !important;
        border: 1px solid #2a2a3e !important;
        border-radius: 10px !important;
        color: #e8e6f0 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem 2rem !important;
    }

    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }

    .stSlider > div {
        color: #7c3aed !important;
    }

    .log-box {
        background: #080810;
        border: 1px solid #1a1a28;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #4ade80;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pipeline": None,
        "query_engine": None,
        "indexed": False,
        "benchmark": {},
        "results": [],
        "query_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def img_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def get_pipeline() -> IndexingPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = IndexingPipeline(
            index_dir="index",
            thumbnail_dir="thumbnails",
            log_dir="logs",
            batch_size=32,
            temporal_window=2,
        )
    return st.session_state.pipeline


def get_query_engine() -> QueryEngine:
    if st.session_state.query_engine is None:
        p = get_pipeline()
        st.session_state.query_engine = QueryEngine(
            index_dir="index",
            output_dir="outputs",
            embedder=p.embedder,
        )
    return st.session_state.query_engine


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Mono;font-size:0.7rem;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:0.15em;margin-bottom:1rem'>"
        "⬡ VIDEOSEARCH AI</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📁 Index Video")
    video_input = st.text_input(
        "Video path or directory",
        placeholder="/path/to/video.mp4",
        help="Absolute path to a .mp4/.avi/.mov file or a folder of videos",
    )

    col1, col2 = st.columns(2)
    with col1:
        force_reindex = st.checkbox("Force re-index", value=False)
    with col2:
        uniform_interval = st.number_input(
            "Sample every (s)", min_value=1, max_value=30, value=5
        )

    if st.button("🚀 Index Video", use_container_width=True):
        if not video_input:
            st.error("Please enter a video path.")
        elif not (
            os.path.isfile(video_input) or os.path.isdir(video_input)
        ):
            st.error(f"Path not found: {video_input}")
        else:
            with st.spinner("Indexing… this may take a few minutes"):
                try:
                    p = get_pipeline()
                    p.sampler.uniform_interval_sec = uniform_interval
                    bench = p.run(video_input, force_reindex=force_reindex)
                    st.session_state.benchmark = bench
                    st.session_state.indexed = True
                    # Reset query engine so it reloads index
                    st.session_state.query_engine = None
                    st.success("✅ Indexing complete!")
                except Exception as e:
                    st.error(f"Search failed: {e}\n\n{traceback.format_exc()}")

    # Check if existing index
    if not st.session_state.indexed and Path("index/faiss.index").exists():
        st.info("Existing index detected. Ready to search.")
        st.session_state.indexed = True
        bench_path = Path("logs/benchmark.json")
        if bench_path.exists():
            with open(bench_path) as f:
                st.session_state.benchmark = json.load(f)

    st.markdown("---")

    # Benchmark display
    if st.session_state.benchmark:
        b = st.session_state.benchmark
        st.markdown("### 📊 Benchmark")
        metrics = [
            ("Frames Indexed", str(b.get("total_frames_sampled", "—"))),
            ("Embed Speed", f"{b.get('embedding_throughput_fps', '—')} fps"),
            ("Peak RAM", f"{b.get('peak_memory_mb', '—')} MB"),
            ("Index Dim", str(b.get("embedding_dim", "—"))),
            ("Device", str(b.get("device", "—"))),
            ("Model", str(b.get("model", "—")).split("/")[-1]),
        ]
        for label, val in metrics:
            st.markdown(
                f"""<div class="bench-card">
                <div class="bench-value">{val}</div>
                <div class="bench-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Query history
    if st.session_state.query_history:
        st.markdown("### 🕐 History")
        for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
            if st.button(f"↩ {q[:40]}", key=f"hist_{q}_{i}", use_container_width=True):
                st.session_state["prefill_query"] = q


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">VideoSearch AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Natural Language Search Over Video Archives</div>',
    unsafe_allow_html=True,
)

if not st.session_state.indexed:
    st.info(
        "👈 Index a video first using the sidebar, or point to an existing index directory."
    )
    st.markdown(
        """
**Example queries you can try after indexing:**
- `person carrying a bag near the entrance`
- `red vehicle in the parking area`
- `two people talking after 18:00`
- `anything unusual in the corridor`
- `someone running between 00:05:00 and 00:15:00`
    """
    )
else:
    # ── Search form ───────────────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill_query", "")
    query = st.text_input(
        "🔍 Natural language query",
        value=prefill,
        placeholder='e.g. "person near entrance" or "car after 18:00"',
    )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        top_k = st.slider("Top K results", 3, 20, 8)
    with col_b:
        video_filter_input = st.text_input(
            "Filter by video name (optional)", placeholder="clip_01"
        )
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Search", use_container_width=True)

    if search_btn and query.strip():
        st.session_state.query_history.append(query.strip())
        with st.spinner("Searching…"):
            try:
                qe = get_query_engine()
                results = qe.search(
                    query=query.strip(),
                    top_k=top_k,
                    video_filter=video_filter_input.strip() or None,
                )
                st.session_state.results = results
            except Exception as e:
                st.error(f"Search failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results
        latency = results[0].get("query_latency_ms", "—") if results else "—"

        st.markdown(
            f"<div style='font-family:Space Mono;font-size:0.8rem;color:#6b7280;"
            f"margin-bottom:1.2rem'>Found <b style='color:#a78bfa'>{len(results)}</b> "
            f"results &nbsp;·&nbsp; {latency} ms</div>",
            unsafe_allow_html=True,
        )

        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, result in enumerate(
                results[row_start : row_start + cols_per_row]
            ):
                with cols[col_idx]:
                    thumb = result.get("thumbnail_path", "")
                    if thumb and Path(thumb).exists():
                        st.image(thumb, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='background:#1a1a28;height:100px;border-radius:8px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:#4b5563;font-size:0.75rem'>No thumbnail</div>",
                            unsafe_allow_html=True,
                        )

                    score = result.get("score", 0)
                    ts = result.get("timestamp_str", "—")
                    vid = Path(result.get("video_path", "unknown")).name

                    st.markdown(
                        f"""<div style="margin-top:6px">
                        <span class="timestamp-badge">{ts}</span>
                        <span class="score-badge">{score:.3f}</span>
                        <div style="font-size:0.7rem;color:#6b7280;margin-top:4px;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                        title="{vid}">{vid}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        with st.expander("📄 Raw JSON results"):
            st.json(results)

        st.markdown(
            f"<div style='font-size:0.75rem;color:#4b5563;font-family:Space Mono'>"
            f"Results saved → outputs/results.json & outputs/results.csv</div>",
            unsafe_allow_html=True,
        )
