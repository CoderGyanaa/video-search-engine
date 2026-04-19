# 🎬 VideoSearch AI — Intelligent Natural Language Video Search Engine

> Search any video archive with natural language. Retrieve timestamped, ranked moments in milliseconds.

**Built by [Gyana Ranjan Sahoo](https://github.com/CoderGyanaa)**

---

## 👨‍💻 About the Author

**Gyana Ranjan Sahoo**
Final-year B.Tech Computer Science Engineering student at C.V. Raman Global University, Bhubaneswar, Odisha, India.

- 📧 gyanaranjansahoo0033@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/gyanaranjansahoo0033)
- 🐙 [GitHub](https://github.com/CoderGyanaa)
- 📱 +91 7787007723

**Highlights:**
- 🏆 Second Runner Up — Infosys Global Hackathon 2025, Hyderabad
- 🌍 TCS CodeVita Season 13 — Ranked 1025 globally among 6000+ participants
- 🤖 AI Intern at Mirai School of Technology (Jan–Feb 2026)

---

## 🎥 Demo Video

[▶ Watch the 1-minute walkthrough](https://your-demo-link-here)

---

## 🏗️ Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE (offline)                   │
│                                                                       │
│  Video File / Dir                                                     │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────┐    scene-change      ┌─────────────────┐            │
│  │ FrameSampler │─── detection ──────▶│ Sampled Frames   │            |
│  │  (OpenCV)    │    + uniform fallback│  (N frames)     │            │
│  └──────────────┘                      └────────┬────────┘            │
│                                                 │                     │
│                                                 ▼                     │
│                                    ┌─────────────────────┐            │
│                                    │   CLIPEmbedder      │            │
│                                    │  (ViT-L/14, FP16)   │            │
│                                    │  Batched inference  │            │
│                                    │  Temporal smoothing │            │
│                                    └──────────┬──────────┘            │
│                                               │ L2-normalised         │
│                                               │ embeddings (D=768)    │
│                                               ▼                       │
│                                    ┌──────────────────────┐           │
│                                    │   FAISS Index        │           │
│                                    │  IndexFlatIP (<50k)  │           │
│                                    │  IndexIVFFlat (≥50k) │           │
│                                    │  + JSON metadata     │           │
│                                    └──────────────────────┘           │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE (online, sub-second)           │
│                                                                       │
│  "person near entrance after 18:00"                                   │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────┐    parse time      ┌─────────────────┐              │
│  │  Query Parse │ ─── filter ───────▶│  CLIP Text Enc  │             │
│  └──────────────┘  (start, end sec)  └────────┬────────┘              │
│                                               │ query embedding       │
│                                               ▼                       │
│                                    ┌─────────────────────┐            │
│                                    │   ANN Search (FAISS) │           │
│                                    │   + temporal filter  │           │
│                                    │   + video filter     │           │
│                                    └──────────┬──────────┘            │
│                                               │ top-K results         │
│                                               ▼                       │
│                                    ┌──────────────────────┐           │
│                                    │  Ranked Results      │           │
│                                    │  timestamp, score,   │           │
│                                    │  thumbnail, video    │           │
│                                    └──────────────────────┘           │
│                                               │                       │
│                              ┌────────────────┼────────────────┐      │
│                              ▼                ▼                ▼      │
│                         Streamlit UI        CLI           results.json│
└───────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/CoderGyanaa/video-search-engine.git
cd video-search-engine
```

### 2. Run setup (Windows)

```bash
setup.bat
```

Or manually:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Index a video

```bash
python cli.py index C:\path\to\video.mp4
```

### 4. Search (Streamlit UI)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. Search (CLI)

```bash
python cli.py search "person carrying a bag near the entrance"
python cli.py search "vehicle on road after 00:03:00" --top-k 5
python cli.py search "anything unusual happening"
```

---

## 📁 Project Structure

```
video-search-engine/
├── src/
│   ├── __init__.py
│   ├── sampler.py        # Frame sampling (scene-change + uniform)
│   ├── embedder.py       # CLIP embedding (batched, FP16, temporal smoothing)
│   ├── vector_index.py   # FAISS index (build, save, load, query)
│   ├── pipeline.py       # Indexing orchestrator + benchmark
│   └── query_engine.py   # Query interface + temporal filter parsing
├── app.py                # Streamlit UI
├── cli.py                # Command-line interface
├── requirements.txt
├── setup.bat             # Windows one-click setup
├── run_ui.bat            # Windows UI launcher
├── README.md
├── index/                # FAISS index + metadata (auto-created)
├── thumbnails/           # Frame thumbnails (auto-created)
├── outputs/              # results.json, results.csv (auto-created)
└── logs/                 # Logs + benchmark.json (auto-created)
```

---

## 🎯 Design Decisions

### Frame Sampling: Scene-Change + Uniform Fallback

**Chosen:** HSV histogram comparison (OpenCV) for scene detection + uniform sampling every N seconds as fallback.

**Why not PySceneDetect?**
- Zero extra dependency — pure OpenCV
- Streaming — never loads full video into RAM, handles 30+ minute videos
- Tunable sensitivity via `scene_threshold`

**Why not uniform-only?**
- Uniform sampling misses rapid visual changes and over-samples static scenes
- Scene-change sampling captures semantically meaningful transitions

### CLIP Model: ViT-L/14

**Chosen:** `openai/clip-vit-large-patch14`

| Model | Embedding Dim | Params | Retrieval Quality | Speed (CPU) |
|-------|--------------|--------|-------------------|-------------|
| ViT-B/32 | 512 | 150M | Good | Fast |
| ViT-L/14 | 768 | 428M | **Best** | Moderate |
| SigLIP-L | 1152 | 428M | Slightly better | Slower |

**FP16:** Halves VRAM, negligible quality loss. CPU automatically uses FP32.

### Temporal Context Smoothing

Each frame embedding is averaged with its ±2 nearest neighbors before indexing.

**Why?** A single frame is often ambiguous — averaging with nearby frames captures motion context cheaply, without requiring video transformers.

### Vector Store: FAISS

| Store | Latency | Infra | Offline | Scale |
|-------|---------|-------|---------|-------|
| FAISS | <1ms | None | ✅ | ✅ |
| Chroma | ~5ms | None | ✅ | Limited |
| Pinecone | ~50ms | Cloud | ❌ | ✅ |
| Weaviate | ~20ms | Docker | Partial | ✅ |

FAISS is the right choice for an offline, latency-critical system. Auto-switches from `IndexFlatIP` (exact) to `IndexIVFFlat` (approximate) above 50k frames.

### Temporal Filter Parsing

Regex-based NLP parser handles temporal expressions:
- `"after 6pm"` → filter timestamps > 18:00
- `"before 8:30"` → filter timestamps < 8:30
- `"between 18:00 and 20:00"` → filter to window

---

## 📊 Benchmark Results

*Tested on: Windows, CPU-only, Python 3.12*

| Metric | Value |
|--------|-------|
| Frames Indexed (9-min video) | 58 frames |
| Embedding throughput (CPU) | ~1.4 frames/sec |
| ANN query latency | 2–15 ms |
| Peak RAM | ~1006 MB |
| Index size (58 frames) | < 1 MB |
| Total pipeline (9-min video) | ~189 sec |
| Embedding dim | 768 |
| Model | openai/clip-vit-large-patch14 |
| Device | CPU |

---

## 🔍 Query Examples

| Type | Query |
|------|-------|
| Object | `vehicle on road` |
| Scene | `outdoor area with trees` |
| Spatial | `person near entrance carrying bag` |
| Temporal | `people walking after 00:03:00` |
| Temporal range | `car between 00:01:00 and 00:05:00` |
| Open-ended | `anything unusual happening` |

---

## ⚠️ Known Limitations

1. **No audio** — purely vision-based; spoken content is not indexed.
2. **Short clips** — very short videos yield few indexed frames; use 5+ minute videos for best results.
3. **CPU speed** — embedding is slow on CPU (~1.4 fps); GPU would give ~100x speedup.
4. **Fine-grained identity** — CLIP struggles with license plates, faces, small text.

### Scalability (1,000 hours)

At scale, the first bottleneck is in-memory FAISS. Solution: shard by video/time, use FAISS-on-disk (mmap), or migrate to Milvus with distributed backend. Indexing should be parallelized across GPUs with a Celery + Redis worker pool.

---

## 🚀 Open-Ended Explorations

1. **Temporal smoothing** — embedding-level ±window averaging as a cheap proxy for temporal context.
2. **Query decomposition** — complex queries could be split into sub-queries and results combined via score fusion.
3. **Re-ranking pathway** — after FAISS returns top-20, a BLIP-2 or LLaVA pass could verify relevance with VQA.

---

## 📬 Submission

**Variphi Take-Home Assignment**
Send GitHub link to: `connect@variphi.ai`
Subject: `Variphi Take-Home — Gyana Ranjan Sahoo`
