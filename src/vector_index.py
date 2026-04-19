"""
Vector Index — FAISS with metadata store
Strategy:
  - IndexFlatIP (exact cosine) for small archives (<100k frames)
  - IndexIVFFlat for larger (auto-switches at 50k frames)
  - Metadata (timestamps, video paths) stored as JSON sidecar
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoVectorIndex:
    """
    FAISS-backed index with JSON metadata sidecar.

    Why FAISS over Pinecone/Weaviate/ChromaDB?
      - Zero infra, runs fully offline
      - Sub-millisecond ANN at 100k vectors on CPU
      - Battle-tested at Meta scale
      - IVF index degrades gracefully as archive grows
    """

    IVF_THRESHOLD = 50_000  # switch to IVF above this

    def __init__(self, index_dir: str = "index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"

        self.index = None
        self.metadata: List[Dict] = []
        self.dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        embeddings: float32 L2-normalised, shape (N, D)
        metadata: list of dicts with keys: timestamp_sec, timestamp_str,
                  frame_idx, video_path, thumbnail_path
        """
        import faiss

        assert embeddings.shape[0] == len(metadata), "Mismatch embeddings/metadata"
        n, d = embeddings.shape
        self.dim = d
        self.metadata = metadata

        logger.info(f"Building FAISS index: {n} vectors, dim={d}")
        t0 = time.time()

        if n < self.IVF_THRESHOLD:
            # Exact — guaranteed best results for small archives
            self.index = faiss.IndexFlatIP(d)
            logger.info("Using IndexFlatIP (exact cosine)")
        else:
            # Approximate — faster for large archives
            nlist = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings.astype(np.float32))
            self.index.nprobe = 16
            logger.info(f"Using IndexIVFFlat nlist={nlist} nprobe=16")

        self.index.add(embeddings.astype(np.float32))
        elapsed = time.time() - t0
        logger.info(f"Index built in {elapsed:.2f}s | {n/elapsed:.0f} vectors/sec" if elapsed > 0 else f"Index built instantly | {n} vectors")

    # ------------------------------------------------------------------
    # Persist / Load
    # ------------------------------------------------------------------

    def save(self):
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(
                {"dim": self.dim, "metadata": self.metadata}, f, indent=2
            )
        logger.info(f"Index saved to {self.index_dir}")

    def load(self):
        import faiss
        if not self.index_path.exists():
            raise FileNotFoundError(f"No index at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path) as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.metadata = data["metadata"]
        logger.info(
            f"Index loaded: {self.index.ntotal} vectors, dim={self.dim}"
        )

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        time_filter: Optional[Tuple[float, float]] = None,
        video_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        query_embedding: shape (1, D) or (D,)
        time_filter: (start_sec, end_sec) — only return frames within window
        Returns list of result dicts sorted by relevance score DESC.
        """
        assert self.index is not None, "Index not built/loaded."

        q = query_embedding.reshape(1, -1).astype(np.float32)
        t0 = time.time()

        # Fetch more candidates if filtering, to ensure top_k after filter
        fetch_k = top_k * 10 if (time_filter or video_filter) else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(q, fetch_k)
        elapsed_ms = (time.time() - t0) * 1000

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            meta["retrieval_latency_ms"] = round(elapsed_ms, 2)

            # Apply filters
            if time_filter:
                start_sec, end_sec = time_filter
                ts = meta.get("timestamp_sec", 0)
                if not (start_sec <= ts <= end_sec):
                    continue

            if video_filter:
                if video_filter.lower() not in meta.get("video_path", "").lower():
                    continue

            results.append(meta)
            if len(results) >= top_k:
                break

        logger.info(
            f"Query: {len(results)} results in {elapsed_ms:.1f}ms"
        )
        return results
