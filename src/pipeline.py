"""
Indexing Pipeline — orchestrates sampling → embedding → indexing
Handles thumbnail generation, logging, and benchmark reporting
"""

import json
import logging
import os
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .sampler import FrameSampler, SampledFrame
from .embedder import CLIPEmbedder
from .vector_index import VideoVectorIndex

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(
        self,
        index_dir: str = "index",
        thumbnail_dir: str = "thumbnails",
        log_dir: str = "logs",
        batch_size: int = 32,
        temporal_window: int = 2,
        scene_threshold: float = 0.4,
        min_interval_sec: float = 1.0,
        uniform_interval_sec: float = 5.0,
    ):
        self.index_dir = Path(index_dir)
        self.thumbnail_dir = Path(thumbnail_dir)
        self.log_dir = Path(log_dir)

        for d in [self.index_dir, self.thumbnail_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.sampler = FrameSampler(
            scene_threshold=scene_threshold,
            min_interval_sec=min_interval_sec,
            uniform_interval_sec=uniform_interval_sec,
        )
        self.embedder = CLIPEmbedder(batch_size=batch_size)
        self.vector_index = VideoVectorIndex(index_dir=str(self.index_dir))

        self.temporal_window = temporal_window
        self.benchmark: Dict = {}

    def _save_thumbnail(self, frame: np.ndarray, name: str) -> str:
        path = self.thumbnail_dir / f"{name}.jpg"
        # Resize to 320x180 for storage efficiency
        thumb = cv2.resize(frame, (320, 180))
        cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(path)

    def run(self, video_input: str, force_reindex: bool = False) -> Dict:
        """
        video_input: path to video file OR directory of videos
        Returns benchmark dict.
        """
        if self.vector_index.exists() and not force_reindex:
            logger.info("Index already exists. Loading existing index.")
            self.vector_index.load()
            return {"status": "loaded_existing"}

        logger.info(f"Starting indexing pipeline for: {video_input}")
        tracemalloc.start()
        pipeline_start = time.time()

        # ── 1. Sampling ──────────────────────────────────────────────
        logger.info("Phase 1: Frame sampling")
        sampled_frames: List[SampledFrame] = []
        sample_start = time.time()

        if os.path.isdir(video_input):
            gen = self.sampler.sample_directory(video_input)
        else:
            gen = self.sampler.sample_video(video_input)

        for sf in gen:
            sampled_frames.append(sf)
            if len(sampled_frames) % 100 == 0:
                logger.info(f"  Sampled {len(sampled_frames)} frames...")

        sample_elapsed = time.time() - sample_start
        logger.info(
            f"Sampling complete: {len(sampled_frames)} frames in {sample_elapsed:.1f}s"
        )

        if not sampled_frames:
            raise RuntimeError("No frames sampled. Check video input.")

        # ── 2. Thumbnails ─────────────────────────────────────────────
        logger.info("Phase 2: Saving thumbnails")
        thumbnail_paths = []
        for sf in sampled_frames:
            name = f"{Path(sf.video_path).stem}_{sf.frame_idx:07d}"
            tp = self._save_thumbnail(sf.frame, name)
            thumbnail_paths.append(tp)

        # ── 3. Embedding ──────────────────────────────────────────────
        logger.info("Phase 3: Generating CLIP embeddings")
        embed_start = time.time()
        raw_frames = [sf.frame for sf in sampled_frames]
        embeddings = self.embedder.embed_frames(raw_frames)
        embed_elapsed = time.time() - embed_start

        frames_per_sec_embed = len(sampled_frames) / embed_elapsed
        logger.info(
            f"Embedding complete: {frames_per_sec_embed:.1f} frames/sec"
        )

        # ── 4. Temporal context smoothing ────────────────────────────
        logger.info(f"Phase 4: Temporal smoothing (window=±{self.temporal_window})")
        embeddings = CLIPEmbedder.apply_temporal_context(
            embeddings, window=self.temporal_window
        )

        # ── 5. Build metadata ────────────────────────────────────────
        metadata = []
        for i, sf in enumerate(sampled_frames):
            metadata.append(
                {
                    "frame_idx": sf.frame_idx,
                    "timestamp_sec": sf.timestamp_sec,
                    "timestamp_str": sf.timestamp_str,
                    "video_path": sf.video_path,
                    "thumbnail_path": thumbnail_paths[i],
                }
            )

        # ── 6. Index ─────────────────────────────────────────────────
        logger.info("Phase 5: Building FAISS index")
        self.vector_index.build(embeddings, metadata)
        self.vector_index.save()

        # ── 7. Benchmark ─────────────────────────────────────────────
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_elapsed = time.time() - pipeline_start

        self.benchmark = {
            "total_frames_sampled": len(sampled_frames),
            "sampling_throughput_fps": round(len(sampled_frames) / sample_elapsed, 2),
            "embedding_throughput_fps": round(frames_per_sec_embed, 2),
            "total_pipeline_sec": round(total_elapsed, 2),
            "peak_memory_mb": round(peak / 1e6, 1),
            "index_size": self.vector_index.index.ntotal,
            "embedding_dim": self.vector_index.dim,
            "model": self.embedder.model_name,
            "device": str(self.embedder.device),
        }

        bench_path = self.log_dir / "benchmark.json"
        with open(bench_path, "w") as f:
            json.dump(self.benchmark, f, indent=2)

        logger.info(f"Pipeline complete. Benchmark: {self.benchmark}")
        logger.info(f"Benchmark saved to {bench_path}")
        return self.benchmark
