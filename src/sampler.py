"""
Frame Sampler — Scene-change-based + uniform fallback
Handles large videos (30+ min) without memory overload via streaming
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Generator, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    frame_idx: int
    timestamp_sec: float
    timestamp_str: str
    frame: np.ndarray  # BGR uint8
    video_path: str


def sec_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class FrameSampler:
    """
    Sampling strategy:
      1. Scene-change detection via histogram difference (fast, no deps beyond OpenCV)
      2. Minimum interval guard to avoid burst-sampling during rapid cuts
      3. Uniform fallback every N seconds to ensure sparse videos are covered

    Why this over PySceneDetect?
      - Zero extra dependency, pure OpenCV
      - Streaming — never loads full video into RAM
      - Tunable sensitivity
    """

    def __init__(
        self,
        scene_threshold: float = 0.4,
        min_interval_sec: float = 1.0,
        uniform_interval_sec: float = 5.0,
        resize_for_diff: Tuple[int, int] = (64, 36),
    ):
        self.scene_threshold = scene_threshold
        self.min_interval_sec = min_interval_sec
        self.uniform_interval_sec = uniform_interval_sec
        self.resize_for_diff = resize_for_diff

    def _histogram_diff(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Fast scene-change score via HSV histogram comparison."""
        a = cv2.resize(frame_a, self.resize_for_diff)
        b = cv2.resize(frame_b, self.resize_for_diff)
        a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        score = 0.0
        for ch in range(3):
            hist_a = cv2.calcHist([a_hsv], [ch], None, [32], [0, 256])
            hist_b = cv2.calcHist([b_hsv], [ch], None, [32], [0, 256])
            cv2.normalize(hist_a, hist_a)
            cv2.normalize(hist_b, hist_b)
            score += cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        return score / 3.0

    def sample_video(self, video_path: str) -> Generator[SampledFrame, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(
            f"Video: {Path(video_path).name} | FPS={fps:.1f} | "
            f"Frames={total_frames} | Duration={sec_to_hms(duration)}"
        )

        prev_frame = None
        last_sampled_sec = -999.0
        last_uniform_sec = -999.0
        frame_idx = 0
        sampled_count = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps

            # --- uniform fallback ---
            if current_sec - last_uniform_sec >= self.uniform_interval_sec:
                yield SampledFrame(
                    frame_idx=frame_idx,
                    timestamp_sec=current_sec,
                    timestamp_str=sec_to_hms(current_sec),
                    frame=frame.copy(),
                    video_path=video_path,
                )
                last_sampled_sec = current_sec
                last_uniform_sec = current_sec
                sampled_count += 1
                prev_frame = frame
                frame_idx += 1
                continue

            # --- scene change detection ---
            if (
                prev_frame is not None
                and current_sec - last_sampled_sec >= self.min_interval_sec
            ):
                diff = self._histogram_diff(prev_frame, frame)
                if diff >= self.scene_threshold:
                    yield SampledFrame(
                        frame_idx=frame_idx,
                        timestamp_sec=current_sec,
                        timestamp_str=sec_to_hms(current_sec),
                        frame=frame.copy(),
                        video_path=video_path,
                    )
                    last_sampled_sec = current_sec
                    sampled_count += 1

            prev_frame = frame
            frame_idx += 1

        cap.release()
        elapsed = time.time() - t0
        throughput = frame_idx / elapsed if elapsed > 0 else 0
        logger.info(
            f"Sampling done: {sampled_count}/{frame_idx} frames sampled "
            f"| {throughput:.1f} frames/sec processed"
        )

    def sample_directory(self, dir_path: str) -> Generator[SampledFrame, None, None]:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        videos = sorted(
            p for p in Path(dir_path).rglob("*") if p.suffix.lower() in exts
        )
        if not videos:
            raise RuntimeError(f"No video files found in {dir_path}")
        logger.info(f"Found {len(videos)} video(s) in {dir_path}")
        for v in videos:
            yield from self.sample_video(str(v))
