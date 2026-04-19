"""
CLIP Embedder — batched inference, FP16, CPU/GPU auto-detect
Temporal context: embeddings averaged over ±window frames before indexing
"""

import logging
import time
from typing import List, Optional
import numpy as np
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    MODEL_PRIORITY = [
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch32",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        logger.info(f"Device: {self.device} | FP16: {self.use_fp16}")
        self._load_model(model_name)

    def _load_model(self, model_name: Optional[str]):
        from transformers import CLIPProcessor, CLIPModel
        names = [model_name] if model_name else self.MODEL_PRIORITY
        for name in names:
            try:
                logger.info(f"Loading CLIP model: {name}")
                self.model = CLIPModel.from_pretrained(name)
                self.processor = CLIPProcessor.from_pretrained(name)
                self.model_name = name
                if self.use_fp16:
                    self.model = self.model.half()
                self.model = self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.projection_dim
                logger.info(
                    f"Loaded {name} | dim={self.embedding_dim} | "
                    f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        raise RuntimeError("Could not load any CLIP model.")

    def _bgr_to_pil(self, frame: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @torch.no_grad()
    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        all_embeddings = []
        t0 = time.time()
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            pil_imgs = [self._bgr_to_pil(f) for f in batch]
            inputs = self.processor(images=pil_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.use_fp16:
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }
            vision_outputs = self.model.vision_model(**inputs)
            feats = vision_outputs.pooler_output
            feats = self.model.visual_projection(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().float().numpy())
        elapsed = time.time() - t0
        result = np.vstack(all_embeddings)
        logger.debug(f"Embedded {len(frames)} frames in {elapsed:.2f}s ({len(frames)/elapsed:.1f} frames/sec)")
        return result

    @torch.no_grad()
    def embed_text(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_outputs = self.model.text_model(**inputs)
        feats = text_outputs.pooler_output
        feats = self.model.text_projection(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @staticmethod
    def apply_temporal_context(embeddings: np.ndarray, window: int = 2) -> np.ndarray:
        n, d = embeddings.shape
        smoothed = np.zeros_like(embeddings)
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            window_embs = embeddings[lo:hi]
            avg = window_embs.mean(axis=0)
            norm = np.linalg.norm(avg)
            smoothed[i] = avg / norm if norm > 0 else avg
        return smoothed