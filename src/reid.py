"""OSNet person re-identification: preprocessing, batched embedding extraction, crop quality."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ReIDConfig, resolve_osnet_weights_name
from .utils import (
    DeviceInfo,
    cosine_similarity,
    cosine_similarity_matrix,
    get_logger,
    l2_normalize,
    resolve_device,
    resolve_half,
)
from .video_utils import clamp_bbox, crop_person, is_valid_bbox

__all__ = [
    "OSNetReID",
    "CropQuality",
    "assess_crop_quality",
    "l2_normalize",
    "cosine_similarity",
    "cosine_similarity_matrix",
]

log = get_logger("reid")


# --------------------------------------------------------------------------- #
# Crop quality
# --------------------------------------------------------------------------- #
@dataclass
class CropQuality:
    ok: bool
    score: float  # 0..1, higher = better Re-ID sample
    reason: str = ""
    width: int = 0
    height: int = 0
    sharpness: float = 0.0


def crop_sharpness(crop: np.ndarray, norm_height: int = 128) -> float:
    """Variance of the Laplacian on a scale-normalised grayscale crop (higher = sharper)."""
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if h != norm_height:
        scale = norm_height / max(1, h)
        crop = cv2.resize(crop, (max(1, int(round(w * scale))), norm_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_crop_quality(
    crop: Optional[np.ndarray],
    det_conf: float = 1.0,
    occlusion: float = 0.0,
    min_width: int = 32,
    min_height: int = 64,
    min_area: int = 2048,
    min_sharpness: float = 0.0,
    compute_sharpness: bool = True,
) -> CropQuality:
    """Decide whether a crop is good enough for Re-ID and score it in [0, 1].

    The score combines crop size, aspect ratio plausibility, sharpness,
    detector confidence and an occlusion estimate (max IoU with other boxes).
    """
    if crop is None or crop.size == 0:
        return CropQuality(False, 0.0, "empty")
    h, w = crop.shape[:2]
    if w < min_width or h < min_height or (w * h) < min_area:
        return CropQuality(False, 0.0, "too_small", w, h)

    sharp = crop_sharpness(crop) if compute_sharpness else 0.0
    if min_sharpness > 0 and compute_sharpness and sharp < min_sharpness:
        return CropQuality(False, 0.0, "blurry", w, h, sharp)

    size_score = min(1.0, h / 192.0) * min(1.0, w / 72.0)
    size_score = float(np.sqrt(size_score))
    aspect = h / max(1.0, float(w))
    if 1.5 <= aspect <= 4.0:
        aspect_score = 1.0
    elif 1.0 <= aspect < 1.5 or 4.0 < aspect <= 5.5:
        aspect_score = 0.7
    else:
        aspect_score = 0.4
    sharp_score = min(1.0, sharp / 100.0) if compute_sharpness else 0.5
    conf_score = float(np.clip(det_conf, 0.0, 1.0))
    occ = float(np.clip(occlusion, 0.0, 1.0))
    score = 0.40 * size_score + 0.20 * aspect_score + 0.20 * sharp_score + 0.20 * conf_score
    score *= 1.0 - 0.5 * occ
    return CropQuality(True, float(np.clip(score, 0.0, 1.0)), "", w, h, sharp)


# --------------------------------------------------------------------------- #
# OSNet wrapper
# --------------------------------------------------------------------------- #
class OSNetReID:
    """OSNet feature extractor built on the BoxMOT/torchreid OSNet implementation.

    Weights (e.g. osnet_x1_0_msmt17.pt) are downloaded automatically into
    `models_dir`. Embeddings are L2-normalised float32 vectors (512-d for x1_0).
    """

    def __init__(
        self,
        model: str = "osnet_x1_0_msmt17.pt",
        device: str = "auto",
        half: object = "auto",
        batch_size: int = 16,
        models_dir: str | Path = "models",
    ):
        import torch
        from boxmot.reid.core import ReID

        self.device: DeviceInfo = resolve_device(device)
        self.half = resolve_half(half, self.device)
        self.batch_size = max(1, int(batch_size))
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.weights_name = resolve_osnet_weights_name(model)
        self.weights_path = (self.models_dir / self.weights_name).resolve()
        self.model_name = self.weights_name.rsplit("_", 1)[0]  # osnet_x1_0_msmt17.pt -> osnet_x1_0

        log.info("Loading OSNet Re-ID model %s on %s (fp16=%s)", self.weights_name, self.device.resolved, self.half)
        torch_device = torch.device(self.device.resolved)
        self._reid = ReID(self.weights_path, device=torch_device, half=self.half)
        backend = self._reid.model
        self.net = backend.model  # torch.nn.Module in eval mode
        self.net.eval()
        self.torch_device = torch_device
        self.input_shape: Tuple[int, int] = tuple(int(v) for v in backend.input_shape)  # (H, W)
        self.mean = backend.mean_array.to(torch_device)
        self.std = backend.std_array.to(torch_device)
        self.dtype = torch.float16 if self.half else torch.float32
        self._lock = threading.Lock()  # one shared model instance can serve several camera workers
        self.embedding_dim = self._probe_embedding_dim()
        log.info("OSNet ready: input %sx%s, embedding dim %d", self.input_shape[0], self.input_shape[1], self.embedding_dim)

    # ------------------------------------------------------------------ #
    def _probe_embedding_dim(self) -> int:
        import torch

        with torch.inference_mode():
            dummy = torch.zeros((1, 3, *self.input_shape), dtype=self.dtype, device=self.torch_device)
            out = self.net(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return int(out.shape[-1])

    def preprocess(self, crops: Sequence[np.ndarray]):
        """BGR crops -> normalised NCHW tensor on the model device.

        Pipeline: resize to model input (W,H) -> BGR->RGB -> [0,1] -> ImageNet mean/std.
        """
        import torch

        h, w = self.input_shape
        batch = np.empty((len(crops), h, w, 3), dtype=np.uint8)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                batch[i] = 0
                continue
            resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            batch[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(batch).to(self.torch_device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        tensor = (tensor - self.mean.to(self.dtype)) / self.std.to(self.dtype)
        return tensor.contiguous()

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR person crops -> (N, D) float32, L2-normalised rows."""
        import torch

        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        outputs: List[np.ndarray] = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                chunk = crops[start:start + self.batch_size]
                tensor = self.preprocess(chunk)
                feats = self.net(tensor)
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                outputs.append(feats.float().cpu().numpy())
        features = np.concatenate(outputs, axis=0).astype(np.float32)
        return l2_normalize(features, axis=1)

    def extract_from_frame(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> Tuple[np.ndarray, List[int]]:
        """Crop boxes from `frame` and embed them. Returns (embeddings, indices of valid boxes)."""
        h, w = frame.shape[:2]
        crops, valid = [], []
        for i, box in enumerate(boxes):
            if not is_valid_bbox(box, w, h, 2, 2):
                continue
            crop = crop_person(frame, clamp_bbox(box, w, h))
            if crop is None:
                continue
            crops.append(crop)
            valid.append(i)
        return self.extract(crops), valid

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Embed a single full-person image (e.g. an uploaded query)."""
        return self.extract([image_bgr])[0]

    @classmethod
    def from_config(cls, cfg: ReIDConfig, device: Optional[str] = None) -> "OSNetReID":
        return cls(
            model=cfg.model,
            device=device or cfg.device,
            half=cfg.half,
            batch_size=cfg.batch_size,
            models_dir=cfg.models_dir,
        )
