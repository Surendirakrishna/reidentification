"""YOLO person detector (Ultralytics)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import DetectorConfig
from .utils import DeviceInfo, get_logger, resolve_device, resolve_half

log = get_logger("detector")

COCO_PERSON_CLASS_ID = 0


class PersonDetector:
    """Detects people with a YOLO model and returns (N, 6) arrays: x1, y1, x2, y2, conf, cls.

    The model is loaded once; thresholds can be changed per call. Inference is
    guarded by a lock so a single instance can be shared by worker threads.
    """

    def __init__(
        self,
        model: str = "yolo11n.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        device: str = "auto",
        imgsz: int = 640,
        half: object = "auto",
        classes: Sequence[str] = ("person",),
        models_dir: str | Path = "models",
    ):
        from ultralytics import YOLO  # heavy import kept local

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self._resolve_model_path(model)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device: DeviceInfo = resolve_device(device)
        self.half = resolve_half(half, self.device)
        self._lock = threading.Lock()

        log.info("Loading YOLO model %s on %s", self.model_path, self.device.resolved)
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names if hasattr(self.model, "names") else {}
        self.class_ids = self._resolve_class_ids(classes)
        self._precision_kwargs = self._detect_precision_kwargs()
        # Move the model to the target device (predict() also does this lazily).
        try:
            self.model.to(self._ultralytics_device())
        except Exception:  # pragma: no cover - older ultralytics
            pass

    # ------------------------------------------------------------------ #
    def _resolve_model_path(self, model: str) -> Path:
        p = Path(model)
        if p.suffix == "":
            p = p.with_suffix(".pt")
        if p.is_file():
            return p
        candidate = self.models_dir / p.name
        # If it is an official model name, Ultralytics downloads it into this path.
        return candidate if not p.is_absolute() else p

    def _resolve_class_ids(self, classes: Sequence[str]) -> List[int]:
        ids: List[int] = []
        names = {int(k): str(v) for k, v in dict(self.names).items()} if self.names else {}
        for cls in classes:
            if isinstance(cls, int) or (isinstance(cls, str) and cls.isdigit()):
                ids.append(int(cls))
                continue
            matched = [k for k, v in names.items() if v.lower() == str(cls).lower()]
            if matched:
                ids.append(matched[0])
            elif str(cls).lower() == "person":
                ids.append(COCO_PERSON_CLASS_ID)
        return sorted(set(ids)) or [COCO_PERSON_CLASS_ID]

    def _ultralytics_device(self) -> str:
        if self.device.is_cuda:
            return self.device.resolved.split(":")[1]  # "0"
        return self.device.resolved

    def _detect_precision_kwargs(self) -> dict:
        """Ultralytics >= 8.4 replaced `half=True` with `quantize='fp16'`."""
        if not self.half:
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            if "quantize" in DEFAULT_CFG_DICT:
                return {"quantize": "fp16"}
        except Exception:
            pass
        return {"half": True}

    # ------------------------------------------------------------------ #
    def warmup(self, size: tuple = (720, 1280, 3)) -> None:
        dummy = np.zeros(size, dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray, confidence: Optional[float] = None, iou: Optional[float] = None) -> np.ndarray:
        """Run person detection on one BGR frame -> np.ndarray (N, 6) float32."""
        conf = self.confidence if confidence is None else float(confidence)
        iou_thr = self.iou if iou is None else float(iou)
        with self._lock:
            results = self.model.predict(
                frame,
                conf=conf,
                iou=iou_thr,
                classes=self.class_ids,
                imgsz=self.imgsz,
                device=self._ultralytics_device(),
                verbose=False,
                **self._precision_kwargs,
            )
        return self._to_array(results[0] if results else None)

    def detect_batch(self, frames: List[np.ndarray], confidence: Optional[float] = None, iou: Optional[float] = None) -> List[np.ndarray]:
        if not frames:
            return []
        conf = self.confidence if confidence is None else float(confidence)
        iou_thr = self.iou if iou is None else float(iou)
        with self._lock:
            results = self.model.predict(
                frames, conf=conf, iou=iou_thr, classes=self.class_ids, imgsz=self.imgsz,
                device=self._ultralytics_device(), verbose=False, **self._precision_kwargs,
            )
        return [self._to_array(r) for r in results]

    @staticmethod
    def _to_array(result) -> np.ndarray:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        cls = boxes.cls.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        dets = np.concatenate([xyxy, conf, cls], axis=1)
        # Drop degenerate boxes (zero width/height or NaNs).
        finite = np.isfinite(dets).all(axis=1)
        wh_ok = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
        return np.ascontiguousarray(dets[finite & wh_ok])

    @classmethod
    def from_config(cls, cfg: DetectorConfig, device: Optional[str] = None) -> "PersonDetector":
        return cls(
            model=cfg.model,
            confidence=cfg.confidence,
            iou=cfg.iou,
            device=device or cfg.device,
            imgsz=cfg.imgsz,
            half=cfg.half,
            classes=cfg.classes,
            models_dir=cfg.models_dir,
        )
