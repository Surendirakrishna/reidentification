"""Typed configuration loaded from config.yaml (with sane defaults)."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

# OSNet weight files that boxmot can download automatically (torchreid model zoo).
OSNET_WEIGHT_CHOICES = [
    "osnet_x1_0_msmt17.pt",
    "osnet_x1_0_market1501.pt",
    "osnet_x1_0_dukemtmcreid.pt",
    "osnet_ain_x1_0_msmt17.pt",
    "osnet_ibn_x1_0_msmt17.pt",
    "osnet_x0_75_msmt17.pt",
    "osnet_x0_5_msmt17.pt",
    "osnet_x0_25_msmt17.pt",
]

YOLO_MODEL_CHOICES = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolov8n.pt",
    "yolov8s.pt",
]


@dataclass
class InputConfig:
    directory: str = "Input Data"
    extensions: List[str] = field(default_factory=lambda: list(VIDEO_EXTENSIONS))
    # Optional wall-clock start time per camera id, e.g. {"cam1": "10:02:11"}.
    camera_start_times: Dict[str, str] = field(default_factory=dict)


@dataclass
class OutputConfig:
    directory: str = "Output"
    save_crops: bool = False
    save_embeddings: bool = True
    save_videos: bool = True
    video_codec: str = "auto"  # auto | h264_nvenc | libx264 | avc1 | mp4v
    video_scale: float = 1.0  # resize factor for written videos (1.0 = original)
    preview_every_n_frames: int = 15


@dataclass
class DetectorConfig:
    model: str = "yolo11n.pt"
    confidence: float = 0.40
    iou: float = 0.50
    device: str = "auto"
    imgsz: int = 640
    half: Any = "auto"
    classes: List[str] = field(default_factory=lambda: ["person"])
    models_dir: str = "models"


@dataclass
class TrackerConfig:
    type: str = "ocsort"
    det_thresh: float = 0.30
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.30
    delta_t: int = 3
    inertia: float = 0.2
    asso_func: str = "iou"
    use_byte: bool = False


@dataclass
class ReIDConfig:
    model: str = "osnet_x1_0_msmt17.pt"  # weights file name (auto-downloaded into models_dir)
    device: str = "auto"
    half: Any = "auto"
    similarity_threshold: float = 0.70
    uncertain_margin: float = 0.08
    embedding_interval: int = 10
    max_embeddings_per_identity: int = 20
    max_embeddings_per_track: int = 5
    batch_size: int = 16
    matching_strategy: str = "top_k_tracks"  # max | mean | top_k | top_k_tracks
    top_k: int = 3
    min_crop_width: int = 32
    min_crop_height: int = 64
    min_crop_area: int = 2048
    min_sharpness: float = 0.0  # Laplacian variance; 0 disables the blur filter
    max_pending_embeddings: int = 3
    models_dir: str = "models"


@dataclass
class TrackingConfig:
    min_track_frames: int = 10


@dataclass
class PerformanceConfig:
    frame_skip: int = 0
    detection_interval: int = 1
    max_frames: int = 0  # 0 = process the whole video
    start_frame: int = 0
    mode: str = "sequential"  # sequential | parallel
    parallel_workers: int = 2


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AppConfig":
        data = data or {}
        cfg = cls()
        for section in fields(cls):
            section_data = data.get(section.name)
            if not isinstance(section_data, dict):
                continue
            target = getattr(cfg, section.name)
            _apply_section(target, section_data)
        return cfg


def _apply_section(target: Any, values: Dict[str, Any]) -> None:
    valid = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in valid:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_section(current, value)
        elif isinstance(current, list) and isinstance(value, (list, tuple)):
            setattr(target, key, list(value))
        elif isinstance(current, dict) and isinstance(value, dict):
            setattr(target, key, dict(value))
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config.yaml; missing file or keys fall back to defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            data = loaded
    return AppConfig.from_dict(data)


def save_config(cfg: AppConfig, path: str | Path | None = None) -> Path:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return path


def resolve_osnet_weights_name(name: str) -> str:
    """Accept 'osnet_x1_0', 'osnet_x1_0_msmt17' or 'osnet_x1_0_msmt17.pt'."""
    name = (name or "osnet_x1_0_msmt17.pt").strip()
    if not name.endswith(".pt"):
        name = name + ".pt"
    stem = name[:-3]
    known_datasets = ("market1501", "dukemtmcreid", "msmt17", "duke", "market", "cuhk03")
    if not any(stem.endswith(ds) for ds in known_datasets):
        # Bare architecture name -> default to the MSMT17 (most general) weights.
        name = f"{stem}_msmt17.pt"
    return name
