"""Shared helpers: logging, device selection, time formatting, small math utils."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional

import numpy as np

LOGGER_NAME = "reid"
_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class MemoryLogHandler(logging.Handler):
    """Keeps the most recent log lines in memory so a UI can display them."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: Deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", _DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            line = self.format(record)
        except Exception:  # pragma: no cover - defensive
            return
        with self._lock:
            self.records.append(line)

    def tail(self, n: int = 100) -> list[str]:
        with self._lock:
            return list(self.records)[-n:]

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


_MEMORY_HANDLER: Optional[MemoryLogHandler] = None


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a logger under the application namespace."""
    if name != LOGGER_NAME and not name.startswith(LOGGER_NAME + "."):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, memory_capacity: int = 500) -> MemoryLogHandler:
    """Configure console + in-memory logging once. Safe to call repeatedly."""
    global _MEMORY_HANDLER
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, MemoryLogHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(console)
    if _MEMORY_HANDLER is None:
        _MEMORY_HANDLER = MemoryLogHandler(capacity=memory_capacity)
        logger.addHandler(_MEMORY_HANDLER)
    # Quieten very chatty third-party loggers.
    for noisy in ("ultralytics", "boxmot", "PIL", "matplotlib", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return _MEMORY_HANDLER


def get_memory_log_handler() -> MemoryLogHandler:
    return setup_logging()


# --------------------------------------------------------------------------- #
# Device management
# --------------------------------------------------------------------------- #
@dataclass
class DeviceInfo:
    requested: str
    resolved: str  # "cuda:0" | "cpu" | "mps"
    cuda_available: bool
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    torch_version: str = ""

    @property
    def is_cuda(self) -> bool:
        return self.resolved.startswith("cuda")

    def summary(self) -> str:
        if self.is_cuda:
            return f"CUDA ({self.gpu_name}, {self.vram_total_mb / 1024:.1f} GB VRAM)"
        return "CPU"


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve 'auto' / 'cpu' / 'cuda' / 'cuda:N' / '0' into a concrete torch device.

    Falls back to CPU automatically when CUDA is not available.
    """
    import torch  # local import keeps this module light for tests

    requested = (requested or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()
    resolved = "cpu"
    if requested in ("auto", ""):
        resolved = "cuda:0" if cuda_ok else "cpu"
    elif requested == "cpu":
        resolved = "cpu"
    elif requested.startswith("cuda") or requested.isdigit():
        idx = 0
        if requested.isdigit():
            idx = int(requested)
        elif ":" in requested:
            try:
                idx = int(requested.split(":", 1)[1])
            except ValueError:
                idx = 0
        if cuda_ok and idx < torch.cuda.device_count():
            resolved = f"cuda:{idx}"
        else:
            resolved = "cpu"
    elif requested == "mps":
        resolved = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    else:
        resolved = "cuda:0" if cuda_ok else "cpu"

    info = DeviceInfo(requested=requested, resolved=resolved, cuda_available=cuda_ok, torch_version=torch.__version__)
    if resolved.startswith("cuda"):
        idx = int(resolved.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        info.gpu_name = props.name
        info.vram_total_mb = props.total_memory / (1024 ** 2)
        try:
            free, _total = torch.cuda.mem_get_info(idx)
            info.vram_free_mb = free / (1024 ** 2)
        except Exception:  # pragma: no cover - depends on driver
            info.vram_free_mb = 0.0
    return info


def resolve_half(flag, device: DeviceInfo) -> bool:
    """'auto' -> FP16 on CUDA only; explicit booleans are respected (never on CPU)."""
    if isinstance(flag, str):
        flag = flag.strip().lower()
        if flag in ("auto", ""):
            return device.is_cuda
        return flag in ("1", "true", "yes", "on")
    return bool(flag) and device.is_cuda


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def format_seconds(seconds: float, with_millis: bool = False) -> str:
    """Format seconds as HH:MM:SS (or HH:MM:SS.mmm)."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}" if with_millis else base


def format_duration(seconds: float) -> str:
    """Human friendly duration such as '4m 32s' or '0.8s'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def parse_clock_time(value: str) -> float:
    """Parse 'HH:MM:SS' or 'MM:SS' into seconds."""
    parts = [p.strip() for p in str(value).split(":")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time string: {value!r}")
    nums = [float(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0.0)
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def frame_to_seconds(frame_idx: int, fps: float) -> float:
    fps = fps if fps and fps > 0 else 30.0
    return frame_idx / fps


class RateMeter:
    """Exponential moving average FPS meter."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._last = None
        self.fps = 0.0

    def tick(self, n: int = 1) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = n / dt
                self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
_CAMERA_PATTERNS = [
    re.compile(r"^(?:cam(?:era)?)[\s_\-]*0*(\d+)$", re.IGNORECASE),
    re.compile(r"^(?:c)[\s_\-]*0*(\d+)$", re.IGNORECASE),
    re.compile(r"^(?:cam(?:era)?)[\s_\-]*0*(\d+)[\s_\-].*$", re.IGNORECASE),
]


def infer_camera_name(filename: str) -> str:
    """Infer a human-friendly camera name from a file name.

    cam1.mp4 -> Camera 1, camera_02.mp4 -> Camera 2, entrance.mp4 -> Entrance,
    parking_lot.mp4 -> Parking Lot.
    """
    stem = Path(filename).stem.strip()
    for pattern in _CAMERA_PATTERNS:
        match = pattern.match(stem)
        if match:
            return f"Camera {int(match.group(1))}"
    words = re.split(r"[\s_\-]+", stem)
    words = [w for w in words if w]
    if not words:
        return stem or "Camera"
    return " ".join(w.capitalize() if not w.isupper() else w for w in words)


def sanitize_camera_id(filename: str) -> str:
    """Create a filesystem/ID safe camera identifier from a filename (its stem)."""
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "camera"


def short_camera_label(index: int) -> str:
    return f"C{index}"


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors along `axis`; zero vectors stay zero."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of `a` (N,D) and rows of `b` (M,D)."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return l2_normalize(a) @ l2_normalize(b).T


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunked(items: Iterable, size: int):
    """Yield successive lists of at most `size` items."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
