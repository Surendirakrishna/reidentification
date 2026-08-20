"""Background processing job used by the Streamlit UI (keeps the interface responsive)."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .camera_processor import CameraProgress
from .config import AppConfig
from .detector import PersonDetector
from .global_identity import GlobalIdentityManager
from .pipeline import VideoProcessor, build_identity_manager
from .reid import OSNetReID
from .results import ProcessingResult
from .utils import get_logger
from .video_utils import VideoSource

log = get_logger("job")


@dataclass
class ProcessingJob:
    """Runs VideoProcessor in a daemon thread; the UI polls `progress` / `previews` / `result`."""

    config: AppConfig
    cameras: List[VideoSource]
    detector: PersonDetector
    reid: OSNetReID
    identity_manager: Optional[GlobalIdentityManager] = None
    progress: Dict[str, CameraProgress] = field(default_factory=dict)
    previews: Dict[str, bytes] = field(default_factory=dict)
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    _thread: Optional[threading.Thread] = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.identity_manager = self.identity_manager or build_identity_manager(self.config)
        for cam in self.cameras:
            self.progress[cam.camera_id] = CameraProgress(cam.camera_id, cam.camera_name, cam.index, status="pending")

    # ------------------------------------------------------------------ #
    def start(self) -> "ProcessingJob":
        if self._thread is not None:
            return self
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="reid-processing", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def finished(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at) if self.started_at else 0.0

    # ------------------------------------------------------------------ #
    def _on_progress(self, p: CameraProgress) -> None:
        with self._lock:
            # Store a copy so the UI never sees a half-updated object.
            self.progress[p.camera_id] = CameraProgress(**{k: getattr(p, k) for k in p.__dataclass_fields__})

    def _on_preview(self, camera_id: str, jpeg: bytes) -> None:
        with self._lock:
            self.previews[camera_id] = jpeg

    def snapshot(self) -> Dict[str, CameraProgress]:
        with self._lock:
            return dict(self.progress)

    def _run(self) -> None:
        try:
            processor = VideoProcessor(
                self.config, self.detector, self.reid, self.identity_manager,
                progress_cb=self._on_progress, preview_cb=self._on_preview, cancel_event=self._cancel,
            )
            self.result = processor.run(self.cameras)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            log.exception("Processing job failed: %s", exc)
            self.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            self.finished_at = time.time()
