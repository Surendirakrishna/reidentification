"""Multi-camera orchestration: runs CameraProcessors against one shared GlobalIdentityManager."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .camera_processor import CameraProcessor, CameraProgress, CameraResult, TrackRecord
from .config import AppConfig
from .detector import PersonDetector
from .global_identity import GlobalIdentityManager
from .reid import OSNetReID
from .results import ProcessingResult, ResultsManager
from .tracker import OCSortTracker
from .utils import get_logger, get_memory_log_handler, parse_clock_time
from .video_utils import VideoSource

log = get_logger("pipeline")


def build_identity_manager(cfg: AppConfig) -> GlobalIdentityManager:
    return GlobalIdentityManager(
        similarity_threshold=cfg.reid.similarity_threshold,
        uncertain_margin=cfg.reid.uncertain_margin,
        max_embeddings_per_identity=cfg.reid.max_embeddings_per_identity,
        matching_strategy=cfg.reid.matching_strategy,
        top_k=cfg.reid.top_k,
        max_pending_embeddings=cfg.reid.max_pending_embeddings,
        max_embeddings_per_track=cfg.reid.max_embeddings_per_track,
    )


class VideoProcessor:
    """Processes several camera videos and maintains ONE global identity space.

    Mode A (sequential, default): Camera 1 -> Camera 2 -> ... (memory friendly).
    Mode B (parallel): a small thread pool processes cameras concurrently while
    sharing the same YOLO / OSNet model instances (guarded by locks) and the
    same GlobalIdentityManager. No per-camera GPU models are created.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: PersonDetector,
        reid: OSNetReID,
        identity_manager: Optional[GlobalIdentityManager] = None,
        *,
        output_dir: Optional[str | Path] = None,
        progress_cb: Optional[Callable[[CameraProgress], None]] = None,
        preview_cb: Optional[Callable[[str, bytes], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.cfg = config
        self.detector = detector
        self.reid = reid
        self.idm = identity_manager or build_identity_manager(config)
        self.output_dir = Path(output_dir or config.output.directory)
        self.progress_cb = progress_cb
        self.preview_cb = preview_cb
        self.cancel_event = cancel_event or threading.Event()
        self.results_manager = ResultsManager(self.output_dir)
        self.camera_results: List[CameraResult] = []

    # ------------------------------------------------------------------ #
    def _time_offset(self, camera: VideoSource) -> float:
        offsets = self.cfg.input.camera_start_times or {}
        raw = offsets.get(camera.camera_id) or offsets.get(camera.path.name) or offsets.get(camera.camera_name)
        if raw in (None, ""):
            return 0.0
        try:
            return parse_clock_time(str(raw))
        except ValueError:
            log.warning("Ignoring invalid start time %r for %s", raw, camera.camera_id)
            return 0.0

    def _make_processor(self, camera: VideoSource) -> CameraProcessor:
        return CameraProcessor(
            camera, self.detector, self.reid, self.idm, self.cfg,
            tracker=OCSortTracker.from_config(self.cfg.tracker),  # independent tracker per camera
            output_dir=self.output_dir, time_offset=self._time_offset(camera),
            progress_cb=self.progress_cb, preview_cb=self.preview_cb, cancel_event=self.cancel_event,
        )

    def run(self, cameras: List[VideoSource]) -> ProcessingResult:
        started = time.time()
        t0 = time.perf_counter()
        self.camera_results = []
        log.info("Found %d input videos to process: %s", len(cameras), ", ".join(c.camera_name for c in cameras))
        mode = (self.cfg.performance.mode or "sequential").lower()
        if mode == "parallel" and len(cameras) > 1:
            self._run_parallel(cameras)
        else:
            self._run_sequential(cameras)
        processing_time = time.perf_counter() - t0
        return self._finish(processing_time, started)

    def _run_sequential(self, cameras: List[VideoSource]) -> None:
        for camera in cameras:
            if self.cancel_event.is_set():
                log.info("Processing cancelled before %s", camera.camera_name)
                break
            result = self._make_processor(camera).run()
            self.camera_results.append(result)

    def _run_parallel(self, cameras: List[VideoSource]) -> None:
        workers = max(1, min(int(self.cfg.performance.parallel_workers), len(cameras)))
        log.info("Parallel mode with %d workers (shared models)", workers)
        results: Dict[str, CameraResult] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="camera") as pool:
            futures = {pool.submit(self._make_processor(cam).run): cam for cam in cameras}
            for fut in as_completed(futures):
                cam = futures[fut]
                try:
                    results[cam.camera_id] = fut.result()
                except Exception as exc:  # pragma: no cover - CameraProcessor already catches
                    log.exception("Camera %s crashed: %s", cam.camera_name, exc)
                    results[cam.camera_id] = CameraResult(cam.camera_id, cam.camera_name, cam.index, cam.path,
                                                          status="error", error=str(exc))
        self.camera_results = [results[c.camera_id] for c in cameras if c.camera_id in results]

    def _finish(self, processing_time: float, started: float) -> ProcessingResult:
        records: List[TrackRecord] = [r for cam in self.camera_results for r in cam.track_records]
        identities = self.idm.identities
        rm = self.results_manager
        tracks_df = rm.tracks_dataframe(records)
        identities_df = rm.identities_dataframe(identities)
        summary = rm.build_summary(self.camera_results, identities, records, processing_time, self.idm.stats)
        result = ProcessingResult(
            camera_results=self.camera_results, identities=identities, tracks_df=tracks_df,
            identities_df=identities_df, summary=summary, output_dir=self.output_dir,
            config_snapshot=self.cfg.to_dict(), started_at=started, finished_at=time.time(),
        )
        try:
            result.report_paths = rm.write_reports(result)
            if self.cfg.output.save_embeddings:
                result.report_paths["embeddings_npz"] = rm.write_embeddings(self.idm, records)
            if self.cfg.output.save_crops:
                result.report_paths["crops_dir"] = rm.write_crops(identities, records)
        except Exception as exc:
            log.exception("Writing reports failed: %s", exc)
        try:
            result.log_tail = get_memory_log_handler().tail(300)
            rm.save_state(result)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not persist session state: %s", exc)
        log.info("Processing completed: %d cameras, %d unique persons, %d tracks, %.1fs",
                 len(self.camera_results), len(identities), len(records), processing_time)
        return result
