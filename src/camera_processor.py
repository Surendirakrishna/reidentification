"""Per-camera processing: video -> YOLO -> OC-SORT -> track states -> OSNet -> global identities."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import AppConfig
from .detector import PersonDetector
from .global_identity import GlobalIdentityManager, MatchStatus
from .reid import OSNetReID, assess_crop_quality
from .tracker import OCSortTracker, TrackObservation
from .utils import format_seconds, get_logger, l2_normalize
from .video_utils import VideoReader, VideoSource, VideoWriter, crop_person, encode_jpeg
from .visualization import draw_hud, draw_track

log = get_logger("camera")


# --------------------------------------------------------------------------- #
# Track state
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    """Live state of one local OC-SORT track."""

    camera_id: str
    camera_name: str
    camera_index: int
    local_track_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen: float  # seconds (camera offset + video time)
    last_seen: float
    bbox: np.ndarray
    confidence: float = 0.0
    frame_count: int = 0  # processed frames in which the track was observed
    max_det_conf: float = 0.0
    latest_embedding: Optional[np.ndarray] = None
    embedding_history: Deque[Tuple[int, np.ndarray, float]] = field(default_factory=lambda: deque(maxlen=32))
    pending_embeddings: List[np.ndarray] = field(default_factory=list)
    pending_qualities: List[float] = field(default_factory=list)
    embedding_sum: Optional[np.ndarray] = None
    n_embeddings: int = 0
    last_embedding_step: int = -10 ** 9  # processed-frame index of the last embedding
    global_id: Optional[str] = None
    status: str = MatchStatus.PENDING.value
    similarity: float = 0.0  # similarity at assignment
    sim_sum: float = 0.0
    sim_count: int = 0
    candidate_id: Optional[str] = None
    candidate_similarity: float = 0.0
    best_crop_jpeg: Optional[bytes] = None
    best_crop_quality: float = -1.0
    best_crop_frame: int = -1
    resolve_attempts: int = 0
    embeddings_since_attempt: int = 0
    covisible_tracks: set = field(default_factory=set)  # local ids seen in the same frames
    finalized: bool = False

    @property
    def local_label(self) -> str:
        return f"C{self.camera_index}-T{self.local_track_id:02d}"

    @property
    def assigned(self) -> bool:
        return self.global_id is not None

    @property
    def avg_similarity(self) -> float:
        return self.sim_sum / self.sim_count if self.sim_count else 0.0

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        if self.embedding_sum is None or self.n_embeddings == 0:
            return None
        return l2_normalize(self.embedding_sum / self.n_embeddings)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


@dataclass
class TrackRecord:
    """Final, immutable summary of one local track (a row of tracks.csv)."""

    global_id: str
    camera_id: str
    camera_name: str
    camera_index: int
    local_track_id: int
    local_label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    max_detection_confidence: float
    average_reid_similarity: float
    assignment_similarity: float
    status: str
    candidate_id: Optional[str]
    candidate_similarity: float
    n_embeddings: int
    frames_observed: int
    mean_embedding: Optional[np.ndarray] = None
    best_crop_jpeg: Optional[bytes] = None


@dataclass
class CameraProgress:
    camera_id: str
    camera_name: str
    camera_index: int
    status: str = "pending"  # pending | running | done | error | cancelled
    frame_idx: int = 0
    frames_processed: int = 0
    frames_total: int = 0
    elapsed: float = 0.0
    eta: float = 0.0
    fps: float = 0.0
    active_tracks: int = 0
    total_tracks: int = 0
    global_ids: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.frames_total <= 0:
            return 1.0 if self.status == "done" else 0.0
        return min(1.0, self.frames_processed / self.frames_total)


@dataclass
class CameraResult:
    camera_id: str
    camera_name: str
    camera_index: int
    source_path: Path
    status: str = "done"  # done | error | cancelled
    error: str = ""
    output_video: Optional[Path] = None
    frames_processed: int = 0
    frames_total: int = 0
    processing_time: float = 0.0
    fps_video: float = 0.0
    width: int = 0
    height: int = 0
    track_records: List[TrackRecord] = field(default_factory=list)
    time_offset: float = 0.0

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.processing_time if self.processing_time > 0 else 0.0

    @property
    def n_tracks(self) -> int:
        return len(self.track_records)

    @property
    def n_assigned_tracks(self) -> int:
        return sum(1 for r in self.track_records if r.global_id)


# --------------------------------------------------------------------------- #
# Track manager
# --------------------------------------------------------------------------- #
class TrackManager:
    """Maintains TrackState objects for one camera and talks to the GlobalIdentityManager."""

    def __init__(
        self,
        camera: VideoSource,
        identity_manager: GlobalIdentityManager,
        *,
        min_track_frames: int = 10,
        embedding_interval: int = 10,
        max_age: int = 30,
        time_offset: float = 0.0,
    ):
        self.camera = camera
        self.idm = identity_manager
        self.min_track_frames = int(max(1, min_track_frames))
        self.embedding_interval = int(max(1, embedding_interval))
        self.max_age = int(max(1, max_age))
        self.time_offset = float(time_offset)
        self.active: Dict[int, TrackState] = {}
        self.records: List[TrackRecord] = []
        self.total_tracks_seen = 0
        self._current: List[TrackState] = []
        self.update_count = 0  # number of tracker updates seen (same clock as OC-SORT's max_age)
        self._last_seen_update: Dict[int, int] = {}
        self._gid_by_local: Dict[int, str] = {}  # local track id -> assigned Global ID (this camera)

    # ------------------------------------------------------------------ #
    def update(self, observations: List[TrackObservation], frame_idx: int, t_sec: float) -> Tuple[List[TrackState], List[TrackRecord]]:
        """Update states with this frame's tracker output. Returns (observed tracks, finalized records)."""
        self.update_count += 1
        observed: List[TrackState] = []
        for obs in observations:
            ts = self.active.get(obs.track_id)
            if ts is None:
                ts = TrackState(
                    camera_id=self.camera.camera_id, camera_name=self.camera.camera_name,
                    camera_index=self.camera.index, local_track_id=obs.track_id,
                    first_seen_frame=frame_idx, last_seen_frame=frame_idx, first_seen=t_sec, last_seen=t_sec,
                    bbox=obs.bbox.astype(np.float32),
                )
                self.active[obs.track_id] = ts
                self.total_tracks_seen += 1
            ts.bbox = obs.bbox.astype(np.float32)
            ts.confidence = float(obs.confidence)
            ts.max_det_conf = max(ts.max_det_conf, float(obs.confidence))
            ts.last_seen_frame = frame_idx
            ts.last_seen = t_sec
            ts.frame_count += 1
            self._last_seen_update[obs.track_id] = self.update_count
            observed.append(ts)

        # Remember who was visible together (used to rule out impossible Re-ID matches).
        observed_ids = {o.track_id for o in observations}
        if len(observed_ids) > 1:
            for ts in observed:
                ts.covisible_tracks.update(observed_ids)
                ts.covisible_tracks.discard(ts.local_track_id)

        # Tracks unseen for longer than the tracker keeps them alive are gone for good.
        finished: List[TrackRecord] = []
        for tid, ts in list(self.active.items()):
            if tid in observed_ids:
                continue
            last = self._last_seen_update.get(tid, self.update_count)
            if (self.update_count - last) > self.max_age:
                finished.append(self.finalize(ts))
        self._current = observed
        return observed, finished

    @property
    def current(self) -> List[TrackState]:
        return self._current

    def tracks_needing_embedding(self, observed: List[TrackState], step: int) -> List[TrackState]:
        """First appearance -> embed; afterwards every `embedding_interval` processed frames."""
        due: List[TrackState] = []
        for ts in observed:
            if ts.finalized:
                continue
            if ts.n_embeddings == 0 or (step - ts.last_embedding_step) >= self.embedding_interval:
                due.append(ts)
        return due

    def add_embedding(self, ts: TrackState, embedding: np.ndarray, quality: float, step: int, frame_idx: int,
                      crop_jpeg: Optional[bytes] = None) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        ts.latest_embedding = emb
        ts.embedding_history.append((frame_idx, emb, float(quality)))
        ts.embedding_sum = emb.copy() if ts.embedding_sum is None else ts.embedding_sum + emb
        ts.n_embeddings += 1
        ts.last_embedding_step = step
        ts.embeddings_since_attempt += 1
        if crop_jpeg is not None and quality > ts.best_crop_quality:
            ts.best_crop_jpeg = crop_jpeg
            ts.best_crop_quality = float(quality)
            ts.best_crop_frame = frame_idx
        if ts.assigned:
            sim = self.idm.add_track_embedding(
                ts.global_id, emb, quality, camera_id=ts.camera_id, local_track_id=ts.local_track_id,
                crop_jpeg=crop_jpeg, crop_quality=float(quality),
            )
            ts.sim_sum += sim
            ts.sim_count += 1
        else:
            ts.pending_embeddings.append(emb)
            ts.pending_qualities.append(float(quality))

    def ready_for_decision(self, ts: TrackState) -> bool:
        return (not ts.assigned and not ts.finalized and ts.frame_count >= self.min_track_frames
                and ts.n_embeddings > 0 and ts.embeddings_since_attempt > 0)

    def covisible_identities(self, ts: TrackState) -> List[str]:
        """Global IDs of tracks that were visible in this camera at the same time as `ts`.

        A person cannot be in two places of the same camera at once, so those IDs are
        impossible answers for `ts`.
        """
        return sorted({self._gid_by_local[t] for t in ts.covisible_tracks if t in self._gid_by_local})

    def try_resolve(self, ts: TrackState, final: bool = False, exclude: Optional[List[str]] = None) -> bool:
        """Ask the identity manager for a Global ID. Returns True when assigned."""
        if ts.assigned or not ts.pending_embeddings:
            return ts.assigned
        ts.resolve_attempts += 1
        ts.embeddings_since_attempt = 0
        if exclude is None:
            exclude = self.covisible_identities(ts)
        result = self.idm.resolve_track(
            ts.pending_embeddings, ts.pending_qualities,
            camera_id=ts.camera_id, camera_name=ts.camera_name, camera_index=ts.camera_index,
            local_track_id=ts.local_track_id, first_seen=ts.first_seen, first_frame=ts.first_seen_frame,
            last_seen=ts.last_seen, last_frame=ts.last_seen_frame, max_det_conf=ts.max_det_conf,
            crop_jpeg=ts.best_crop_jpeg, crop_quality=ts.best_crop_quality, final=final, exclude=exclude,
        )
        ts.candidate_id = result.best_id
        ts.candidate_similarity = float(result.best_similarity)
        if result.status == MatchStatus.PENDING:
            ts.status = MatchStatus.PENDING.value
            return False
        ts.global_id = result.global_id
        ts.status = result.status.value
        ts.similarity = float(result.similarity)
        if ts.global_id:
            self._gid_by_local[ts.local_track_id] = ts.global_id
        if result.status == MatchStatus.MATCHED:
            ts.sim_sum += float(result.similarity) * len(ts.pending_embeddings)
            ts.sim_count += len(ts.pending_embeddings)
        ts.pending_embeddings.clear()
        ts.pending_qualities.clear()
        return True

    def finalize(self, ts: TrackState) -> TrackRecord:
        """Close a track: force a decision when possible and emit its record."""
        if ts.finalized:
            raise RuntimeError(f"track {ts.local_label} already finalized")
        if not ts.assigned:
            if ts.frame_count < self.min_track_frames:
                ts.status = MatchStatus.SHORT.value
            elif ts.n_embeddings == 0:
                ts.status = MatchStatus.NO_EMBEDDING.value
            else:
                self.try_resolve(ts, final=True)
        if ts.assigned:
            self.idm.update_visit(
                ts.global_id, ts.camera_id, ts.local_track_id, last_seen=ts.last_seen, last_frame=ts.last_seen_frame,
                avg_similarity=ts.avg_similarity if ts.sim_count else ts.similarity, max_det_conf=ts.max_det_conf,
                n_embeddings=ts.n_embeddings, close=True,
            )
        ts.finalized = True
        self.active.pop(ts.local_track_id, None)
        self._last_seen_update.pop(ts.local_track_id, None)
        record = TrackRecord(
            global_id=ts.global_id or "",
            camera_id=ts.camera_id, camera_name=ts.camera_name, camera_index=ts.camera_index,
            local_track_id=ts.local_track_id, local_label=ts.local_label,
            start_frame=ts.first_seen_frame, end_frame=ts.last_seen_frame,
            start_time=ts.first_seen, end_time=ts.last_seen, duration=ts.duration,
            max_detection_confidence=ts.max_det_conf,
            average_reid_similarity=ts.avg_similarity if ts.sim_count else ts.similarity,
            assignment_similarity=ts.similarity, status=ts.status, candidate_id=ts.candidate_id,
            candidate_similarity=ts.candidate_similarity, n_embeddings=ts.n_embeddings,
            frames_observed=ts.frame_count, mean_embedding=ts.mean_embedding, best_crop_jpeg=ts.best_crop_jpeg,
        )
        self.records.append(record)
        return record

    def finalize_all(self) -> List[TrackRecord]:
        return [self.finalize(ts) for ts in list(self.active.values())]


# --------------------------------------------------------------------------- #
# Camera processor
# --------------------------------------------------------------------------- #
ProgressCallback = Callable[[CameraProgress], None]
PreviewCallback = Callable[[str, bytes], None]


def _pairwise_max_iou(boxes: np.ndarray) -> np.ndarray:
    """For each box, the maximum IoU with any other box (occlusion proxy)."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1).astype(np.float32)


class CameraProcessor:
    """Runs the full single-camera pipeline and feeds the shared GlobalIdentityManager."""

    def __init__(
        self,
        camera: VideoSource,
        detector: PersonDetector,
        reid: OSNetReID,
        identity_manager: GlobalIdentityManager,
        config: AppConfig,
        *,
        tracker: Optional[OCSortTracker] = None,
        output_dir: str | Path = "Output",
        time_offset: float = 0.0,
        progress_cb: Optional[ProgressCallback] = None,
        preview_cb: Optional[PreviewCallback] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_every: int = 10,
    ):
        self.camera = camera
        self.detector = detector
        self.reid = reid
        self.idm = identity_manager
        self.cfg = config
        self.tracker = tracker or OCSortTracker.from_config(config.tracker)
        self.output_dir = Path(output_dir)
        self.time_offset = float(time_offset)
        self.progress_cb = progress_cb
        self.preview_cb = preview_cb
        self.cancel_event = cancel_event
        self.progress_every = max(1, int(progress_every))
        # A lost track may be re-confirmed by OC-SORT up to max_age (+ min_hits re-confirmation)
        # updates later, so keep its state around for that long before finalizing it.
        self.track_manager = TrackManager(
            camera, identity_manager,
            min_track_frames=config.tracking.min_track_frames,
            embedding_interval=config.reid.embedding_interval,
            max_age=config.tracker.max_age + config.tracker.min_hits,
            time_offset=time_offset,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> CameraResult:
        cam = self.camera
        result = CameraResult(cam.camera_id, cam.camera_name, cam.index, cam.path, time_offset=self.time_offset)
        progress = CameraProgress(cam.camera_id, cam.camera_name, cam.index, status="running")
        t0 = time.perf_counter()
        writer: Optional[VideoWriter] = None
        reader: Optional[VideoReader] = None
        perf = self.cfg.performance
        out_cfg = self.cfg.output
        try:
            reader = VideoReader(cam.path)
            fps = reader.fps
            stride = perf.frame_skip + 1
            total_raw = max(0, reader.frame_count - perf.start_frame)
            total_processed = int(np.ceil(total_raw / stride)) if total_raw else 0
            if perf.max_frames:
                total_processed = min(total_processed, perf.max_frames) if total_processed else perf.max_frames
            progress.frames_total = total_processed
            result.frames_total = total_processed
            result.fps_video, result.width, result.height = fps, reader.width, reader.height

            if out_cfg.save_videos:
                out_w = max(2, int(round(reader.width * out_cfg.video_scale)))
                out_h = max(2, int(round(reader.height * out_cfg.video_scale)))
                out_path = self.output_dir / "processed" / f"{cam.camera_id}_reid.mp4"
                writer = VideoWriter(out_path, fps / stride, out_w, out_h, codec=out_cfg.video_codec)
                result.output_video = out_path
                log.info("Writing %s with %s/%s", out_path.name, writer.backend, writer.encoder)

            log.info("Processing %s (%s): %dx%d @ %.2f fps, %d frames, stride %d",
                     cam.camera_name, cam.path.name, reader.width, reader.height, fps, reader.frame_count, stride)
            self._loop(reader, writer, result, progress, fps, stride)
            if self.cancel_event is not None and self.cancel_event.is_set():
                result.status = "cancelled"
                progress.status = "cancelled"
            else:
                progress.status = "done"
        except Exception as exc:  # one bad video must not kill the run
            log.exception("Camera %s failed: %s", cam.camera_name, exc)
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            progress.status = "error"
            progress.message = result.error
        finally:
            # Flush remaining tracks so their records/identities are complete.
            try:
                self.track_manager.finalize_all()
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("Finalizing tracks failed: %s", exc)
            result.track_records = list(self.track_manager.records)
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:  # pragma: no cover
                    log.warning("Closing video writer failed: %s", exc)
            if reader is not None:
                reader.release()
            result.processing_time = time.perf_counter() - t0
            progress.elapsed = result.processing_time
            progress.total_tracks = len(result.track_records)
            progress.global_ids = len(self.idm)
            self._emit(progress)
        log.info("Camera %s finished: %d frames in %.1fs (%.1f fps), %d tracks, status=%s",
                 cam.camera_name, result.frames_processed, result.processing_time, result.processing_fps,
                 result.n_tracks, result.status)
        return result

    # ------------------------------------------------------------------ #
    def _loop(self, reader: VideoReader, writer: Optional[VideoWriter], result: CameraResult,
              progress: CameraProgress, fps: float, stride: int) -> None:
        perf = self.cfg.performance
        reid_cfg = self.cfg.reid
        out_cfg = self.cfg.output
        tm = self.track_manager
        detection_interval = max(1, int(perf.detection_interval))
        draw_states: List[TrackState] = []
        t_start = time.perf_counter()
        step = 0
        last_preview_step = -10 ** 9

        for frame_idx, frame in reader.frames(perf.frame_skip, perf.start_frame, perf.max_frames):
            if self.cancel_event is not None and self.cancel_event.is_set():
                log.info("Cancellation requested; stopping %s", self.camera.camera_name)
                break
            t_video = frame_idx / fps
            t_abs = self.time_offset + t_video

            if step % detection_interval == 0:
                detections = self.detector.detect(frame)
                observations = self.tracker.update(detections, frame)
                observed, _finished = tm.update(observations, frame_idx, t_abs)
                self._embed_tracks(frame, observed, step, frame_idx, reid_cfg)
                for ts in observed:
                    if tm.ready_for_decision(ts):
                        tm.try_resolve(ts)
                draw_states = observed

            if writer is not None or self.preview_cb is not None:
                for ts in draw_states:
                    # MATCHED -> similarity at assignment; UNCERTAIN -> score of the closest other identity.
                    shown_sim = ts.candidate_similarity if ts.status == "UNCERTAIN" else ts.similarity
                    draw_track(frame, ts.bbox, ts.local_label, ts.global_id, ts.confidence, shown_sim, ts.status)
                draw_hud(frame, self.camera.camera_name, frame_idx, reader.frame_count,
                         format_seconds(t_abs), len(draw_states), len(self.idm))
            if writer is not None:
                writer.write(frame)

            step += 1
            result.frames_processed = step
            if step % self.progress_every == 0 or step == 1:
                elapsed = time.perf_counter() - t_start
                progress.frame_idx = frame_idx
                progress.frames_processed = step
                progress.elapsed = elapsed
                progress.fps = step / elapsed if elapsed > 0 else 0.0
                remaining = max(0, progress.frames_total - step)
                progress.eta = remaining / progress.fps if progress.fps > 0 else 0.0
                progress.active_tracks = len(draw_states)
                progress.total_tracks = tm.total_tracks_seen
                progress.global_ids = len(self.idm)
                self._emit(progress)
            if self.preview_cb is not None and (step - last_preview_step) >= max(1, out_cfg.preview_every_n_frames):
                last_preview_step = step
                try:
                    self.preview_cb(self.camera.camera_id, encode_jpeg(frame, quality=80, max_side=960))
                except Exception:  # pragma: no cover - preview is best effort
                    pass
        progress.frames_processed = step
        progress.frame_idx = max(progress.frame_idx, step)

    def _embed_tracks(self, frame: np.ndarray, observed: List[TrackState], step: int, frame_idx: int, reid_cfg) -> None:
        """Batch OSNet inference for the tracks that are due for an embedding."""
        tm = self.track_manager
        due = tm.tracks_needing_embedding(observed, step)
        if not due:
            return
        boxes = np.stack([ts.bbox for ts in observed], axis=0) if observed else np.zeros((0, 4), dtype=np.float32)
        occlusion = _pairwise_max_iou(boxes)
        occ_by_id = {ts.local_track_id: float(o) for ts, o in zip(observed, occlusion)}
        crops, states, qualities = [], [], []
        for ts in due:
            crop = crop_person(frame, ts.bbox)
            q = assess_crop_quality(
                crop, det_conf=ts.confidence, occlusion=occ_by_id.get(ts.local_track_id, 0.0),
                min_width=reid_cfg.min_crop_width, min_height=reid_cfg.min_crop_height,
                min_area=reid_cfg.min_crop_area, min_sharpness=reid_cfg.min_sharpness,
            )
            if not q.ok:
                # Retry a little later instead of waiting a full interval.
                ts.last_embedding_step = step - tm.embedding_interval + min(3, tm.embedding_interval)
                continue
            crops.append(crop)
            states.append(ts)
            qualities.append(q.score)
        if not crops:
            return
        try:
            embeddings = self.reid.extract(crops)
        except Exception as exc:
            log.warning("OSNet inference failed on %d crops: %s", len(crops), exc)
            return
        for ts, crop, q, emb in zip(states, crops, qualities, embeddings):
            jpeg = None
            if q > ts.best_crop_quality:
                try:
                    jpeg = encode_jpeg(crop, quality=85, max_side=256)
                except Exception:
                    jpeg = None
            tm.add_embedding(ts, emb, q, step, frame_idx, crop_jpeg=jpeg)

    def _emit(self, progress: CameraProgress) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(progress)
            except Exception:  # pragma: no cover - UI callbacks must not break processing
                pass
