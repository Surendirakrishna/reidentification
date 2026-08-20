"""Aggregation of processing results and report generation (CSV / JSON / NPZ / crops)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .camera_processor import CameraResult, TrackRecord
from .global_identity import GlobalIdentity, GlobalIdentityManager
from .utils import ensure_dir, format_duration, format_seconds, get_logger

log = get_logger("results")

TRACK_COLUMNS = [
    "global_id", "camera_id", "camera_name", "camera_index", "local_track_id", "local_label",
    "start_frame", "end_frame", "start_time", "end_time", "start_seconds", "end_seconds", "duration",
    "duration_seconds", "max_detection_confidence", "average_reid_similarity", "assignment_similarity",
    "status", "candidate_id", "candidate_similarity", "n_embeddings", "frames_observed",
]

IDENTITY_COLUMNS = [
    "global_id", "cameras_seen", "camera_names", "num_cameras", "first_seen", "last_seen",
    "first_seen_seconds", "last_seen_seconds", "total_duration", "total_duration_seconds",
    "number_of_tracks", "gallery_size", "uncertain_tracks",
]


@dataclass
class ProcessingResult:
    """Everything the dashboard needs after a run."""

    camera_results: List[CameraResult]
    identities: List[GlobalIdentity]
    tracks_df: pd.DataFrame
    identities_df: pd.DataFrame
    summary: Dict
    output_dir: Path
    report_paths: Dict[str, Path] = field(default_factory=dict)
    config_snapshot: Dict = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    log_tail: List[str] = field(default_factory=list)

    @property
    def processed_videos(self) -> Dict[str, Path]:
        return {r.camera_id: r.output_video for r in self.camera_results if r.output_video and Path(r.output_video).exists()}

    def identity(self, global_id: str) -> Optional[GlobalIdentity]:
        for ident in self.identities:
            if ident.global_id == global_id:
                return ident
        return None

    def camera_result(self, camera_id: str) -> Optional[CameraResult]:
        for r in self.camera_results:
            if r.camera_id == camera_id:
                return r
        return None


class ResultsManager:
    """Builds tables/summary from camera results and writes all report files."""

    def __init__(self, output_dir: str | Path = "Output"):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    @staticmethod
    def tracks_dataframe(records: List[TrackRecord]) -> pd.DataFrame:
        rows = []
        for r in records:
            rows.append({
                "global_id": r.global_id,
                "camera_id": r.camera_id,
                "camera_name": r.camera_name,
                "camera_index": r.camera_index,
                "local_track_id": r.local_track_id,
                "local_label": r.local_label,
                "start_frame": r.start_frame,
                "end_frame": r.end_frame,
                "start_time": format_seconds(r.start_time),
                "end_time": format_seconds(r.end_time),
                "start_seconds": round(r.start_time, 3),
                "end_seconds": round(r.end_time, 3),
                "duration": format_seconds(r.duration),
                "duration_seconds": round(r.duration, 3),
                "max_detection_confidence": round(r.max_detection_confidence, 4),
                "average_reid_similarity": round(r.average_reid_similarity, 4),
                "assignment_similarity": round(r.assignment_similarity, 4),
                "status": r.status,
                "candidate_id": r.candidate_id or "",
                "candidate_similarity": round(r.candidate_similarity, 4),
                "n_embeddings": r.n_embeddings,
                "frames_observed": r.frames_observed,
            })
        df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
        if len(df):
            df = df.sort_values(["global_id", "start_seconds", "camera_index"], kind="stable").reset_index(drop=True)
        return df

    @staticmethod
    def identities_dataframe(identities: List[GlobalIdentity]) -> pd.DataFrame:
        rows = []
        for ident in identities:
            first = ident.first_seen if np.isfinite(ident.first_seen) else 0.0
            rows.append({
                "global_id": ident.global_id,
                "cameras_seen": ",".join(ident.camera_ids),
                "camera_names": ", ".join(ident.camera_names),
                "num_cameras": len(ident.cameras_seen),
                "first_seen": format_seconds(first),
                "last_seen": format_seconds(ident.last_seen),
                "first_seen_seconds": round(first, 3),
                "last_seen_seconds": round(ident.last_seen, 3),
                "total_duration": format_seconds(ident.total_duration),
                "total_duration_seconds": round(ident.total_duration, 3),
                "number_of_tracks": ident.num_tracks,
                "gallery_size": len(ident.embeddings),
                "uncertain_tracks": sum(1 for v in ident.visits if v.status == "UNCERTAIN"),
            })
        return pd.DataFrame(rows, columns=IDENTITY_COLUMNS)

    @staticmethod
    def build_summary(camera_results: List[CameraResult], identities: List[GlobalIdentity], records: List[TrackRecord],
                      processing_time: float, identity_stats: Optional[Dict] = None) -> Dict:
        frames = sum(r.frames_processed for r in camera_results)
        assigned = [r for r in records if r.global_id]
        multi_cam = sum(1 for i in identities if len(i.cameras_seen) > 1)
        return {
            "total_cameras": len(camera_results),
            "cameras_ok": sum(1 for r in camera_results if r.status == "done"),
            "cameras_failed": sum(1 for r in camera_results if r.status == "error"),
            "total_unique_persons": len(identities),
            "persons_seen_in_multiple_cameras": multi_cam,
            "total_tracks": len(records),
            "assigned_tracks": len(assigned),
            "short_or_unassigned_tracks": len(records) - len(assigned),
            "matched_tracks": sum(1 for r in assigned if r.status == "MATCHED"),
            "new_id_tracks": sum(1 for r in assigned if r.status == "NEW ID"),
            "uncertain_tracks": sum(1 for r in assigned if r.status == "UNCERTAIN"),
            "frames_processed": frames,
            "processing_time_seconds": round(processing_time, 2),
            "average_fps": round(frames / processing_time, 2) if processing_time > 0 else 0.0,
            "per_camera": [
                {
                    "camera_id": r.camera_id,
                    "camera_name": r.camera_name,
                    "status": r.status,
                    "error": r.error,
                    "frames_processed": r.frames_processed,
                    "processing_time_seconds": round(r.processing_time, 2),
                    "fps": round(r.processing_fps, 2),
                    "tracks": r.n_tracks,
                    "assigned_tracks": r.n_assigned_tracks,
                    "output_video": str(r.output_video) if r.output_video else "",
                }
                for r in camera_results
            ],
            "identity_manager_stats": dict(identity_stats or {}),
        }

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def write_reports(self, result: ProcessingResult) -> Dict[str, Path]:
        reports = ensure_dir(self.output_dir / "reports")
        paths: Dict[str, Path] = {}
        tracks_path = reports / "tracks.csv"
        result.tracks_df.to_csv(tracks_path, index=False)
        paths["tracks_csv"] = tracks_path
        identities_path = reports / "identities.csv"
        result.identities_df.to_csv(identities_path, index=False)
        paths["identities_csv"] = identities_path
        summary_path = reports / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(result.summary, fh, indent=2)
        paths["summary_json"] = summary_path
        config_path = reports / "config_used.json"
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(result.config_snapshot, fh, indent=2, default=str)
        paths["config_json"] = config_path
        log.info("Reports written to %s", reports)
        return paths

    def write_embeddings(self, idm: GlobalIdentityManager, records: List[TrackRecord]) -> Path:
        emb_dir = ensure_dir(self.output_dir / "embeddings")
        path = emb_dir / "embeddings.npz"
        arrays = idm.export_embeddings()
        track_rows = [(r, r.mean_embedding) for r in records if r.mean_embedding is not None]
        dim = arrays["gallery_embeddings"].shape[1] if arrays["gallery_embeddings"].size else (
            track_rows[0][1].shape[0] if track_rows else 0)
        arrays["track_embeddings"] = (np.stack([e for _, e in track_rows], axis=0)
                                      if track_rows else np.zeros((0, dim), dtype=np.float32))
        arrays["track_keys"] = np.asarray([f"{r.camera_id}:T{r.local_track_id}" for r, _ in track_rows], dtype=str)
        arrays["track_global_ids"] = np.asarray([r.global_id for r, _ in track_rows], dtype=str)
        np.savez_compressed(path, **arrays)
        log.info("Embeddings saved to %s (%d gallery vectors, %d track vectors)", path,
                 len(arrays["gallery_embeddings"]), len(arrays["track_embeddings"]))
        return path

    def write_crops(self, identities: List[GlobalIdentity], records: List[TrackRecord]) -> Path:
        """Representative crops only: one per (identity, camera, track) plus one per identity."""
        crops_dir = ensure_dir(self.output_dir / "crops")
        n = 0
        for ident in identities:
            ident_dir = ensure_dir(crops_dir / ident.global_id)
            if ident.representative_image:
                (ident_dir / "representative.jpg").write_bytes(ident.representative_image)
                n += 1
            for cam_id, jpeg in ident.camera_images.items():
                cam_dir = ensure_dir(ident_dir / cam_id)
                (cam_dir / "best.jpg").write_bytes(jpeg)
                n += 1
        for r in records:
            if r.global_id and r.best_crop_jpeg:
                cam_dir = ensure_dir(crops_dir / r.global_id / r.camera_id)
                (cam_dir / f"{r.camera_id}_T{r.local_track_id:02d}_f{r.start_frame}.jpg").write_bytes(r.best_crop_jpeg)
                n += 1
        log.info("Saved %d representative crops under %s", n, crops_dir)
        return crops_dir

    # ------------------------------------------------------------------ #
    # Session persistence (lets the dashboard reload the last run)
    # ------------------------------------------------------------------ #
    def save_state(self, result: ProcessingResult) -> Path:
        reports = ensure_dir(self.output_dir / "reports")
        path = reports / "last_run.pkl"
        with open(path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_state(self) -> Optional[ProcessingResult]:
        path = self.output_dir / "reports" / "last_run.pkl"
        if not path.is_file():
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            log.warning("Could not load previous results: %s", exc)
            return None


def human_summary_lines(summary: Dict) -> List[str]:
    return [
        f"Cameras: {summary.get('total_cameras', 0)} (ok {summary.get('cameras_ok', 0)}, failed {summary.get('cameras_failed', 0)})",
        f"Unique persons: {summary.get('total_unique_persons', 0)} ({summary.get('persons_seen_in_multiple_cameras', 0)} seen in 2+ cameras)",
        f"Tracks: {summary.get('total_tracks', 0)} (assigned {summary.get('assigned_tracks', 0)})",
        f"Processing time: {format_duration(summary.get('processing_time_seconds', 0.0))}, avg {summary.get('average_fps', 0)} fps",
    ]
