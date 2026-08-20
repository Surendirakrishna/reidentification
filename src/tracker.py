"""OC-SORT multi-object tracker (BoxMOT implementation) - one instance per camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import TrackerConfig
from .utils import get_logger

log = get_logger("tracker")


@dataclass
class TrackObservation:
    """One tracked person in the current frame (local to a single camera)."""

    track_id: int
    bbox: np.ndarray  # x1, y1, x2, y2 (float32)
    confidence: float
    class_id: int
    det_index: int  # index into the detections passed to update(), -1 if unmatched


class OCSortTracker:
    """Thin, typed wrapper around `boxmot.trackers.bbox.ocsort.OcSort`.

    Input detections: np.ndarray (N, 6) -> x1, y1, x2, y2, conf, cls
    Output: list[TrackObservation] for tracks that are active in this frame.
    """

    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        asso_func: str = "iou",
        use_byte: bool = False,
        min_conf: float = 0.1,
    ):
        from boxmot.trackers.bbox.ocsort import OcSort  # real OC-SORT implementation

        self.params = dict(
            det_thresh=float(det_thresh),
            max_age=int(max_age),
            min_hits=int(min_hits),
            iou_threshold=float(iou_threshold),
            delta_t=int(delta_t),
            inertia=float(inertia),
            asso_func=asso_func,
            use_byte=bool(use_byte),
            min_conf=float(min_conf),
        )
        self._tracker = OcSort(
            det_thresh=self.params["det_thresh"],
            max_age=self.params["max_age"],
            min_hits=self.params["min_hits"],
            iou_threshold=self.params["iou_threshold"],
            delta_t=self.params["delta_t"],
            inertia=self.params["inertia"],
            asso_func=self.params["asso_func"],
            use_byte=self.params["use_byte"],
            min_conf=self.params["min_conf"],
        )
        self.frame_count = 0

    @property
    def max_age(self) -> int:
        return self.params["max_age"]

    def update(self, detections: Optional[np.ndarray], frame: np.ndarray) -> List[TrackObservation]:
        """Advance the tracker by one frame. Must be called once per processed frame."""
        if detections is None or len(detections) == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.ndim != 2 or dets.shape[1] < 6:
                raise ValueError(f"detections must be (N, 6) [x1,y1,x2,y2,conf,cls]; got {dets.shape}")
            dets = np.ascontiguousarray(dets[:, :6])
        self.frame_count += 1
        out = self._tracker.update(dets, frame)
        out = np.asarray(out, dtype=np.float32)
        tracks: List[TrackObservation] = []
        if out.size == 0:
            return tracks
        for row in out:
            # BoxMOT AABB layout: x1, y1, x2, y2, id, conf, cls, det_ind
            tracks.append(
                TrackObservation(
                    track_id=int(row[4]),
                    bbox=row[:4].copy(),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                    det_index=int(row[7]) if row.shape[0] > 7 else -1,
                )
            )
        return tracks

    def reset(self) -> None:
        self._tracker.reset()
        self.frame_count = 0

    @classmethod
    def from_config(cls, cfg: TrackerConfig) -> "OCSortTracker":
        if str(cfg.type).lower() not in ("ocsort", "oc-sort", "oc_sort"):
            log.warning("Tracker type %s not supported; using OC-SORT", cfg.type)
        return cls(
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_threshold,
            delta_t=cfg.delta_t,
            inertia=cfg.inertia,
            asso_func=cfg.asso_func,
            use_byte=cfg.use_byte,
        )


def create_trackers(camera_ids: List[str], cfg: TrackerConfig) -> dict:
    """Independent tracker per camera: {"camera_1": OCSortTracker, ...}."""
    return {cam: OCSortTracker.from_config(cfg) for cam in camera_ids}
