"""Tests for the OC-SORT wrapper (real BoxMOT implementation, synthetic detections)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("boxmot")

from src.config import TrackerConfig
from src.tracker import OCSortTracker, TrackObservation, create_trackers

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _det(x: float, y: float, w: float = 40, h: float = 100, conf: float = 0.9) -> list:
    return [x, y, x + w, y + h, conf, 0]


def test_tracker_keeps_consistent_id_for_moving_person():
    tracker = OCSortTracker(det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3)
    ids = []
    for step in range(15):
        dets = np.array([_det(50 + step * 4, 100 + step * 2)], dtype=np.float32)
        tracks = tracker.update(dets, FRAME)
        if tracks:
            assert isinstance(tracks[0], TrackObservation)
            ids.append(tracks[0].track_id)
    assert len(ids) >= 10  # confirmed after min_hits
    assert len(set(ids)) == 1  # same local id throughout
    last = tracker.update(np.array([_det(50 + 15 * 4, 100 + 15 * 2)], dtype=np.float32), FRAME)[0]
    assert last.bbox.shape == (4,) and last.confidence == pytest.approx(0.9, abs=1e-3)
    assert last.class_id == 0 and last.det_index == 0


def test_two_people_get_two_ids_and_empty_frames_are_ok():
    tracker = OCSortTracker(det_thresh=0.3, max_age=5, min_hits=2, iou_threshold=0.3)
    for step in range(6):
        dets = np.array([_det(50 + step * 3, 100), _det(400 - step * 3, 200)], dtype=np.float32)
        tracks = tracker.update(dets, FRAME)
    assert len(tracks) == 2
    assert len({t.track_id for t in tracks}) == 2
    # Frames without detections must not crash and simply return no active tracks.
    assert tracker.update(None, FRAME) == []
    assert tracker.update(np.zeros((0, 6), dtype=np.float32), FRAME) == []


def test_trackers_are_independent_per_camera():
    trackers = create_trackers(["cam1", "cam2"], TrackerConfig(min_hits=1))
    assert set(trackers) == {"cam1", "cam2"}
    assert trackers["cam1"] is not trackers["cam2"]
    for _ in range(3):
        t1 = trackers["cam1"].update(np.array([_det(10, 10)], dtype=np.float32), FRAME)
        t2 = trackers["cam2"].update(np.array([_det(300, 300), _det(100, 100)], dtype=np.float32), FRAME)
    # Local IDs are allocated per tracker: both cameras start counting from 0 independently.
    assert {t.track_id for t in t1} == {0}
    assert {t.track_id for t in t2} == {0, 1}


def test_invalid_detection_shape_raises():
    tracker = OCSortTracker()
    with pytest.raises(ValueError):
        tracker.update(np.zeros((2, 4), dtype=np.float32), FRAME)


def test_reset_restarts_ids():
    tracker = OCSortTracker(min_hits=1)
    for _ in range(2):
        tracker.update(np.array([_det(10, 10)], dtype=np.float32), FRAME)
    tracker.reset()
    tracks = tracker.update(np.array([_det(200, 200)], dtype=np.float32), FRAME)
    assert tracks and tracks[0].track_id == 0
