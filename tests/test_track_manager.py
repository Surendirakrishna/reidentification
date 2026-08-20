"""Tests for TrackManager (track lifecycle -> embeddings -> Global ID decisions) without any models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.camera_processor import TrackManager
from src.global_identity import GlobalIdentityManager, MatchStatus
from src.tracker import TrackObservation
from src.video_utils import VideoSource


def _cam(idx: int = 1) -> VideoSource:
    return VideoSource(path=Path(f"cam{idx}.mp4"), camera_id=f"cam{idx}", camera_name=f"Camera {idx}", index=idx)


def _obs(tid: int, x: float = 10) -> TrackObservation:
    return TrackObservation(track_id=tid, bbox=np.array([x, 10, x + 40, 110], dtype=np.float32),
                            confidence=0.9, class_id=0, det_index=0)


def test_short_track_is_ignored(unit_vectors):
    idm = GlobalIdentityManager()
    tm = TrackManager(_cam(), idm, min_track_frames=10, embedding_interval=5, max_age=3)
    emb = unit_vectors(1)[0]
    for step in range(4):
        observed, _ = tm.update([_obs(1)], frame_idx=step, t_sec=step / 30)
        for ts in tm.tracks_needing_embedding(observed, step):
            tm.add_embedding(ts, emb, 0.8, step, step)
    # Track disappears; after max_age updates without it, it is finalized as SHORT.
    for step in range(4, 10):
        _, finished = tm.update([], frame_idx=step, t_sec=step / 30)
        if finished:
            break
    assert finished and finished[0].status == MatchStatus.SHORT.value
    assert finished[0].global_id == "" and len(idm) == 0


def test_embedding_schedule_first_then_interval(unit_vectors):
    idm = GlobalIdentityManager()
    tm = TrackManager(_cam(), idm, min_track_frames=2, embedding_interval=5, max_age=3)
    emb = unit_vectors(1)[0]
    due_steps = []
    for step in range(12):
        observed, _ = tm.update([_obs(1)], frame_idx=step, t_sec=step / 30)
        for ts in tm.tracks_needing_embedding(observed, step):
            due_steps.append(step)
            tm.add_embedding(ts, emb, 0.8, step, step)
    assert due_steps == [0, 5, 10]


def test_track_gets_global_id_and_second_track_matches(unit_vectors, rng):
    idm = GlobalIdentityManager(similarity_threshold=0.65)
    tm = TrackManager(_cam(), idm, min_track_frames=3, embedding_interval=2, max_age=2)
    base = unit_vectors(1)[0]
    for step in range(6):
        observed, _ = tm.update([_obs(1)], frame_idx=step, t_sec=step / 30)
        for ts in tm.tracks_needing_embedding(observed, step):
            tm.add_embedding(ts, base, 0.8, step, step)
        for ts in observed:
            if tm.ready_for_decision(ts):
                tm.try_resolve(ts)
    ts1 = tm.active[1]
    assert ts1.assigned and ts1.global_id == "P001" and ts1.status == MatchStatus.NEW.value
    # Track 1 leaves, later track 2 (same appearance) appears -> MATCHED to P001.
    for step in range(6, 12):
        tm.update([], frame_idx=step, t_sec=step / 30)
    assert 1 not in tm.active and tm.records[0].global_id == "P001"
    noisy = base + rng.standard_normal(base.shape).astype(np.float32) * 0.01
    noisy /= np.linalg.norm(noisy)
    for step in range(12, 18):
        observed, _ = tm.update([_obs(2, x=200)], frame_idx=step, t_sec=step / 30)
        for ts in tm.tracks_needing_embedding(observed, step):
            tm.add_embedding(ts, noisy, 0.8, step, step)
        for ts in observed:
            if tm.ready_for_decision(ts):
                tm.try_resolve(ts)
    ts2 = tm.active[2]
    assert ts2.global_id == "P001" and ts2.status == MatchStatus.MATCHED.value and ts2.similarity > 0.9
    records = tm.finalize_all()
    assert len(records) == 1 and records[0].local_label == "C1-T02"
    assert idm.get("P001").num_tracks == 2


def test_covisible_tracks_never_share_an_id(unit_vectors):
    idm = GlobalIdentityManager(similarity_threshold=0.65, max_pending_embeddings=3)
    tm = TrackManager(_cam(), idm, min_track_frames=2, embedding_interval=1, max_age=2)
    same = unit_vectors(1)[0]  # two simultaneously visible tracks with identical appearance
    for step in range(6):
        observed, _ = tm.update([_obs(1, x=10), _obs(2, x=300)], frame_idx=step, t_sec=step / 30)
        for ts in tm.tracks_needing_embedding(observed, step):
            tm.add_embedding(ts, same, 0.8, step, step)
        for ts in observed:
            if tm.ready_for_decision(ts):
                tm.try_resolve(ts)
    gids = {tm.active[1].global_id, tm.active[2].global_id}
    assert None not in gids and len(gids) == 2
    assert idm.stats["covisibility_rejections"] >= 1
