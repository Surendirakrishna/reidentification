"""Tests for video discovery, bounding-box validation, cropping and the video writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils import format_seconds, infer_camera_name, parse_clock_time, sanitize_camera_id
from src.video_utils import (
    VideoReader,
    VideoWriter,
    clamp_bbox,
    crop_person,
    decode_jpeg,
    discover_videos,
    encode_jpeg,
    is_valid_bbox,
)
from tests.conftest import make_person_like_frame


def test_infer_camera_name():
    assert infer_camera_name("cam1.mp4") == "Camera 1"
    assert infer_camera_name("camera_02.mp4") == "Camera 2"
    assert infer_camera_name("Cam-7.avi") == "Camera 7"
    assert infer_camera_name("entrance.mp4") == "Entrance"
    assert infer_camera_name("parking_lot.mkv") == "Parking Lot"
    assert sanitize_camera_id("Camera 02 (front).mp4") == "camera_02_front"


def test_discover_videos_filters_and_sorts(tmp_path: Path):
    for name in ["cam10.mp4", "cam2.MP4", "notes.txt", "cam1.avi", "lobby.mov", "clip.mkv", "image.jpg"]:
        (tmp_path / name).write_bytes(b"\x00" * 16)
    (tmp_path / "subdir").mkdir()
    sources = discover_videos(tmp_path)
    names = [s.path.name for s in sources]
    assert names == ["cam1.avi", "cam2.MP4", "cam10.mp4", "clip.mkv", "lobby.mov"]
    assert [s.index for s in sources] == [1, 2, 3, 4, 5]
    assert sources[0].camera_name == "Camera 1" and sources[0].short_label == "C1"
    assert sources[2].camera_name == "Camera 10"
    assert sources[4].camera_id == "lobby"
    assert discover_videos(tmp_path / "does_not_exist") == []
    assert discover_videos(tmp_path, extensions=[".mov"])[0].path.name == "lobby.mov"


def test_discover_videos_unique_ids(tmp_path: Path):
    (tmp_path / "Cam 1.mp4").write_bytes(b"0")
    (tmp_path / "cam_1.mov").write_bytes(b"0")
    ids = [s.camera_id for s in discover_videos(tmp_path)]
    assert len(set(ids)) == 2


def test_bbox_validation_and_clamping():
    w, h = 640, 480
    assert is_valid_bbox([10, 20, 110, 220], w, h)
    assert not is_valid_bbox([10, 20, 10, 220], w, h)  # zero width
    assert not is_valid_bbox([-50, -50, -10, -10], w, h)  # entirely outside
    assert not is_valid_bbox([0, 0, float("nan"), 10], w, h)
    assert not is_valid_bbox([0, 0, 5, 5], w, h, min_w=10, min_h=10)
    assert clamp_bbox([-10.4, -5, 700.2, 500], w, h) == (0, 0, 640, 480)
    assert clamp_bbox([100.6, 50.2, 20.1, 10.9], w, h) == (20, 10, 101, 51)  # swapped corners fixed


def test_crop_person_returns_view_and_rejects_bad_boxes():
    frame = make_person_like_frame()
    crop = crop_person(frame, [270, 60, 370, 400])
    assert crop is not None and crop.shape == (340, 100, 3)
    assert crop_person(frame, [700, 700, 800, 800]) is None
    partial = crop_person(frame, [600, 400, 700, 500])  # clamped to the frame
    assert partial is not None and partial.shape == (80, 40, 3)
    with_margin = crop_person(frame, [270, 60, 370, 400], margin=0.1)
    assert with_margin.shape[0] > crop.shape[0]


def test_jpeg_roundtrip():
    frame = make_person_like_frame(200, 300)
    data = encode_jpeg(frame, quality=90, max_side=150)
    img = decode_jpeg(data)
    assert img.shape[0] == 150 and img.shape[1] == 100


def test_video_writer_and_reader_roundtrip(tmp_path: Path):
    path = tmp_path / "out.mp4"
    frames = 12
    with VideoWriter(path, fps=10, width=320, height=240, codec="auto") as writer:
        for i in range(frames):
            frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
            cv2.putText(frame, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            writer.write(frame)
        backend = writer.backend
    assert path.is_file() and path.stat().st_size > 0
    with VideoReader(path) as reader:
        assert reader.width == 320 and reader.height == 240
        read = sum(1 for _ in reader.frames())
    assert read == frames, f"{backend} writer produced {read} frames"
    with VideoReader(path) as reader:
        idx = [i for i, _ in reader.frames(frame_skip=2, start_frame=0, max_frames=3)]
    assert idx == [0, 3, 6]


def test_video_reader_rejects_missing_file(tmp_path: Path):
    with pytest.raises(IOError):
        VideoReader(tmp_path / "missing.mp4")


def test_time_helpers():
    assert format_seconds(3725.5) == "01:02:05"
    assert format_seconds(0.25, with_millis=True) == "00:00:00.250"
    assert parse_clock_time("10:02:11") == 36131
    assert parse_clock_time("02:11") == 131
