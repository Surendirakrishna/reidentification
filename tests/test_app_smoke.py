"""End-to-end smoke test of the Streamlit app via streamlit.testing (slow; needs Input Data videos)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.video_utils import discover_videos

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_streamlit_app_processes_one_camera(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    videos = discover_videos(ROOT / "Input Data")
    if not videos:
        pytest.skip("no videos in Input Data")
    monkeypatch.chdir(ROOT)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    at.run()
    assert not at.exception, at.exception
    # Sidebar shows the discovered cameras as checkboxes.
    labels = [cb.label for cb in at.sidebar.checkbox]
    assert any(videos[0].path.name in lbl for lbl in labels)

    # Keep only the first camera, process a short slice, write into a temp output folder.
    for cb in at.sidebar.checkbox:
        if cb.key and cb.key.startswith("cam_"):
            cb.set_value(cb.key == f"cam_{videos[0].camera_id}")
    for key, value in {"max_frames": 60, "frame_skip": 1, "start_frame": 3000}.items():
        [w for w in at.sidebar.number_input if w.key == key][0].set_value(value)
    at.run()
    assert not at.exception, at.exception

    # Output directory: patch the config in session state so the test does not touch ./Output.
    cfg = at.session_state["base_config"]
    cfg.output.directory = str(tmp_path / "Output")
    at.session_state["base_config"] = cfg
    at.run()

    [b for b in at.sidebar.button if b.key == "start_btn"][0].click().run()
    assert not at.exception, at.exception

    deadline = time.time() + 600
    while time.time() < deadline and "result" not in at.session_state:
        time.sleep(2)
        at.run()
        assert not at.exception, at.exception
    assert "result" in at.session_state, "processing did not finish in time"
    result = at.session_state["result"]
    assert result.summary["total_cameras"] == 1
    assert result.summary["frames_processed"] > 0
    assert (tmp_path / "Output" / "reports" / "tracks.csv").is_file()
    assert (tmp_path / "Output" / "reports" / "summary.json").is_file()
    # Dashboard tabs rendered without errors.
    assert any("Dashboard" in t.label for t in at.tabs)
