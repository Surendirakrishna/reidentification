"""Video discovery, reading, writing and bounding-box helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import VIDEO_EXTENSIONS
from .utils import get_logger, infer_camera_name, sanitize_camera_id

log = get_logger("video")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class VideoSource:
    """A video file that represents one camera."""

    path: Path
    camera_id: str  # filesystem safe id, e.g. "cam1"
    camera_name: str  # display name, e.g. "Camera 1"
    index: int  # 1-based index used for the short label C1, C2, ...
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    readable: Optional[bool] = None  # None = not probed yet

    @property
    def short_label(self) -> str:
        return f"C{self.index}"

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps and self.frame_count else 0.0

    def probe(self) -> "VideoSource":
        """Read container metadata (cheap, does not decode frames)."""
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                self.readable = False
                return self
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.readable = self.width > 0 and self.height > 0
        finally:
            cap.release()
        return self


def discover_videos(
    input_dir: str | os.PathLike,
    extensions: Sequence[str] = VIDEO_EXTENSIONS,
    probe: bool = False,
) -> List[VideoSource]:
    """Scan `input_dir` for video files and build VideoSource entries.

    Files are sorted naturally (cam2 before cam10). Camera ids are made unique.
    """
    root = Path(input_dir)
    if not root.is_dir():
        return []
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=_natural_key)

    sources: List[VideoSource] = []
    seen_ids: dict[str, int] = {}
    for idx, path in enumerate(files, start=1):
        cam_id = sanitize_camera_id(path.name)
        if cam_id in seen_ids:
            seen_ids[cam_id] += 1
            cam_id = f"{cam_id}_{seen_ids[cam_id]}"
        else:
            seen_ids[cam_id] = 1
        src = VideoSource(path=path, camera_id=cam_id, camera_name=infer_camera_name(path.name), index=idx)
        if probe:
            src.probe()
        sources.append(src)
    return sources


def _natural_key(path: Path):
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
class VideoReader:
    """Thin wrapper over cv2.VideoCapture with frame skipping and metadata."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not self.fps or self.fps <= 0 or self.fps > 1000:
            log.warning("Video %s reports invalid FPS (%s); assuming 30", self.path.name, self.fps)
            self.fps = 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.width <= 0 or self.height <= 0:
            self.cap.release()
            raise IOError(f"Video has invalid dimensions (unsupported codec?): {self.path}")

    def seek(self, frame_idx: int) -> None:
        if frame_idx > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def frames(self, frame_skip: int = 0, start_frame: int = 0, max_frames: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame_bgr). `frame_skip` frames are dropped between yields."""
        frame_skip = max(0, int(frame_skip))
        self.seek(start_frame)
        idx = int(start_frame)
        yielded = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            yielded += 1
            idx += 1
            if max_frames and yielded >= max_frames:
                break
            for _ in range(frame_skip):
                if not self.cap.grab():
                    return
                idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()


def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable (imageio-ffmpeg bundle or PATH)."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _encoder_works(ffmpeg: str, encoder: str) -> bool:
    """Probe whether `encoder` can actually encode on this machine."""
    key = (ffmpeg, encoder)
    with _ENCODER_LOCK:
        if key in _ENCODER_CACHE:
            return _ENCODER_CACHE[key]
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=128x128:d=0.2:r=10",
        "-c:v", encoder, "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok = res.returncode == 0
    except Exception:
        ok = False
    with _ENCODER_LOCK:
        _ENCODER_CACHE[key] = ok
    return ok


def select_video_backend(preference: str = "auto") -> Tuple[str, Optional[str]]:
    """Return (backend, encoder): backend is 'ffmpeg' or 'opencv'."""
    preference = (preference or "auto").lower()
    ffmpeg = find_ffmpeg()
    if preference in ("mp4v", "avc1", "opencv"):
        return "opencv", preference if preference != "opencv" else "mp4v"
    if ffmpeg:
        candidates = ["h264_nvenc", "libx264"] if preference == "auto" else [preference]
        for enc in candidates:
            if _encoder_works(ffmpeg, enc):
                return "ffmpeg", enc
        if preference != "auto" and _encoder_works(ffmpeg, "libx264"):
            log.warning("Encoder %s unavailable; falling back to libx264", preference)
            return "ffmpeg", "libx264"
    return "opencv", "mp4v"


class VideoWriter:
    """Writes BGR frames to an MP4 file.

    Prefers ffmpeg (H.264 -> plays in browsers / Streamlit). Falls back to
    OpenCV's VideoWriter (mp4v), which may not play inline in a browser.
    """

    def __init__(self, path: str | os.PathLike, fps: float, width: int, height: int, codec: str = "auto"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = int(width)
        self.height = int(height)
        self.frames_written = 0
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: List[str] = []
        self.backend, self.encoder = select_video_backend(codec)
        if self.backend == "ffmpeg":
            self._open_ffmpeg()
        if self._proc is None:
            self._open_opencv(self.encoder if self.backend == "opencv" else "mp4v")

    # ------------------------------------------------------------------ #
    def _open_ffmpeg(self) -> None:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return
        # H.264 requires even dimensions for yuv420p.
        out_w, out_h = self.width - (self.width % 2), self.height - (self.height % 2)
        vf = []
        if (out_w, out_h) != (self.width, self.height):
            vf = ["-vf", f"crop={out_w}:{out_h}:0:0"]
        enc_args = ["-c:v", self.encoder, "-pix_fmt", "yuv420p"]
        if self.encoder == "libx264":
            enc_args += ["-preset", "veryfast", "-crf", "23"]
        elif self.encoder == "h264_nvenc":
            enc_args += ["-preset", "p4", "-cq", "23"]
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:.4f}", "-i", "-",
            "-an", *vf, *enc_args, "-movflags", "+faststart", str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # pragma: no cover - depends on machine
            log.warning("Could not start ffmpeg (%s); falling back to OpenCV writer", exc)
            self._proc = None
            return
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.backend = "ffmpeg"

    def _drain_stderr(self) -> None:
        try:
            assert self._proc is not None and self._proc.stderr is not None
            for raw in self._proc.stderr:
                line = raw.decode("utf-8", "ignore").strip()
                if line:
                    self._stderr_tail.append(line)
                    self._stderr_tail = self._stderr_tail[-20:]
        except Exception:
            pass

    def _open_opencv(self, fourcc_name: str = "mp4v") -> None:
        fourcc = cv2.VideoWriter.fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened() and fourcc_name != "mp4v":
            writer.release()
            fourcc_name = "mp4v"
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for {self.path}")
        self._cv_writer = writer
        self.backend, self.encoder = "opencv", fourcc_name

    # ------------------------------------------------------------------ #
    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self._proc is not None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError) as exc:
                tail = " | ".join(self._stderr_tail[-3:])
                raise IOError(f"ffmpeg writer failed: {exc} {tail}") from exc
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=120)
            except Exception:
                self._proc.kill()
            finally:
                if self._proc.returncode not in (0, None):
                    log.warning("ffmpeg exited with code %s: %s", self._proc.returncode, " | ".join(self._stderr_tail[-3:]))
                self._proc = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Bounding boxes / crops
# --------------------------------------------------------------------------- #
def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp an (x1, y1, x2, y2) box to image bounds and return ints."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(max(0, min(width, np.floor(x1))))
    y1 = int(max(0, min(height, np.floor(y1))))
    x2 = int(max(0, min(width, np.ceil(x2))))
    y2 = int(max(0, min(height, np.ceil(y2))))
    return x1, y1, x2, y2


def is_valid_bbox(bbox: Sequence[float], width: int, height: int, min_w: int = 1, min_h: int = 1) -> bool:
    """A valid box is finite, inside the image (after clamping) and at least min_w x min_h."""
    try:
        vals = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return False
    if len(vals) < 4 or any(not np.isfinite(v) for v in vals):
        return False
    x1, y1, x2, y2 = clamp_bbox(vals, width, height)
    return (x2 - x1) >= min_w and (y2 - y1) >= min_h


def crop_person(frame: np.ndarray, bbox: Sequence[float], margin: float = 0.0) -> Optional[np.ndarray]:
    """Crop a person from a BGR frame (view, not copy). Returns None for invalid boxes."""
    h, w = frame.shape[:2]
    if margin:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        bbox = (x1 - bw * margin, y1 - bh * margin, x2 + bw * margin, y2 + bh * margin)
    if not is_valid_bbox(bbox, w, h):
        return None
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def encode_jpeg(image: np.ndarray, quality: int = 85, max_side: int = 0) -> bytes:
    """Encode a BGR image as JPEG bytes, optionally downscaling the longest side."""
    if max_side and max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
