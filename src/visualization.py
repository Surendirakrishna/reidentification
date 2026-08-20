"""Drawing helpers for annotated output videos."""

from __future__ import annotations

import colorsys
import hashlib
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]

# A fixed, high-contrast palette (BGR) - identities beyond the palette get hashed hues.
_PALETTE: Sequence[Color] = (
    (56, 181, 255), (255, 128, 0), (0, 200, 83), (255, 64, 129), (255, 214, 0), (156, 39, 176),
    (0, 172, 193), (233, 30, 99), (76, 175, 80), (255, 87, 34), (3, 169, 244), (205, 220, 57),
    (121, 85, 72), (96, 125, 139), (244, 67, 54), (103, 58, 183), (0, 150, 136), (255, 193, 7),
)

PENDING_COLOR: Color = (160, 160, 160)
UNCERTAIN_COLOR: Color = (0, 165, 255)


def color_for_id(global_id: Optional[str], fallback: Color = PENDING_COLOR) -> Color:
    """Deterministic colour per Global ID (text labels remain the primary cue)."""
    if not global_id:
        return fallback
    digits = "".join(ch for ch in global_id if ch.isdigit())
    if digits:
        idx = int(digits)
        if idx - 1 < len(_PALETTE):
            return _PALETTE[(idx - 1) % len(_PALETTE)]
    h = int(hashlib.md5(global_id.encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_size(text: str, scale: float, thickness: int):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h + base


def draw_label_box(frame: np.ndarray, x: int, y: int, lines: Sequence[str], color: Color,
                   scale: float = 0.5, thickness: int = 1, pad: int = 4, above: bool = True) -> None:
    """Draw a filled label box with several text lines anchored at (x, y)."""
    if not lines:
        return
    sizes = [_text_size(t, scale, thickness) for t in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    line_h = max(h for _, h in sizes) + 2
    box_h = line_h * len(lines) + 2 * pad
    h_img, w_img = frame.shape[:2]
    x0 = int(max(0, min(x, w_img - box_w)))
    y0 = int(y - box_h) if above else int(y)
    if y0 < 0:
        y0 = int(y)  # not enough room above -> draw below the anchor
    y0 = int(max(0, min(y0, h_img - box_h)))
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), color, -1)
    # Pick black or white text depending on the fill brightness.
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if lum > 150 else (255, 255, 255)
    for i, text in enumerate(lines):
        ty = y0 + pad + line_h * (i + 1) - 3
        cv2.putText(frame, text, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


def draw_track(
    frame: np.ndarray,
    bbox: Sequence[float],
    local_label: str,
    global_id: Optional[str] = None,
    confidence: Optional[float] = None,
    similarity: Optional[float] = None,
    status: str = "",
    thickness: int = 2,
    font_scale: float = 0.5,
) -> None:
    """Draw one tracked person: bounding box + multi-line label.

    Label layout:
        P001 | C1-T03          (global id + local track id)
        Person 0.91            (detector confidence)
        ReID 0.83 MATCHED      (similarity + status)
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    # Colour encodes the identity (consistent across cameras); the status lives in the text.
    color = color_for_id(global_id) if global_id else PENDING_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    head = f"{global_id} | {local_label}" if global_id else f"{local_label} | pending"
    lines = [head]
    compact = (x2 - x1) < 55 or (y2 - y1) < 110  # small/distant person -> one-line label only
    if compact:
        draw_label_box(frame, x1, y1, lines, color, scale=max(0.4, font_scale - 0.1), thickness=1)
        return
    if confidence is not None:
        lines.append(f"Person {confidence:.2f}")
    if global_id:
        if status == "NEW ID":
            lines.append("ReID new ID")
        elif status == "UNCERTAIN":
            # similarity here is the score of the closest *other* identity (below threshold)
            lines.append(f"ReID UNCERTAIN ({similarity:.2f})" if similarity else "ReID UNCERTAIN")
        elif similarity is not None and similarity > 0:
            lines.append(f"ReID {similarity:.2f} {status}".rstrip())
        elif status:
            lines.append(f"ReID {status}")
    draw_label_box(frame, x1, y1, lines, color, scale=font_scale, thickness=1)
    # A small tag with the Global ID at the bottom-right corner helps when labels overlap.
    if global_id:
        tag_scale = max(0.4, font_scale - 0.05)
        tw, th = _text_size(global_id, tag_scale, 1)
        tx, ty = max(0, x2 - tw - 6), min(frame.shape[0] - 2, y2)
        cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 6, ty), color, -1)
        cv2.putText(frame, global_id, (tx + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, camera_name: str, frame_idx: int, total_frames: int, timestamp: str,
             n_tracks: int, n_identities: int, fps: float = 0.0) -> None:
    """Top-left heads-up display with camera name, frame counter and stats."""
    text = f"{camera_name} | frame {frame_idx}/{total_frames} | {timestamp} | tracks {n_tracks} | global IDs {n_identities}"
    if fps:
        text += f" | {fps:.1f} fps"
    scale = 0.6 if frame.shape[1] >= 1280 else 0.5
    tw, th = _text_size(text, scale, 1)
    cv2.rectangle(frame, (0, 0), (tw + 12, th + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def make_thumbnail_grid(images: Sequence[np.ndarray], cols: int = 4, thumb_size: Tuple[int, int] = (128, 256),
                        labels: Optional[Sequence[str]] = None) -> np.ndarray:
    """Assemble crops into a grid image (used for quick visual checks / gallery export)."""
    if not images:
        return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
    w, h = thumb_size
    rows = int(np.ceil(len(images) / cols))
    grid = np.full((rows * h, cols * w, 3), 40, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = thumb
        if labels and i < len(labels):
            cv2.putText(grid, str(labels[i]), (c * w + 4, r * h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return grid
