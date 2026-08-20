"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def unit_vectors(rng):
    """Helper producing random L2-normalised vectors."""

    def _make(n: int = 1, dim: int = 512) -> np.ndarray:
        v = rng.standard_normal((n, dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return _make


def make_person_like_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Synthetic BGR frame with a vertical 'person-like' rectangle (used for crop tests)."""
    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    frame[100:400, 280:360] = (30, 60, 200)  # body
    frame[60:100, 300:340] = (180, 200, 230)  # head
    return frame
