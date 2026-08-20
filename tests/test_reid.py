"""Tests for embedding maths, crop quality and (if weights are available) real OSNet inference."""

from __future__ import annotations

import numpy as np
import pytest

from src.reid import assess_crop_quality, cosine_similarity, cosine_similarity_matrix, l2_normalize
from src.config import resolve_osnet_weights_name
from tests.conftest import make_person_like_frame


def test_l2_normalize_unit_norm(rng):
    x = rng.standard_normal((5, 512)).astype(np.float32) * 10
    y = l2_normalize(x, axis=1)
    assert y.shape == x.shape
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-5)


def test_l2_normalize_zero_vector_is_safe():
    z = np.zeros(8, dtype=np.float32)
    out = l2_normalize(z)
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, [0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)  # scale invariant
    assert cosine_similarity(a, np.zeros(3)) == 0.0


def test_cosine_similarity_matrix_shape_and_values(unit_vectors):
    a, b = unit_vectors(3), unit_vectors(4)
    m = cosine_similarity_matrix(a, b)
    assert m.shape == (3, 4)
    assert m[0, 0] == pytest.approx(float(a[0] @ b[0]), abs=1e-5)
    self_sim = cosine_similarity_matrix(a, a)
    np.testing.assert_allclose(np.diag(self_sim), 1.0, atol=1e-5)
    assert cosine_similarity_matrix(np.zeros((0, 5)), b).shape == (0, 4)


def test_crop_quality_rejects_small_and_empty():
    assert assess_crop_quality(None).ok is False
    tiny = np.zeros((20, 10, 3), dtype=np.uint8)
    q = assess_crop_quality(tiny, min_width=32, min_height=64)
    assert q.ok is False and q.reason == "too_small"


def test_crop_quality_scores_reasonable_crop():
    frame = make_person_like_frame()
    crop = frame[60:400, 270:370]
    q = assess_crop_quality(crop, det_conf=0.9, occlusion=0.0)
    assert q.ok and 0.0 < q.score <= 1.0
    occluded = assess_crop_quality(crop, det_conf=0.9, occlusion=0.8)
    assert occluded.score < q.score  # occlusion lowers the quality


def test_resolve_osnet_weights_name():
    assert resolve_osnet_weights_name("osnet_x1_0") == "osnet_x1_0_msmt17.pt"
    assert resolve_osnet_weights_name("osnet_x1_0_market1501") == "osnet_x1_0_market1501.pt"
    assert resolve_osnet_weights_name("osnet_ain_x1_0_msmt17.pt") == "osnet_ain_x1_0_msmt17.pt"


@pytest.mark.slow
def test_osnet_real_embeddings_are_normalised_and_consistent():
    """Real OSNet inference (downloads weights on first run). Skipped if unavailable/offline."""
    torch = pytest.importorskip("torch")
    from src.reid import OSNetReID

    try:
        reid = OSNetReID(model="osnet_x1_0_msmt17.pt", device="auto", batch_size=4, models_dir="models")
    except Exception as exc:  # pragma: no cover - offline / download failure
        pytest.skip(f"OSNet not available: {exc}")
    frame = make_person_like_frame()
    crop = frame[60:400, 270:370]
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(300, 120, 3), dtype=np.uint8)
    emb = reid.extract([crop, crop.copy(), noise, noise, noise])  # 5 crops -> 2 batches of 4 + 1
    assert emb.shape == (5, reid.embedding_dim)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    # Deterministic: identical crops -> identical embeddings; different content -> lower similarity.
    assert cosine_similarity(emb[0], emb[1]) == pytest.approx(1.0, abs=1e-3)
    assert cosine_similarity(emb[0], emb[2]) < 0.99
    # Embeddings are real model outputs, not random: repeated calls agree.
    again = reid.extract([crop])
    assert cosine_similarity(emb[0], again[0]) == pytest.approx(1.0, abs=1e-3)
