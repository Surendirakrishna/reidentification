"""Tests for the GlobalIdentityManager (cross-camera matching logic)."""

from __future__ import annotations

import numpy as np
import pytest

from src.global_identity import GlobalIdentityManager, MatchStatus


def _perturb(vec: np.ndarray, rng, scale: float) -> np.ndarray:
    noisy = vec + rng.standard_normal(vec.shape).astype(np.float32) * scale
    return noisy / np.linalg.norm(noisy)


def _resolve(mgr, embs, cam="cam1", idx=1, tid=0, t=0.0, final=True, exclude=None):
    return mgr.resolve_track(
        embs, [0.8] * len(embs), camera_id=cam, camera_name=f"Camera {idx}", camera_index=idx, local_track_id=tid,
        first_seen=t, first_frame=int(t * 30), last_seen=t + 1, last_frame=int(t * 30) + 30, final=final,
        exclude=exclude,
    )


def test_first_track_creates_new_global_id(unit_vectors):
    mgr = GlobalIdentityManager(similarity_threshold=0.65)
    res = _resolve(mgr, [unit_vectors(1)[0]], tid=3)
    assert res.status == MatchStatus.NEW and res.is_new
    assert res.global_id == "P001"
    assert len(mgr) == 1
    ident = mgr.get("P001")
    assert ident is not None and ident.num_tracks == 1 and ident.camera_ids == ["cam1"]
    assert ident.visits[0].local_label == "C1-T03"


def test_similar_embedding_from_other_camera_matches_existing_id(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.65)
    base = unit_vectors(1)[0]
    first = _resolve(mgr, [base], cam="cam1", idx=1, tid=3)
    similar = _perturb(base, rng, 0.01)  # cosine ~0.99
    second = _resolve(mgr, [similar], cam="cam2", idx=2, tid=17, t=100.0)
    assert second.status == MatchStatus.MATCHED
    assert second.global_id == first.global_id == "P001"
    assert second.similarity > 0.9
    ident = mgr.get("P001")
    assert ident.camera_ids == ["cam1", "cam2"]  # same person, two cameras
    assert ident.num_tracks == 2
    assert len(mgr) == 1  # no fragmentation


def test_dissimilar_embedding_creates_second_id(unit_vectors):
    mgr = GlobalIdentityManager(similarity_threshold=0.65)
    a, b = unit_vectors(2)
    _resolve(mgr, [a], tid=1)
    res = _resolve(mgr, [b], cam="cam2", idx=2, tid=2)
    assert res.status == MatchStatus.NEW and res.global_id == "P002"
    assert mgr.global_ids == ["P001", "P002"]


def test_uncertain_band_creates_flagged_new_id(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.80, uncertain_margin=0.20, matching_strategy="max")
    base = unit_vectors(1)[0]
    _resolve(mgr, [base], tid=1)
    # Find a perturbation whose similarity lands inside [0.60, 0.80).
    cand = None
    for scale in np.linspace(0.02, 0.09, 200):
        trial = _perturb(base, rng, scale)
        if 0.62 <= float(trial @ base) < 0.78:
            cand = trial
            break
    assert cand is not None, "could not build an uncertain-band query"
    res = _resolve(mgr, [cand], cam="cam2", idx=2, tid=2)
    assert res.status == MatchStatus.UNCERTAIN
    assert res.global_id == "P002" and res.is_new
    assert res.best_id == "P001" and 0.6 <= res.best_similarity < 0.8
    visit = mgr.get("P002").visits[0]
    assert visit.status == "UNCERTAIN" and visit.candidate_id == "P001"


def test_pending_then_final_decision(unit_vectors):
    mgr = GlobalIdentityManager(similarity_threshold=0.65, max_pending_embeddings=3)
    a, b = unit_vectors(2)
    _resolve(mgr, [a], tid=1)
    # A non-matching track with one embedding is deferred while not final...
    res = _resolve(mgr, [b], tid=2, final=False)
    assert res.status == MatchStatus.PENDING and res.global_id is None
    assert len(mgr) == 1
    # ...and decided once the track ends.
    res = _resolve(mgr, [b], tid=2, final=True)
    assert res.status == MatchStatus.NEW and res.global_id == "P002"


def test_best_candidate_wins_not_first(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.6, matching_strategy="max")
    vecs = unit_vectors(3)
    for i, v in enumerate(vecs):
        _resolve(mgr, [v], tid=i)
    query = _perturb(vecs[2], rng, 0.05)  # clearly P003
    res = _resolve(mgr, [query], cam="cam2", idx=2, tid=9)
    assert res.global_id == "P003" and res.status == MatchStatus.MATCHED


def test_gallery_update_respects_size_and_quality(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.65, max_embeddings_per_identity=5, max_embeddings_per_track=5,
                                duplicate_similarity=0.999)
    base = unit_vectors(1)[0]
    res = _resolve(mgr, [base], tid=1)
    gid = res.global_id
    ident = mgr.get(gid)
    # add_track_embedding only accepts samples consistent with the identity (>= uncertain threshold).
    for i in range(10):
        emb = _perturb(base, rng, 0.01 + 0.002 * i)  # cosine ~0.98 .. ~0.85 to the base vector
        mgr.add_track_embedding(gid, emb, quality=0.5 + 0.04 * i, camera_id="cam1", local_track_id=1)
    assert len(ident.embeddings) == 5  # capped
    assert min(ident.qualities) >= 0.5  # low-quality samples were replaced by better ones
    # A sample that disagrees with the identity is NOT added to the gallery.
    unrelated = unit_vectors(1)[0]
    sim = mgr.add_track_embedding(gid, unrelated, quality=0.99, camera_id="cam1", local_track_id=1)
    assert sim < mgr.uncertain_threshold
    assert len(ident.embeddings) == 5 and max(ident.qualities) < 0.99


def test_per_track_contribution_cap(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.65, max_embeddings_per_identity=20,
                                max_embeddings_per_track=3, duplicate_similarity=0.9999)
    base = unit_vectors(1)[0]
    gid = _resolve(mgr, [base], tid=1).global_id
    for i in range(8):
        mgr.add_track_embedding(gid, _perturb(base, rng, 0.03), quality=0.5, camera_id="cam1", local_track_id=1)
    assert len(mgr.get(gid).embeddings) == 3  # one track cannot fill the gallery


def test_top_k_tracks_resists_single_wrong_track(unit_vectors, rng):
    """A contaminated track must not turn an identity into a hub for a different person."""
    mgr = GlobalIdentityManager(similarity_threshold=0.7, matching_strategy="top_k_tracks", top_k=3,
                                max_embeddings_per_identity=20, duplicate_similarity=0.9999)
    person_a, person_b = unit_vectors(2)
    gid = _resolve(mgr, [_perturb(person_a, rng, 0.02) for _ in range(3)], tid=1).global_id
    _resolve(mgr, [_perturb(person_a, rng, 0.02) for _ in range(3)], cam="cam2", idx=2, tid=2)
    ident = mgr.get(gid)
    # Simulate a wrong merge: inject person B samples under a third track.
    for _ in range(3):
        mgr._add_to_gallery(ident, _perturb(person_b, rng, 0.02), 0.9, "cam3:T9")
    # Query person B: under "max" it would match strongly, under top_k_tracks the other tracks pull it down.
    cands = mgr.match(_perturb(person_b, rng, 0.02))
    assert cands[0][0] == gid and cands[0][1] < mgr.similarity_threshold
    mgr.matching_strategy = "max"
    assert mgr.match(_perturb(person_b, rng, 0.02))[0][1] > mgr.similarity_threshold


def test_covisibility_exclusion(unit_vectors, rng):
    mgr = GlobalIdentityManager(similarity_threshold=0.65)
    base = unit_vectors(1)[0]
    gid = _resolve(mgr, [base], tid=1).global_id
    query = _perturb(base, rng, 0.01)
    res = _resolve(mgr, [query], tid=2, exclude=[gid])  # same person visible elsewhere -> impossible
    assert res.global_id != gid and res.status == MatchStatus.NEW
    assert mgr.stats["covisibility_rejections"] == 1


def test_export_and_rebuild(unit_vectors):
    mgr = GlobalIdentityManager(similarity_threshold=0.65)
    vecs = unit_vectors(3)
    for i, v in enumerate(vecs):
        _resolve(mgr, [v], tid=i)
    arrays = mgr.export_embeddings()
    assert arrays["gallery_embeddings"].shape == (3, 512)
    assert list(arrays["global_ids"]) == ["P001", "P002", "P003"]
    rebuilt = GlobalIdentityManager.from_identities(mgr.identities, similarity_threshold=0.65)
    assert rebuilt.global_ids == mgr.global_ids
    assert rebuilt.match(vecs[1])[0][0] == "P002"
    new = _resolve(rebuilt, [unit_vectors(1)[0]], tid=99)
    assert new.global_id == "P004"  # id counter continues


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        GlobalIdentityManager(matching_strategy="bogus")
