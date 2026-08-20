"""Global (cross-camera) identity management based on OSNet embedding galleries.

Local OC-SORT track IDs never leave their camera. Every track that lives long
enough gets a *Global Person ID* (P001, P002, ...) by comparing its appearance
embeddings against the galleries of the identities seen so far (cosine
similarity). The same Global ID can therefore be attached to tracks from many
cameras, at different times.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import get_logger, l2_normalize

log = get_logger("identity")


class MatchStatus(str, Enum):
    MATCHED = "MATCHED"  # similarity >= threshold -> existing Global ID
    NEW = "NEW ID"  # clearly unseen appearance -> new Global ID
    UNCERTAIN = "UNCERTAIN"  # borderline similarity -> new Global ID, flagged with closest candidate
    PENDING = "PENDING"  # decision deferred until more embeddings are available
    SHORT = "SHORT"  # track too short, never assigned
    NO_EMBEDDING = "NO_EMBEDDING"  # no usable crop, never assigned


@dataclass
class CameraVisit:
    """One local track of a global identity in one camera."""

    camera_id: str
    camera_name: str
    camera_index: int
    local_track_id: int
    first_seen: float  # seconds (camera offset + video time)
    last_seen: float
    first_frame: int
    last_frame: int
    similarity: float  # similarity at assignment time
    status: str  # MATCHED / NEW ID / UNCERTAIN
    avg_similarity: float = 0.0  # running mean over all embeddings of the track
    candidate_id: Optional[str] = None
    candidate_similarity: float = 0.0
    max_det_conf: float = 0.0
    n_embeddings: int = 0
    open: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def local_label(self) -> str:
        return f"C{self.camera_index}-T{self.local_track_id:02d}"


@dataclass
class GlobalIdentity:
    global_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # "cam1:T3" per embedding
    cameras_seen: Dict[str, str] = field(default_factory=dict)  # camera_id -> camera_name (insertion ordered)
    first_seen: float = float("inf")
    last_seen: float = 0.0
    representative_image: Optional[bytes] = None  # JPEG bytes
    representative_quality: float = -1.0
    camera_images: Dict[str, bytes] = field(default_factory=dict)  # best crop per camera
    camera_image_quality: Dict[str, float] = field(default_factory=dict)
    visits: List[CameraVisit] = field(default_factory=list)
    created_order: int = 0

    # ------------------------------------------------------------------ #
    @property
    def gallery(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self.embeddings, axis=0)

    @property
    def mean_embedding(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0,), dtype=np.float32)
        return l2_normalize(np.mean(self.gallery, axis=0))

    @property
    def num_tracks(self) -> int:
        return len(self.visits)

    @property
    def total_duration(self) -> float:
        return float(sum(v.duration for v in self.visits))

    @property
    def camera_ids(self) -> List[str]:
        return list(self.cameras_seen.keys())

    @property
    def camera_names(self) -> List[str]:
        return list(self.cameras_seen.values())

    def visits_in(self, camera_id: str) -> List[CameraVisit]:
        return [v for v in self.visits if v.camera_id == camera_id]


@dataclass
class MatchResult:
    status: MatchStatus
    global_id: Optional[str] = None
    similarity: float = 0.0  # similarity to the assigned identity (0 for NEW)
    best_id: Optional[str] = None  # closest existing identity (may equal global_id)
    best_similarity: float = 0.0
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # top candidates (id, score)
    is_new: bool = False

    @property
    def assigned(self) -> bool:
        return self.global_id is not None


def _topk_mean(values: np.ndarray, k: int) -> float:
    """Mean of the k largest values (k is capped by the number of values)."""
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return -1.0
    k = min(max(1, k), values.size)
    if k == values.size:
        return float(values.mean())
    return float(np.partition(values, -k)[-k:].mean())


def _group_indices(sources: Sequence[str]) -> List[np.ndarray]:
    """Indices of gallery samples grouped by the track that produced them."""
    groups: Dict[str, List[int]] = {}
    for i, src in enumerate(sources):
        groups.setdefault(src, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


class GlobalIdentityManager:
    """Maintains the database of global identities and matches new tracks against it.

    Matching strategies (query vs. an identity's gallery):
      * max          - highest cosine similarity over all gallery samples
      * mean         - mean similarity over all gallery samples
      * top_k        - mean of the k highest similarities over all gallery samples
      * top_k_tracks - (default) two-level top-k: first per contributing track (mean of its
                       k best samples), then the mean of the k best tracks. A single wrongly
                       merged track can therefore not "hijack" an identity - the other tracks
                       keep pulling the score down.

    Thread-safe (a re-entrant lock guards all mutations and queries) so it can be
    shared by camera workers running in parallel.
    """

    STRATEGIES = ("max", "mean", "top_k", "top_k_tracks")

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        uncertain_margin: float = 0.08,
        max_embeddings_per_identity: int = 20,
        matching_strategy: str = "top_k",
        top_k: int = 3,
        max_pending_embeddings: int = 3,
        max_embeddings_per_track: int = 5,
        id_prefix: str = "P",
        duplicate_similarity: float = 0.985,
    ):
        if matching_strategy not in self.STRATEGIES:
            raise ValueError(f"matching_strategy must be one of {self.STRATEGIES}")
        self.similarity_threshold = float(similarity_threshold)
        self.uncertain_margin = float(max(0.0, uncertain_margin))
        self.max_embeddings = int(max(1, max_embeddings_per_identity))
        self.matching_strategy = matching_strategy
        self.top_k = int(max(1, top_k))
        self.max_pending_embeddings = int(max(1, max_pending_embeddings))
        # A single (possibly wrongly matched) track may contribute at most this many gallery
        # samples, so one track can never take over an identity's appearance model.
        self.max_embeddings_per_track = int(max(1, max_embeddings_per_track))
        self.id_prefix = id_prefix
        self.duplicate_similarity = float(duplicate_similarity)

        self._identities: Dict[str, GlobalIdentity] = {}
        self._order: List[str] = []
        self._next_id = 1
        self._lock = threading.RLock()
        # Cached gallery matrix for fast matching.
        self._matrix: Optional[np.ndarray] = None
        self._offsets: Optional[np.ndarray] = None
        self._source_groups: List[List[np.ndarray]] = []
        self._dirty = True
        self.stats = {
            "matched": 0, "new": 0, "uncertain": 0, "pending": 0, "gallery_updates": 0,
            # Times the best candidate was above threshold but visible elsewhere in the same
            # camera at the same time (a physical impossibility). Many of these = threshold too low.
            "covisibility_rejections": 0,
        }

    # ------------------------------------------------------------------ #
    # Read access
    # ------------------------------------------------------------------ #
    @property
    def uncertain_threshold(self) -> float:
        return max(0.0, self.similarity_threshold - self.uncertain_margin)

    @property
    def identities(self) -> List[GlobalIdentity]:
        with self._lock:
            return [self._identities[g] for g in self._order]

    @property
    def global_ids(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def get(self, global_id: str) -> Optional[GlobalIdentity]:
        with self._lock:
            return self._identities.get(global_id)

    def __contains__(self, global_id: str) -> bool:
        return global_id in self._identities

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    def _ensure_matrix(self) -> None:
        if not self._dirty and self._matrix is not None:
            return
        blocks, offsets, groups, total = [], [0], [], 0
        for gid in self._order:
            ident = self._identities[gid]
            emb = ident.embeddings
            total += len(emb)
            offsets.append(total)
            groups.append(_group_indices(ident.sources) if emb else [])
            if emb:
                blocks.append(np.stack(emb, axis=0))
        self._matrix = np.concatenate(blocks, axis=0).astype(np.float32) if blocks else None
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._source_groups = groups
        self._dirty = False

    def _score(self, sims: np.ndarray, groups: Optional[List[np.ndarray]] = None) -> float:
        """Identity score from the similarities of its gallery samples (see class docstring)."""
        if sims.size == 0:
            return -1.0
        if self.matching_strategy == "max":
            return float(sims.max())
        if self.matching_strategy == "mean":
            return float(sims.mean())
        if self.matching_strategy == "top_k" or not groups:
            return _topk_mean(sims, self.top_k)
        per_track = np.asarray([_topk_mean(sims[g], self.top_k) for g in groups], dtype=np.float32)
        return _topk_mean(per_track, self.top_k)

    def _aggregate(self, sims: np.ndarray) -> np.ndarray:
        """Per-identity score from per-embedding similarities."""
        n = len(self._order)
        scores = np.full(n, -1.0, dtype=np.float32)
        offsets = self._offsets
        for i in range(n):
            s = sims[offsets[i]:offsets[i + 1]]
            if s.size == 0:
                continue
            scores[i] = self._score(s, self._source_groups[i] if self._source_groups else None)
        return scores

    def match(self, embedding: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Score `embedding` against every identity; returns [(global_id, score)] best first."""
        q = l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        with self._lock:
            if not self._order:
                return []
            self._ensure_matrix()
            if self._matrix is None or q.size != self._matrix.shape[1]:
                return []
            sims = self._matrix @ q
            scores = self._aggregate(sims)
            order = np.argsort(-scores)[:top_n]
            return [(self._order[i], float(scores[i])) for i in order if scores[i] > -1.0]

    def search(self, embedding: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Alias used by the Re-ID search UI."""
        return self.match(embedding, top_n=top_n)

    def classify(self, best_similarity: float) -> MatchStatus:
        if best_similarity >= self.similarity_threshold:
            return MatchStatus.MATCHED
        if best_similarity >= self.uncertain_threshold:
            return MatchStatus.UNCERTAIN
        return MatchStatus.NEW

    def decide(self, embedding: np.ndarray, exclude: Optional[Sequence[str]] = None) -> MatchResult:
        """Pure decision (no side effects): which identity would this embedding get?

        `exclude` lists identities that cannot be the answer (e.g. people visible at the
        same moment elsewhere in the same camera); they are removed from the candidates.
        """
        candidates = self.match(embedding, top_n=10)
        if exclude:
            excluded = set(exclude)
            kept = [(g, s) for g, s in candidates if g not in excluded]
            if candidates and kept != candidates:
                top_id, top_sim = candidates[0]
                if top_id in excluded and top_sim >= self.similarity_threshold:
                    self.stats["covisibility_rejections"] += 1
                    log.debug("Candidate %s (%.3f) rejected: visible simultaneously in the same camera", top_id, top_sim)
            candidates = kept
        if not candidates:
            return MatchResult(MatchStatus.NEW, best_similarity=0.0, candidates=[])
        best_id, best_sim = candidates[0]
        status = self.classify(best_sim)
        return MatchResult(status, global_id=best_id if status == MatchStatus.MATCHED else None,
                           similarity=best_sim if status == MatchStatus.MATCHED else 0.0,
                           best_id=best_id, best_similarity=best_sim, candidates=candidates[:5])

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #
    def resolve_track(
        self,
        embeddings: Sequence[np.ndarray],
        qualities: Sequence[float],
        *,
        camera_id: str,
        camera_name: str,
        camera_index: int,
        local_track_id: int,
        first_seen: float,
        first_frame: int,
        last_seen: Optional[float] = None,
        last_frame: Optional[int] = None,
        max_det_conf: float = 0.0,
        crop_jpeg: Optional[bytes] = None,
        crop_quality: float = 0.0,
        final: bool = False,
        exclude: Optional[Sequence[str]] = None,
    ) -> MatchResult:
        """Assign a Global ID to a track (or defer the decision).

        The query is the L2-normalised mean of the track's embeddings so far.
        - similarity >= threshold                       -> MATCHED (existing ID)
        - otherwise, if not final and still few samples -> PENDING (ask again later)
        - otherwise                                     -> NEW ID / UNCERTAIN (new identity)
        `exclude`: Global IDs that are impossible for this track (co-visible in the same camera).
        """
        if len(embeddings) == 0:
            return MatchResult(MatchStatus.PENDING)
        query = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        with self._lock:
            verdict = self.decide(query, exclude=exclude)
            status = verdict.status
            if status != MatchStatus.MATCHED and not final and len(embeddings) < self.max_pending_embeddings:
                self.stats["pending"] += 1
                return MatchResult(MatchStatus.PENDING, best_id=verdict.best_id,
                                   best_similarity=verdict.best_similarity, candidates=verdict.candidates)

            label = f"C{camera_index}-T{local_track_id:02d}"
            if status == MatchStatus.MATCHED:
                gid = verdict.best_id
                identity = self._identities[gid]
                self.stats["matched"] += 1
                log.info("Track %s matched to %s similarity=%.3f", label, gid, verdict.best_similarity)
                result = MatchResult(MatchStatus.MATCHED, global_id=gid, similarity=verdict.best_similarity,
                                     best_id=gid, best_similarity=verdict.best_similarity,
                                     candidates=verdict.candidates)
            else:
                identity = self._create_identity()
                gid = identity.global_id
                if status == MatchStatus.UNCERTAIN:
                    self.stats["uncertain"] += 1
                    log.info("Track %s -> new Global ID %s (UNCERTAIN, closest %s at %.3f)",
                             label, gid, verdict.best_id, verdict.best_similarity)
                else:
                    self.stats["new"] += 1
                    log.info("Track %s -> new Global ID %s (best existing similarity %.3f)",
                             label, gid, verdict.best_similarity)
                result = MatchResult(status, global_id=gid, similarity=0.0, best_id=verdict.best_id,
                                     best_similarity=verdict.best_similarity, candidates=verdict.candidates,
                                     is_new=True)

            # Add the track's embeddings to the gallery (quality gated) and register the visit.
            source = f"{camera_id}:T{local_track_id}"
            for emb, q in zip(embeddings, qualities):
                self._add_to_gallery(identity, emb, float(q), source)
            self._register_visit(
                identity, camera_id, camera_name, camera_index, local_track_id, first_seen, first_frame,
                last_seen if last_seen is not None else first_seen,
                last_frame if last_frame is not None else first_frame,
                result, max_det_conf, len(embeddings),
            )
            if crop_jpeg is not None:
                self._update_images(identity, camera_id, crop_jpeg, crop_quality)
            return result

    def add_track_embedding(
        self,
        global_id: str,
        embedding: np.ndarray,
        quality: float,
        *,
        camera_id: str,
        local_track_id: int,
        crop_jpeg: Optional[bytes] = None,
        crop_quality: float = 0.0,
    ) -> float:
        """Add a later embedding of an already assigned track. Returns its similarity to the gallery.

        Samples that disagree with the identity's current appearance model (similarity below
        the uncertain threshold) are not added - this keeps a single bad match from dragging
        the whole gallery towards a different person.
        """
        with self._lock:
            identity = self._identities.get(global_id)
            if identity is None:
                return 0.0
            sim = self._similarity_to_identity(identity, embedding)
            if sim >= self.uncertain_threshold:
                self._add_to_gallery(identity, embedding, float(quality), f"{camera_id}:T{local_track_id}")
                if crop_jpeg is not None:
                    self._update_images(identity, camera_id, crop_jpeg, crop_quality)
            return sim

    def update_visit(
        self,
        global_id: str,
        camera_id: str,
        local_track_id: int,
        *,
        last_seen: float,
        last_frame: int,
        avg_similarity: Optional[float] = None,
        max_det_conf: Optional[float] = None,
        n_embeddings: Optional[int] = None,
        close: bool = False,
    ) -> None:
        with self._lock:
            identity = self._identities.get(global_id)
            if identity is None:
                return
            for visit in identity.visits:
                if visit.camera_id == camera_id and visit.local_track_id == local_track_id:
                    visit.last_seen = max(visit.last_seen, float(last_seen))
                    visit.last_frame = max(visit.last_frame, int(last_frame))
                    if avg_similarity is not None:
                        visit.avg_similarity = float(avg_similarity)
                    if max_det_conf is not None:
                        visit.max_det_conf = max(visit.max_det_conf, float(max_det_conf))
                    if n_embeddings is not None:
                        visit.n_embeddings = int(n_embeddings)
                    if close:
                        visit.open = False
                    identity.last_seen = max(identity.last_seen, visit.last_seen)
                    break

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _create_identity(self) -> GlobalIdentity:
        gid = f"{self.id_prefix}{self._next_id:03d}"
        self._next_id += 1
        identity = GlobalIdentity(global_id=gid, created_order=len(self._order))
        self._identities[gid] = identity
        self._order.append(gid)
        self._dirty = True
        log.info("New Global ID %s", gid)
        return identity

    def _similarity_to_identity(self, identity: GlobalIdentity, embedding: np.ndarray) -> float:
        if not identity.embeddings:
            return 0.0
        q = l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        sims = identity.gallery @ q
        return self._score(sims, _group_indices(identity.sources))

    def _add_to_gallery(self, identity: GlobalIdentity, embedding: np.ndarray, quality: float, source: str) -> bool:
        emb = l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        if emb.size == 0 or not np.isfinite(emb).all():
            return False
        if identity.embeddings:
            sims = identity.gallery @ emb
            # Near-duplicate of an equally good sample -> nothing new to learn.
            j = int(np.argmax(sims))
            if sims[j] >= self.duplicate_similarity and identity.qualities[j] >= quality:
                return False
        same_source = [i for i, s in enumerate(identity.sources) if s == source]
        if len(same_source) >= self.max_embeddings_per_track:
            # This track already filled its quota: only swap in a better sample of its own.
            worst = min(same_source, key=lambda i: identity.qualities[i])
            if identity.qualities[worst] >= quality:
                return False
            identity.embeddings[worst] = emb
            identity.qualities[worst] = quality
        elif len(identity.embeddings) < self.max_embeddings:
            identity.embeddings.append(emb)
            identity.qualities.append(quality)
            identity.sources.append(source)
        else:
            worst = int(np.argmin(identity.qualities))
            if identity.qualities[worst] >= quality:
                return False  # never replace good samples with poorer ones
            identity.embeddings[worst] = emb
            identity.qualities[worst] = quality
            identity.sources[worst] = source
        self._dirty = True
        self.stats["gallery_updates"] += 1
        return True

    def _register_visit(self, identity, camera_id, camera_name, camera_index, local_track_id, first_seen, first_frame,
                        last_seen, last_frame, result: MatchResult, max_det_conf: float, n_embeddings: int) -> None:
        visit = CameraVisit(
            camera_id=camera_id, camera_name=camera_name, camera_index=camera_index, local_track_id=local_track_id,
            first_seen=float(first_seen), last_seen=float(last_seen), first_frame=int(first_frame),
            last_frame=int(last_frame), similarity=float(result.similarity), status=result.status.value,
            avg_similarity=float(result.similarity), candidate_id=result.best_id if result.is_new else None,
            candidate_similarity=float(result.best_similarity) if result.is_new else 0.0,
            max_det_conf=float(max_det_conf), n_embeddings=int(n_embeddings),
        )
        identity.visits.append(visit)
        identity.cameras_seen.setdefault(camera_id, camera_name)
        identity.first_seen = min(identity.first_seen, visit.first_seen)
        identity.last_seen = max(identity.last_seen, visit.last_seen)

    def _update_images(self, identity: GlobalIdentity, camera_id: str, jpeg: bytes, quality: float) -> None:
        if quality > identity.representative_quality:
            identity.representative_image = jpeg
            identity.representative_quality = quality
        if quality > identity.camera_image_quality.get(camera_id, -1.0):
            identity.camera_images[camera_id] = jpeg
            identity.camera_image_quality[camera_id] = quality

    # ------------------------------------------------------------------ #
    # Export / maintenance
    # ------------------------------------------------------------------ #
    @classmethod
    def from_identities(cls, identities: Sequence[GlobalIdentity], **kwargs) -> "GlobalIdentityManager":
        """Rebuild a manager from saved identities (e.g. a reloaded previous run)."""
        manager = cls(**kwargs)
        with manager._lock:
            for ident in identities:
                manager._identities[ident.global_id] = ident
                manager._order.append(ident.global_id)
                digits = "".join(ch for ch in ident.global_id if ch.isdigit())
                if digits:
                    manager._next_id = max(manager._next_id, int(digits) + 1)
            manager._dirty = True
        return manager

    def export_embeddings(self) -> Dict[str, np.ndarray]:
        """Arrays suitable for np.savez: gallery embeddings with their owner ids."""
        with self._lock:
            rows, owners, sources, qualities = [], [], [], []
            for gid in self._order:
                ident = self._identities[gid]
                for emb, src, q in zip(ident.embeddings, ident.sources, ident.qualities):
                    rows.append(emb)
                    owners.append(gid)
                    sources.append(src)
                    qualities.append(q)
            dim = rows[0].shape[0] if rows else 0
            return {
                "global_ids": np.asarray(self._order, dtype=str),
                "gallery_embeddings": np.stack(rows, axis=0) if rows else np.zeros((0, dim), dtype=np.float32),
                "gallery_owner": np.asarray(owners, dtype=str),
                "gallery_source": np.asarray(sources, dtype=str),
                "gallery_quality": np.asarray(qualities, dtype=np.float32),
                "mean_embeddings": np.stack([self._identities[g].mean_embedding for g in self._order], axis=0)
                if rows else np.zeros((0, dim), dtype=np.float32),
            }

    def merge(self, source_id: str, target_id: str) -> bool:
        """Manually merge identity `source_id` into `target_id` (not done automatically)."""
        with self._lock:
            if source_id == target_id or source_id not in self._identities or target_id not in self._identities:
                return False
            src, dst = self._identities[source_id], self._identities[target_id]
            for emb, q, s in zip(src.embeddings, src.qualities, src.sources):
                self._add_to_gallery(dst, emb, q, s)
            dst.visits.extend(src.visits)
            for cid, cname in src.cameras_seen.items():
                dst.cameras_seen.setdefault(cid, cname)
            for cid, img in src.camera_images.items():
                self._update_images(dst, cid, img, src.camera_image_quality.get(cid, 0.0))
            dst.first_seen = min(dst.first_seen, src.first_seen)
            dst.last_seen = max(dst.last_seen, src.last_seen)
            del self._identities[source_id]
            self._order.remove(source_id)
            self._dirty = True
            log.info("Merged %s into %s", source_id, target_id)
            return True

    def reset(self) -> None:
        with self._lock:
            self._identities.clear()
            self._order.clear()
            self._next_id = 1
            self._matrix = None
            self._offsets = None
            self._source_groups = []
            self._dirty = True
            for k in self.stats:
                self.stats[k] = 0
