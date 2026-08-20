"""Multi-camera person tracking and re-identification package.

Pipeline: video -> YOLO (person detection) -> OC-SORT (per-camera tracking)
-> OSNet (appearance embeddings) -> GlobalIdentityManager (cross-camera Re-ID).
"""

__version__ = "1.0.0"
