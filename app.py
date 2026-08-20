"""Multi-Camera Person Re-Identification - Streamlit application.

Run with:  streamlit run app.py
Pipeline:  Input Data/*.mp4 -> YOLO (person) -> OC-SORT (per camera) -> OSNet embeddings
           -> GlobalIdentityManager (cross-camera Global IDs) -> annotated videos + reports.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import OSNET_WEIGHT_CHOICES, YOLO_MODEL_CHOICES, AppConfig, load_config
from src.detector import PersonDetector
from src.job import ProcessingJob
from src.reid import OSNetReID
from src.results import ResultsManager
from src.ui_components import (
    render_dashboard,
    render_gallery,
    render_progress,
    render_reports,
    render_search,
    render_timeline,
    render_videos,
)
from src.utils import format_duration, get_logger, resolve_device, setup_logging
from src.video_utils import discover_videos, find_ffmpeg

st.set_page_config(page_title="Multi-Camera Person Re-ID", page_icon=":movie_camera:", layout="wide")
setup_logging()
log = get_logger("app")

THRESHOLD_WARNING = (
    "Re-ID similarity thresholds are model- and environment-dependent. The threshold should be "
    "calibrated using representative footage from the actual cameras."
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models(yolo_model: str, device: str, imgsz: int, half: str, osnet_model: str, batch_size: int,
                models_dir: str):
    """Load YOLO + OSNet once per (model, device) combination - survives Streamlit reruns."""
    detector = PersonDetector(model=yolo_model, device=device, imgsz=imgsz, half=half, models_dir=models_dir)
    reid = OSNetReID(model=osnet_model, device=device, half=half, batch_size=batch_size, models_dir=models_dir)
    return detector, reid


@st.cache_resource(show_spinner=False)
def job_registry() -> dict:
    """Process-wide holder for the running job (survives browser refreshes)."""
    return {"job": None}


@st.cache_data(show_spinner=False, ttl=30)
def scan_inputs(directory: str, extensions: tuple):
    return [(v.path.name, v.camera_id, v.camera_name, v.index) for v in discover_videos(directory, extensions)]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(base: AppConfig) -> tuple[AppConfig, list[str], bool, bool]:
    """Render the sidebar and return (config, selected camera ids, start clicked, stop clicked)."""
    cfg = base.copy()
    with st.sidebar:
        st.title("Multi-Camera Person Re-ID")
        st.caption("YOLO detection - OC-SORT tracking - OSNet re-identification")

        st.subheader("Input Folder")
        cfg.input.directory = st.text_input("Directory", value=base.input.directory, key="input_dir")
        if st.button("Rescan folder", key="rescan"):
            scan_inputs.clear()
        videos = scan_inputs(cfg.input.directory, tuple(cfg.input.extensions))
        st.markdown("**Available Cameras**")
        selected: list[str] = []
        if not videos:
            st.warning(f"No videos found in `{cfg.input.directory}` "
                       f"(supported: {', '.join(cfg.input.extensions)}).")
        for name, cam_id, cam_name, idx in videos:
            if st.checkbox(f"{cam_name} - {name}", value=True, key=f"cam_{cam_id}"):
                selected.append(cam_id)

        st.subheader("Detection")
        yolo_choices = list(YOLO_MODEL_CHOICES)
        if base.detector.model not in yolo_choices:
            yolo_choices.insert(0, base.detector.model)
        yolo_choices.append("custom...")
        yolo_sel = st.selectbox("Model", yolo_choices, index=yolo_choices.index(base.detector.model), key="yolo_model")
        if yolo_sel == "custom...":
            yolo_sel = st.text_input("Custom YOLO weights (.pt path or name)", value=base.detector.model, key="yolo_custom")
        cfg.detector.model = yolo_sel
        cfg.detector.confidence = st.slider("Confidence", 0.05, 0.95, float(base.detector.confidence), 0.05, key="det_conf")
        cfg.detector.iou = st.slider("IoU (NMS)", 0.1, 0.9, float(base.detector.iou), 0.05, key="det_iou")
        sizes = [640, 800, 960, 1280]
        if base.detector.imgsz not in sizes:
            sizes.append(int(base.detector.imgsz))
            sizes.sort()
        cfg.detector.imgsz = st.select_slider("Image size", sizes, value=int(base.detector.imgsz), key="det_imgsz",
                                              help="Larger = better recall on small/distant people, slower.")
        device_opts = ["auto", "cpu", "cuda:0"]
        dev_default = base.detector.device if base.detector.device in device_opts else "auto"
        cfg.detector.device = st.selectbox("Device", device_opts, index=device_opts.index(dev_default), key="device")
        cfg.reid.device = cfg.detector.device

        st.subheader("Tracking (OC-SORT)")
        cfg.tracker.max_age = st.number_input("Max age (frames)", 1, 300, int(base.tracker.max_age), key="trk_max_age")
        cfg.tracker.min_hits = st.number_input("Min hits", 1, 20, int(base.tracker.min_hits), key="trk_min_hits")
        cfg.tracker.iou_threshold = st.slider("Association IoU", 0.05, 0.9, float(base.tracker.iou_threshold), 0.05,
                                              key="trk_iou")
        cfg.tracking.min_track_frames = st.number_input("Min track frames for a Global ID", 1, 300,
                                                        int(base.tracking.min_track_frames), key="min_track_frames")

        st.subheader("Re-ID (OSNet)")
        osnet_choices = list(OSNET_WEIGHT_CHOICES)
        if base.reid.model not in osnet_choices:
            osnet_choices.insert(0, base.reid.model)
        cfg.reid.model = st.selectbox("OSNet model", osnet_choices, index=osnet_choices.index(base.reid.model),
                                      key="osnet_model")
        cfg.reid.similarity_threshold = st.slider("Similarity threshold", 0.30, 0.95,
                                                  float(base.reid.similarity_threshold), 0.01, key="sim_thr")
        st.caption(THRESHOLD_WARNING)
        cfg.reid.uncertain_margin = st.slider("Uncertain margin", 0.0, 0.3, float(base.reid.uncertain_margin), 0.01,
                                              key="unc_margin",
                                              help="Similarities in [threshold - margin, threshold) create a new ID "
                                                   "flagged UNCERTAIN instead of merging.")
        cfg.reid.embedding_interval = st.number_input("Embedding interval (frames)", 1, 300,
                                                      int(base.reid.embedding_interval), key="emb_interval",
                                                      help="OSNet runs on a track at its first appearance and then "
                                                           "every N processed frames.")
        cfg.reid.max_embeddings_per_identity = st.number_input("Gallery size per ID", 1, 200,
                                                               int(base.reid.max_embeddings_per_identity), key="gal_size")
        strategies = ["top_k_tracks", "top_k", "max", "mean"]
        cfg.reid.matching_strategy = st.selectbox("Matching strategy", strategies,
                                                  index=strategies.index(base.reid.matching_strategy)
                                                  if base.reid.matching_strategy in strategies else 0,
                                                  key="match_strategy",
                                                  help="top_k_tracks: mean of the best k tracks (each the mean of its "
                                                       "k best samples) - robust to one wrong merge. top_k/max/mean "
                                                       "operate on all gallery samples.")
        cfg.reid.batch_size = st.number_input("OSNet batch size", 1, 128, int(base.reid.batch_size), key="reid_batch")

        st.subheader("Processing")
        cfg.output.save_crops = st.checkbox("Save crops", value=base.output.save_crops, key="save_crops",
                                            help="Representative crops only (Output/crops/P001/camera/...).")
        cfg.output.save_embeddings = st.checkbox("Save embeddings", value=base.output.save_embeddings, key="save_emb")
        cfg.output.save_videos = st.checkbox("Generate output videos", value=base.output.save_videos, key="save_videos")
        cfg.performance.frame_skip = st.number_input("Frame skip", 0, 30, int(base.performance.frame_skip),
                                                     key="frame_skip",
                                                     help="Skip N frames between processed frames (0 = every frame). "
                                                          "Skipped frames are not analysed or written.")
        cfg.performance.detection_interval = st.number_input("Detection interval", 1, 30,
                                                             int(base.performance.detection_interval), key="det_int",
                                                             help="Run YOLO + tracker every N processed frames; "
                                                                  "in-between frames reuse the last boxes.")
        cfg.performance.max_frames = st.number_input("Max frames per camera (0 = all)", 0, 10_000_000,
                                                     int(base.performance.max_frames), step=100, key="max_frames")
        cfg.performance.start_frame = st.number_input("Start frame", 0, 10_000_000, int(base.performance.start_frame),
                                                      step=100, key="start_frame")
        modes = ["sequential", "parallel"]
        cfg.performance.mode = st.selectbox("Mode", modes, index=modes.index(base.performance.mode)
                                            if base.performance.mode in modes else 0, key="mode",
                                            help="Sequential = one camera after another (memory friendly). Parallel = "
                                                 "several cameras at once sharing the same models (no extra GPU "
                                                 "models).")
        if cfg.performance.mode == "parallel":
            cfg.performance.parallel_workers = st.number_input("Parallel workers", 1, 8,
                                                               int(base.performance.parallel_workers), key="workers")
        with st.expander("What do these values mean?"):
            st.markdown(
                "- **Frame skip**: lowers the effective frame rate (e.g. 1 = every 2nd frame). Biggest speed-up.\n"
                "- **Detection interval**: YOLO/OC-SORT only every N processed frames; boxes are frozen in between.\n"
                "- **Embedding interval**: how often OSNet re-embeds an existing track; the first appearance is "
                "always embedded. Larger = faster, fewer gallery samples.\n"
                "- **Gallery size**: embeddings kept per Global ID (different views / lighting).\n"
                "- **Batch size**: crops embedded per OSNet forward pass.\n"
                "- **Max frames / start frame**: process only a slice of each video (quick experiments)."
            )

        st.divider()
        start = st.button("START PROCESSING", type="primary", width="stretch", key="start_btn",
                          disabled=not selected)
        stop = st.button("STOP", width="stretch", key="stop_btn")
    return cfg, selected, start, stop


# --------------------------------------------------------------------------- #
# Status line
# --------------------------------------------------------------------------- #
def render_status(cfg: AppConfig, models_loaded: bool, detector=None, reid=None) -> None:
    dev = resolve_device(cfg.detector.device)
    cols = st.columns(5)
    cols[0].markdown(f"**Device:** {'CUDA' if dev.is_cuda else 'CPU'}")
    cols[1].markdown(f"**GPU:** {dev.gpu_name or '-'}")
    cols[2].markdown(f"**VRAM:** {dev.vram_total_mb / 1024:.1f} GB" if dev.is_cuda else "**VRAM:** -")
    cols[3].markdown(f"**torch:** {dev.torch_version}")
    ffmpeg = find_ffmpeg()
    cols[4].markdown(f"**Video encoder:** {'ffmpeg (H.264)' if ffmpeg else 'OpenCV mp4v'}")
    checks = [
        ("YOLO loaded" + (f" ({Path(detector.model_path).name})" if detector else ""), models_loaded),
        ("OC-SORT initialized (per camera)", True),
        ("OSNet loaded" + (f" ({reid.weights_name}, {reid.embedding_dim}-d)" if reid else ""), models_loaded),
        ("CUDA available", dev.cuda_available),
    ]
    st.markdown("  ".join(("✓" if ok else "✗") + " " + name for name, ok in checks))


@st.fragment(run_every=1.0)
def progress_fragment() -> None:
    """Auto-refreshing progress view; only this fragment reruns every second."""
    current = job_registry().get("job")
    if current is None:
        return
    render_progress(current)
    if not current.running:
        st.rerun(scope="app")


def identity_manager_for(result, job: ProcessingJob | None):
    """Identity manager matching `result` (live job or rebuilt from a loaded run) for image search."""
    if job is not None and job.result is result:
        return job.identity_manager
    from src.global_identity import GlobalIdentityManager

    reid_cfg = result.config_snapshot.get("reid", {})
    return GlobalIdentityManager.from_identities(
        result.identities,
        similarity_threshold=reid_cfg.get("similarity_threshold", 0.7),
        uncertain_margin=reid_cfg.get("uncertain_margin", 0.08),
        max_embeddings_per_identity=reid_cfg.get("max_embeddings_per_identity", 20),
        matching_strategy=reid_cfg.get("matching_strategy", "top_k_tracks"),
        top_k=reid_cfg.get("top_k", 3),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if "base_config" not in st.session_state:
        st.session_state["base_config"] = load_config("config.yaml")
    base_cfg: AppConfig = st.session_state["base_config"]
    cfg, selected, start_clicked, stop_clicked = sidebar(base_cfg)
    registry = job_registry()
    job: ProcessingJob | None = registry.get("job")

    st.title("Multi-Camera Person Tracking & Re-Identification")
    st.caption("Local OC-SORT track IDs (C1-T03) are per camera; Global Person IDs (P001) come from OSNet "
               "appearance matching across cameras.")

    # ---- models ------------------------------------------------------- #
    detector = reid = None
    models_loaded = False
    load_error = None
    with st.spinner("Loading models (YOLO + OSNet)... first start downloads the weights"):
        try:
            detector, reid = load_models(cfg.detector.model, cfg.detector.device, int(cfg.detector.imgsz),
                                         str(cfg.detector.half), cfg.reid.model, int(cfg.reid.batch_size),
                                         cfg.detector.models_dir)
            models_loaded = True
        except Exception as exc:  # model download / CUDA problems
            load_error = exc
            log.exception("Model loading failed: %s", exc)
    render_status(cfg, models_loaded, detector, reid)
    if load_error is not None:
        st.error(f"Model loading failed: {load_error}. Check the model name, internet access (first download) "
                 f"and the device setting.")

    # ---- start / stop ------------------------------------------------- #
    if stop_clicked and job is not None and job.running:
        job.cancel()
        st.warning("Stop requested - finishing the current frame and writing partial results...")

    if start_clicked:
        if job is not None and job.running:
            st.warning("A processing job is already running.")
        elif not models_loaded:
            st.error("Models are not loaded; cannot start.")
        else:
            cameras = [v for v in discover_videos(cfg.input.directory, cfg.input.extensions) if v.camera_id in selected]
            if not cameras:
                st.error("No cameras selected.")
            else:
                # Runtime thresholds may have changed since the models were cached.
                detector.confidence, detector.iou = float(cfg.detector.confidence), float(cfg.detector.iou)
                reid.batch_size = int(cfg.reid.batch_size)
                st.session_state.pop("result", None)
                st.session_state.pop("selected_gid", None)
                job = ProcessingJob(config=cfg, cameras=cameras, detector=detector, reid=reid).start()
                registry["job"] = job
                log.info("Started processing %d cameras", len(cameras))
                st.rerun()

    # ---- running job ---------------------------------------------------- #
    if job is not None and job.running:
        st.subheader("Processing")
        st.info("Processing runs in the background - the interface stays responsive. Use STOP to cancel.")
        progress_fragment()
        return

    # ---- finished job -> adopt result -------------------------------- #
    if job is not None and job.finished and "result" not in st.session_state:
        if job.error:
            st.error(f"Processing failed: {job.error}")
        if job.result is not None:
            st.session_state["result"] = job.result
            st.success(f"Processing completed in {format_duration(job.elapsed)}"
                       + (" (cancelled early)" if job.cancelled else ""))

    result = st.session_state.get("result")
    if result is None:
        rm = ResultsManager(cfg.output.directory)
        if (Path(cfg.output.directory) / "reports" / "last_run.pkl").is_file():
            if st.button("Load results of the previous run", key="load_prev"):
                prev = rm.load_state()
                if prev is not None:
                    st.session_state["result"] = prev
                    st.rerun()
                else:
                    st.error("Could not load previous results.")
        st.markdown(
            "1. Put camera videos in the **Input Data** folder (`.mp4`, `.avi`, `.mkv`, `.mov`).\n"
            "2. Select cameras and tune detection / tracking / Re-ID settings in the sidebar.\n"
            "3. Click **START PROCESSING**. Progress, a live preview and logs appear here.\n"
            "4. Afterwards explore the dashboard, person gallery, videos, timeline and Re-ID search."
        )
        return

    # ---- dashboard ------------------------------------------------------ #
    tabs = st.tabs(["Dashboard", "Person Gallery", "Camera Videos", "Track Timeline", "Re-ID Search", "Reports & Logs"])
    with tabs[0]:
        render_dashboard(result)
    with tabs[1]:
        render_gallery(result)
    with tabs[2]:
        render_videos(result)
    with tabs[3]:
        render_timeline(result)
    with tabs[4]:
        render_search(result, reid if models_loaded else None, identity_manager_for(result, job))
    with tabs[5]:
        render_reports(result)


if __name__ == "__main__":
    main()
