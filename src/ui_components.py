"""Streamlit rendering components for the multi-camera Re-ID dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .global_identity import GlobalIdentity
from .job import ProcessingJob
from .results import ProcessingResult
from .utils import format_duration, format_seconds, get_memory_log_handler

STATUS_COLORS = {
    "MATCHED": "#2e7d32",
    "NEW ID": "#1565c0",
    "UNCERTAIN": "#ef6c00",
    "SHORT": "#757575",
    "NO_EMBEDDING": "#9e9e9e",
    "PENDING": "#9e9e9e",
}


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.8em;margin-right:4px'>{text}</span>")


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def render_progress(job: ProcessingJob, show_previews: bool = True) -> None:
    """Per-camera progress bars, live stats and preview frames."""
    snapshot = job.snapshot()
    total_done = sum(p.frames_processed for p in snapshot.values())
    total_all = sum(p.frames_total for p in snapshot.values())
    overall = total_done / total_all if total_all else 0.0
    st.progress(min(1.0, overall), text=f"Overall {overall * 100:.0f}% - {total_done}/{total_all} frames - "
                                           f"elapsed {format_duration(job.elapsed)}")
    cols = st.columns(4)
    cols[0].metric("Cameras done", f"{sum(1 for p in snapshot.values() if p.status == 'done')}/{len(snapshot)}")
    cols[1].metric("Global IDs so far", f"{len(job.identity_manager)}")
    cols[2].metric("Tracks so far", f"{sum(p.total_tracks for p in snapshot.values())}")
    running = [p for p in snapshot.values() if p.status == "running"]
    cols[3].metric("Current FPS", f"{sum(p.fps for p in running):.1f}" if running else "-")

    for cam_id, p in snapshot.items():
        icon = {"pending": "[..]", "running": "[>>]", "done": "[OK]", "error": "[!!]", "cancelled": "[--]"}.get(p.status, "")
        label = (f"{icon} {p.camera_name} - {p.status} - frame {p.frame_idx} - "
                 f"{p.frames_processed}/{p.frames_total} - elapsed {format_duration(p.elapsed)} - "
                 f"ETA {format_duration(p.eta)} - {p.fps:.1f} fps - active tracks {p.active_tracks}")
        st.progress(p.fraction, text=label)
        if p.message:
            st.caption(p.message)

    if show_previews:
        with job._lock:
            previews = dict(job.previews)
        if previews:
            st.markdown("**Live preview**")
            items = list(previews.items())
            cols = st.columns(min(2, len(items)))
            for i, (cam_id, jpeg) in enumerate(items[-2:]):
                name = snapshot[cam_id].camera_name if cam_id in snapshot else cam_id
                cols[i % len(cols)].image(jpeg, caption=name, width="stretch")

    with st.expander("Log", expanded=False):
        st.code("\n".join(get_memory_log_handler().tail(40)) or "(no log output yet)", language="text")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def render_dashboard(result: ProcessingResult) -> None:
    s = result.summary
    cols = st.columns(6)
    cols[0].metric("Cameras", s.get("total_cameras", 0))
    cols[1].metric("Unique persons", s.get("total_unique_persons", 0))
    cols[2].metric("Seen in 2+ cameras", s.get("persons_seen_in_multiple_cameras", 0))
    cols[3].metric("Tracks", s.get("total_tracks", 0), help="Local OC-SORT tracks over all cameras")
    cols[4].metric("Processing time", format_duration(s.get("processing_time_seconds", 0.0)))
    cols[5].metric("Average FPS", f"{s.get('average_fps', 0.0):.1f}")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Per camera**")
        rows = []
        for cam in result.camera_results:
            rows.append({
                "Camera": cam.camera_name,
                "File": cam.source_path.name,
                "Status": cam.status + (f" - {cam.error}" if cam.error else ""),
                "Frames": cam.frames_processed,
                "Resolution": f"{cam.width}x{cam.height}" if cam.width else "",
                "Video FPS": round(cam.fps_video, 2),
                "Proc. time": format_duration(cam.processing_time),
                "Proc. FPS": round(cam.processing_fps, 1),
                "Tracks": cam.n_tracks,
                "Assigned": cam.n_assigned_tracks,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Track status**")
        if len(result.tracks_df):
            counts = result.tracks_df["status"].value_counts().rename_axis("status").reset_index(name="tracks")
            st.bar_chart(counts.set_index("status"), width="stretch")
        st.markdown(
            _badge("MATCHED", STATUS_COLORS["MATCHED"]) + "existing Global ID reused " +
            _badge("NEW ID", STATUS_COLORS["NEW ID"]) + "new person " +
            _badge("UNCERTAIN", STATUS_COLORS["UNCERTAIN"]) + "new ID, close to an existing one " +
            _badge("SHORT", STATUS_COLORS["SHORT"]) + "track too short, ignored",
            unsafe_allow_html=True,
        )

    stats = s.get("identity_manager_stats", {})
    rejections = stats.get("covisibility_rejections", 0)
    thr = result.config_snapshot.get("reid", {}).get("similarity_threshold", "?")
    if rejections:
        st.info(
            f"Calibration hint: {rejections} times the best-matching identity scored above the similarity "
            f"threshold ({thr}) while that person was visible *elsewhere in the same camera at the same time* "
            f"(a physical impossibility, so the match was rejected). A high count means the threshold is too "
            f"permissive for this footage - consider raising it.")
    st.caption("Re-ID does not guarantee identity: MATCHED means the appearance similarity exceeded the "
               "threshold, UNCERTAIN means it was close but below it. Verify with the timeline and gallery.")


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def _identity_caption(ident: GlobalIdentity) -> str:
    cams = ",".join(str(v.camera_index) for v in _unique_visit_cameras(ident))
    return f"{ident.global_id} | Cam: {cams} | Visits: {ident.num_tracks}"


def _unique_visit_cameras(ident: GlobalIdentity):
    seen, out = set(), []
    for v in ident.visits:
        if v.camera_id not in seen:
            seen.add(v.camera_id)
            out.append(v)
    return out


def render_identity_details(ident: GlobalIdentity, result: ProcessingResult) -> None:
    first = ident.first_seen if np.isfinite(ident.first_seen) else 0.0
    c1, c2 = st.columns([1, 2])
    with c1:
        if ident.representative_image:
            st.image(ident.representative_image, caption=f"{ident.global_id} representative", width=180)
        st.markdown(
            f"**Global ID:** {ident.global_id}  \n"
            f"**Cameras:** {', '.join(ident.camera_names) or '-'}  \n"
            f"**First seen:** {format_seconds(first)}  \n"
            f"**Last seen:** {format_seconds(ident.last_seen)}  \n"
            f"**Total duration:** {format_duration(ident.total_duration)}  \n"
            f"**Number of tracks:** {ident.num_tracks}  \n"
            f"**Gallery size:** {len(ident.embeddings)} embeddings"
        )
    with c2:
        if ident.camera_images:
            st.markdown("**Best crop per camera**")
            imgs = list(ident.camera_images.items())
            cols = st.columns(min(6, len(imgs)))
            for i, (cam_id, jpeg) in enumerate(imgs):
                cols[i % len(cols)].image(jpeg, caption=ident.cameras_seen.get(cam_id, cam_id), width=110)
        rows = []
        for v in ident.visits:
            rows.append({
                "Camera": v.camera_name, "Local track": v.local_label, "Start": format_seconds(v.first_seen),
                "End": format_seconds(v.last_seen), "Duration": format_seconds(v.duration),
                "Similarity": round(v.similarity, 3) if v.similarity else None,
                "Avg similarity": round(v.avg_similarity, 3) if v.avg_similarity else None,
                "Status": v.status,
                "Closest other": f"{v.candidate_id} ({v.candidate_similarity:.2f})" if v.candidate_id else "",
                "Embeddings": v.n_embeddings,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_gallery(result: ProcessingResult, page_size: int = 24) -> None:
    identities = list(result.identities)
    if not identities:
        st.info("No global identities were created.")
        return
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    min_cams = f1.selectbox("Min. cameras", [1, 2, 3, 4], index=0, key="gal_min_cams")
    sort_by = f2.selectbox("Sort by", ["Global ID", "Most tracks", "Most cameras", "Longest duration"], key="gal_sort")
    only_multi = f3.checkbox("Only multi-track IDs", value=False, key="gal_multi")
    filtered = [i for i in identities if len(i.cameras_seen) >= min_cams and (not only_multi or i.num_tracks > 1)]
    if sort_by == "Most tracks":
        filtered.sort(key=lambda i: -i.num_tracks)
    elif sort_by == "Most cameras":
        filtered.sort(key=lambda i: (-len(i.cameras_seen), -i.num_tracks))
    elif sort_by == "Longest duration":
        filtered.sort(key=lambda i: -i.total_duration)
    pages = max(1, int(np.ceil(len(filtered) / page_size)))
    page = f4.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="gal_page")
    st.caption(f"{len(filtered)} identities (page {page}/{pages})")
    subset = filtered[(page - 1) * page_size: page * page_size]

    cols_per_row = 6
    for start in range(0, len(subset), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, ident in zip(cols, subset[start:start + cols_per_row]):
            with col:
                if ident.representative_image:
                    st.image(ident.representative_image, width="stretch")
                else:
                    st.markdown("*(no crop)*")
                st.caption(_identity_caption(ident))
                if st.button("Details", key=f"gal_btn_{ident.global_id}", width="stretch"):
                    st.session_state["selected_gid"] = ident.global_id
    selected = st.session_state.get("selected_gid")
    if selected:
        ident = result.identity(selected)
        if ident is not None:
            st.divider()
            render_identity_details(ident, result)


# --------------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------------- #
def render_videos(result: ProcessingResult) -> None:
    videos = result.processed_videos
    if not videos:
        st.info("No processed videos available (enable 'Generate output videos' before processing).")
        return
    names = {cam.camera_id: cam.camera_name for cam in result.camera_results}
    options = ["All Cameras"] + [names[c] for c in videos]
    choice = st.selectbox("Camera", options, key="video_choice")
    if choice == "All Cameras":
        items = list(videos.items())
        for start in range(0, len(items), 2):
            cols = st.columns(2)
            for col, (cam_id, path) in zip(cols, items[start:start + 2]):
                with col:
                    st.markdown(f"**{names[cam_id]}**")
                    _video(path)
    else:
        cam_id = next(c for c, n in names.items() if n == choice)
        _video(videos[cam_id])
        st.caption(f"{videos[cam_id]}")
    st.caption("If a video does not play inline, the file was written with an OpenCV codec (mp4v); "
               "install imageio-ffmpeg for H.264 output or open the file from Output/processed with a media player.")


def _video(path: Path) -> None:
    try:
        st.video(str(path))
    except Exception as exc:  # pragma: no cover - depends on browser/codec
        st.warning(f"Cannot display {path.name}: {exc}")


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
def render_timeline(result: ProcessingResult) -> None:
    df = result.tracks_df.copy()
    if df.empty:
        st.info("No tracks recorded.")
        return
    f1, f2, f3 = st.columns(3)
    gids = sorted(g for g in df["global_id"].unique() if isinstance(g, str) and g)
    sel_gids = f1.multiselect("Global IDs", gids, default=[], key="tl_gids", help="Empty = all")
    cams = list(dict.fromkeys(df["camera_name"].tolist()))
    sel_cams = f2.multiselect("Cameras", cams, default=cams, key="tl_cams")
    statuses = list(dict.fromkeys(df["status"].tolist()))
    sel_status = f3.multiselect("Status", statuses, default=[s for s in statuses if s not in ("SHORT", "NO_EMBEDDING")],
                                key="tl_status")
    view = df[df["camera_name"].isin(sel_cams) & df["status"].isin(sel_status)]
    if sel_gids:
        view = view[view["global_id"].isin(sel_gids)]
    show_cols = ["global_id", "camera_name", "local_label", "start_time", "end_time", "duration",
                 "assignment_similarity", "average_reid_similarity", "status", "candidate_id",
                 "candidate_similarity", "n_embeddings", "frames_observed"]
    st.dataframe(view[show_cols].rename(columns={
        "global_id": "Global ID", "camera_name": "Camera", "local_label": "Local Track", "start_time": "Start",
        "end_time": "End", "duration": "Duration", "assignment_similarity": "Similarity",
        "average_reid_similarity": "Avg similarity", "status": "Status", "candidate_id": "Closest other",
        "candidate_similarity": "Closest sim.", "n_embeddings": "Embeddings", "frames_observed": "Frames",
    }), width="stretch", hide_index=True, height=min(600, 60 + 28 * len(view)))

    chart_df = view[view["global_id"] != ""].copy()
    if chart_df.empty:
        return
    max_ids = st.slider("Identities shown in the chart", 5, 100, 30, key="tl_max_ids")
    top = chart_df.groupby("global_id")["duration_seconds"].sum().sort_values(ascending=False).head(max_ids).index
    chart_df = chart_df[chart_df["global_id"].isin(top)]
    try:
        import altair as alt

        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadius=2)
            .encode(
                x=alt.X("start_seconds:Q", title="Time (s)"),
                x2="end_seconds:Q",
                y=alt.Y("global_id:N", title="Global ID", sort=list(top)),
                color=alt.Color("camera_name:N", title="Camera"),
                tooltip=["global_id", "camera_name", "local_label", "start_time", "end_time", "duration",
                         "assignment_similarity", "status"],
            )
            .properties(height=max(200, 18 * len(top)))
        )
        st.altair_chart(chart, width="stretch")
    except Exception as exc:  # pragma: no cover - altair missing/old
        st.caption(f"Timeline chart unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def render_search(result: ProcessingResult, reid=None, identity_manager=None) -> None:
    identities = result.identities
    if not identities:
        st.info("Nothing to search yet.")
        return
    st.markdown("#### Search by Global ID")
    gid = st.selectbox("Search Person", [i.global_id for i in identities], key="search_gid")
    ident = result.identity(gid)
    if ident is not None:
        render_identity_details(ident, result)

    st.markdown("#### Search by query image")
    st.caption("Upload a cropped image of a person; OSNet embeds it and the closest Global IDs are listed.")
    upload = st.file_uploader("Query image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="search_upload")
    top_n = st.slider("Results", 1, 10, 5, key="search_topn")
    if upload is not None:
        if reid is None or identity_manager is None:
            st.warning("Models are not loaded in this session; load them (sidebar) to enable image search.")
            return
        import cv2

        data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not decode the uploaded image.")
            return
        emb = reid.embed_image(image)
        matches = identity_manager.search(emb, top_n=top_n)
        c1, c2 = st.columns([1, 3])
        c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Query", width=160)
        with c2:
            if not matches:
                st.info("No identities in the gallery.")
            for g, score in matches:
                ident = result.identity(g) or (identity_manager.get(g) if identity_manager else None)
                cols = st.columns([1, 4])
                if ident is not None and ident.representative_image:
                    cols[0].image(ident.representative_image, width=80)
                status = "MATCH" if score >= identity_manager.similarity_threshold else (
                    "UNCERTAIN" if score >= identity_manager.uncertain_threshold else "low")
                cols[1].markdown(f"**{g}** - similarity {score:.3f} ({status})  \n"
                                 f"Cameras: {', '.join(ident.camera_names) if ident else '-'}")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def render_reports(result: ProcessingResult) -> None:
    st.markdown(f"Output directory: `{result.output_dir.resolve()}`")
    for name, path in result.report_paths.items():
        path = Path(path)
        if path.is_file():
            with open(path, "rb") as fh:
                st.download_button(f"Download {path.name}", fh.read(), file_name=path.name, key=f"dl_{name}")
        elif path.is_dir():
            st.markdown(f"- `{path}` (directory)")
    with st.expander("summary.json", expanded=False):
        st.json(result.summary)
    with st.expander("identities.csv preview", expanded=False):
        st.dataframe(result.identities_df, width="stretch", hide_index=True)
    with st.expander("Processing log", expanded=False):
        st.code("\n".join(result.log_tail[-200:]) or "(empty)", language="text")
