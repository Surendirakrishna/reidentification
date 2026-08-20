# Multi-Camera Person Re-Identification (YOLO + OC-SORT + OSNet + Streamlit)

A modular, GPU-accelerated Streamlit application that

1. detects people with **YOLO** (Ultralytics, `yolo11n.pt` by default),
2. tracks them **inside each camera** with **OC-SORT** (BoxMOT implementation, one tracker per camera),
3. embeds person crops with **OSNet** (torchreid architecture + MSMT17 weights via BoxMOT),
4. links tracks **across cameras** into **Global Person IDs** with a cosine-similarity gallery
   (`GlobalIdentityManager`), and
5. renders annotated videos, a dashboard, a person gallery, a track timeline, Re-ID search and CSV/JSON reports.

Local OC-SORT ids (`C1-T03`) and Global IDs (`P001`) are strictly separate: a Global ID is only
ever created by appearance matching, never by reusing a tracker id.

```
Camera 1: Local Track C1-T03  ->  Global ID P001
Camera 2: Local Track C2-T17  ->  Global ID P001
Camera 3: Local Track C3-T05  ->  Global ID P001
```

---

## 1. Project overview

| Component | Library / implementation | Role |
|-----------|--------------------------|------|
| Person detection | `ultralytics` (YOLO11 / YOLOv8, person class only) | boxes + confidences per frame |
| Tracking | `boxmot.trackers.bbox.ocsort.OcSort` (OC-SORT) | temporally consistent local track ids per camera |
| Re-ID | `boxmot.reid.core.ReID` -> OSNet (`osnet_x1_0_msmt17.pt`, 512-d) | appearance embeddings (L2-normalised) |
| Global identities | `src/global_identity.py` | cosine-similarity gallery matching, MATCHED / NEW ID / UNCERTAIN |
| UI | Streamlit | configuration, progress, dashboard, gallery, timeline, search |
| Video I/O | OpenCV (+ bundled ffmpeg via `imageio-ffmpeg`) | reading, H.264 writing |

Sample footage in `Input Data/` (7 synchronised 1080p/60 fps cameras of a crowded square) is
processed in the "Development log" section at the end.

## 2. Architecture

```
Input Data/*.mp4 ──► VideoReader ──► PersonDetector (YOLO) ──► OCSortTracker (per camera)
                                                                      │  local track ids
                                                                      ▼
                                              TrackManager (TrackState per local track)
                                                │ first appearance + every N frames
                                                ▼
                               crop -> quality check -> OSNetReID.extract (batched, fp16 on CUDA)
                                                │ L2-normalised embeddings
                                                ▼
                     GlobalIdentityManager (shared across ALL cameras, thread-safe)
                       cosine similarity vs. identity galleries -> P001 / P002 / ...
                                                │
                       ┌────────────────────────┼──────────────────────────┐
                       ▼                        ▼                          ▼
              visualization.py          ResultsManager                 Streamlit UI
         annotated videos (H.264)   tracks.csv / identities.csv    dashboard, gallery,
                                     summary.json / embeddings.npz  timeline, search
```

```
app.py                  Streamlit UI (sidebar config, background job, dashboard tabs)
run_pipeline.py         headless CLI runner (same pipeline)
config.yaml             all settings (overridable in the sidebar / CLI flags)
src/
  config.py             typed config dataclasses + YAML loading
  utils.py              logging, device selection, time/maths helpers
  video_utils.py        video discovery, VideoReader, VideoWriter (ffmpeg/OpenCV), bbox & crop helpers
  detector.py           PersonDetector (YOLO)
  tracker.py            OCSortTracker (BoxMOT OC-SORT wrapper) - one instance per camera
  reid.py               OSNetReID (preprocessing, batched inference), crop quality scoring
  global_identity.py    GlobalIdentityManager, GlobalIdentity, MatchResult / MatchStatus
  camera_processor.py   TrackState, TrackManager, CameraProcessor (single camera pipeline)
  pipeline.py           VideoProcessor (sequential / parallel multi-camera orchestration)
  results.py            ResultsManager (CSV / JSON / NPZ / crops), ProcessingResult
  job.py                ProcessingJob (background thread used by the UI)
  visualization.py      overlays (boxes, labels, HUD)
  ui_components.py      Streamlit rendering functions
tests/                  pytest suite (maths, identity manager, track manager, tracker, video utils, app smoke test)
models/                 downloaded weights (yolo11n.pt, osnet_x1_0_msmt17.pt, ...)
Input Data/             input videos        Output/   processed videos, crops, embeddings, reports
```

## 3. Installation

Requirements: Python 3.10-3.13 (3.12 recommended), Windows/Linux/macOS, optional NVIDIA GPU.

```powershell
# Windows (creates .venv, installs CUDA torch automatically when nvidia-smi is present)
.\setup.ps1
.\run.ps1                      # -> http://localhost:8501
```

Manual installation:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows    |   source .venv/bin/activate   # Linux/macOS
python -m pip install --upgrade pip
# GPU: install the CUDA build of torch first (pick the CUDA tag supported by your driver, e.g. cu126/cu128/cu130)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
streamlit run app.py
```

Notes
* On this machine the environment lives outside OneDrive at `C:\Users\suren\.venvs\eye-reid`
  (`run.ps1` finds it automatically; set `EYE_REID_VENV` to use another path).
* `boxmot` pins `opencv-python<5` and `pandas<3`; `requirements.txt` respects that.
* Without a GPU everything works on CPU (expect ~1-3 fps at 1080p with `imgsz=1280`; use `frame_skip`, `imgsz=640`).

## 4. Model setup

Weights are downloaded automatically into `models/` on first start:

* YOLO: `yolo11n.pt` from the Ultralytics release assets (any `yolo11*/yolov8*` detection checkpoint or a custom
  `.pt` path works - `detector.model`).
* OSNet: `osnet_x1_0_msmt17.pt` from the torchreid model zoo (Google Drive, via BoxMOT's downloader).
  Alternatives selectable in the sidebar: `osnet_ain_x1_0_msmt17.pt` (instance-norm variant, better under
  domain shift), `osnet_x0_25_msmt17.pt` (fastest), Market-1501 / DukeMTMC weights.

If downloads are blocked, place the files manually in `models/` (see `models/README.md`).

## 5. Input video format

Put one file per camera into `Input Data/` (`.mp4`, `.avi`, `.mkv`, `.mov`). The folder is scanned on
start ("Rescan folder" button). Camera names are inferred from file names:

| file | camera id | display name |
|------|-----------|--------------|
| `cam1.mp4` | `cam1` | Camera 1 |
| `camera_02.mp4` | `camera_02` | Camera 2 |
| `entrance.mp4` | `entrance` | Entrance |
| `parking_lot.mkv` | `parking_lot` | Parking Lot |

Optional wall-clock start times per camera (`input.camera_start_times` in `config.yaml`) shift the timeline
so that "Person appears in Camera 1 at 10:05 and in Camera 2 at 10:09" is displayed with real clock times.
Videos from different cameras do **not** need to be synchronised or overlap in time.

## 6. Running the application

```bash
streamlit run app.py            # UI
python run_pipeline.py --help   # headless, same pipeline (e.g. --cameras cam1 cam2 --max-frames 600 --frame-skip 1)
python -m pytest                # tests (add -m "not slow" to skip the model-loading tests)
```

Workflow: select cameras -> tune settings -> **START PROCESSING** -> watch per-camera progress, ETA, FPS,
live preview and log -> explore the **Dashboard**, **Person Gallery**, **Camera Videos**, **Track Timeline**,
**Re-ID Search** and **Reports & Logs** tabs. Processing runs in a background thread, so the UI stays
responsive and **STOP** cancels gracefully (partial results are still written). The last run can be reloaded
from `Output/reports/last_run.pkl` after restarting the app.

## 7. Streamlit controls

| Sidebar section | Controls |
|-----------------|----------|
| Input Folder | directory, rescan, per-camera checkboxes |
| Detection | YOLO model, confidence, IoU, image size, device (auto/cpu/cuda) |
| Tracking (OC-SORT) | max age, min hits, association IoU, min track frames for a Global ID |
| Re-ID (OSNet) | OSNet weights, similarity threshold (+ calibration warning), uncertain margin, embedding interval, gallery size, matching strategy, batch size |
| Processing | save crops / embeddings / videos, frame skip, detection interval, max frames, start frame, sequential/parallel mode |

The status line shows device, GPU, VRAM, torch version, video encoder and model load state
(`✓ YOLO loaded ✓ OC-SORT initialized ✓ OSNet loaded ✓ CUDA available`). Models are cached with
`st.cache_resource` and only reload when the model name / device / image size changes.

## 8. YOLO configuration (`detector`)

| key | default | meaning |
|-----|---------|---------|
| `model` | `yolo11n.pt` | any Ultralytics detection checkpoint (n/s/m/l/x, custom path) |
| `confidence` | 0.40 | minimum detection confidence |
| `iou` | 0.50 | NMS IoU |
| `imgsz` | 1280 | inference size; 640 fastest, 960/1280 find small, distant people (the sample 1080p footage went from 8 to 20 detections per frame at almost no GPU cost) |
| `device` | auto | `auto` -> CUDA when available, else CPU |
| `half` | auto | FP16 on CUDA (`quantize='fp16'` / `half=True` depending on the Ultralytics version) |
| `classes` | `[person]` | only the person class enters the tracker |

## 9. OC-SORT configuration (`tracker`)

| key | default | meaning |
|-----|---------|---------|
| `det_thresh` | 0.30 | OC-SORT high-confidence threshold (detections below it are dropped unless `use_byte`) |
| `max_age` | 30 | processed frames a lost track survives (also the time a track can be re-confirmed) |
| `min_hits` | 3 | consecutive hits before a track is reported |
| `iou_threshold` | 0.30 | association threshold |
| `delta_t`, `inertia` | 3, 0.2 | observation-centric momentum parameters |
| `asso_func` | iou | `iou`, `giou`, `ciou`, `diou`, `centroid` |
| `use_byte` | false | ByteTrack-style second association with low-score detections |

Every camera gets its **own** `OCSortTracker` instance (`create_trackers()` / `VideoProcessor._make_processor`),
ids restart at 0 per camera and are displayed as `C<camera>-T<id>`. `max_age` is counted in *processed* frames,
so with `frame_skip = 1` it covers twice as many real frames.

## 10. OSNet configuration (`reid`)

| key | default | meaning |
|-----|---------|---------|
| `model` | `osnet_x1_0_msmt17.pt` | OSNet weights (architecture derived from the file name) |
| `similarity_threshold` | 0.70 | cosine similarity needed to reuse an existing Global ID |
| `uncertain_margin` | 0.08 | `[threshold - margin, threshold)` -> new ID flagged **UNCERTAIN** (never auto-merged) |
| `embedding_interval` | 10 | re-embed an active track every N processed frames (first appearance always) |
| `max_embeddings_per_identity` | 20 | gallery size per Global ID (front/side/back views, lighting...) |
| `max_embeddings_per_track` | 5 | one track can contribute at most this many gallery samples |
| `batch_size` | 16 | crops per OSNet forward pass (all due tracks of a frame are batched) |
| `matching_strategy` | `top_k_tracks` | `max`, `mean`, `top_k` (k best samples) or `top_k_tracks` (mean of the k best tracks, each the mean of its k best samples - robust to one wrong merge) |
| `top_k` | 3 | k for the top-k strategies |
| `min_crop_width/height/area` | 32 / 64 / 2048 | crops below these sizes are not embedded |
| `min_sharpness` | 0 | Laplacian-variance blur filter (0 = off) |
| `max_pending_embeddings` | 3 | embeddings collected before a NEW/UNCERTAIN decision becomes final |

Preprocessing follows the torchreid/BoxMOT OSNet recipe: crop -> clamp to the frame -> resize to the model's
input (256x128, read from the loaded backend) -> BGR->RGB -> [0,1] -> ImageNet mean/std -> OSNet -> L2
normalisation. Inference runs under `torch.inference_mode()` with `model.eval()`, FP16 on CUDA, batched.

Track -> Global ID logic (`TrackManager` + `GlobalIdentityManager.resolve_track`):

1. A track is embedded at its first appearance and then every `embedding_interval` frames (only while it is
   actually observed; stale tracks cost nothing). Crops are quality-scored (size, aspect ratio, sharpness,
   detector confidence, overlap with other boxes) - the best crop becomes the representative image.
2. Once a track has lived `min_track_frames` frames and has embeddings, the mean embedding is matched against
   all identity galleries.
   * best score >= threshold -> **MATCHED** (existing Global ID; the track's samples enrich that gallery)
   * otherwise the decision is deferred until `max_pending_embeddings` samples exist or the track ends, then
     **NEW ID** (clearly new) or **UNCERTAIN** (within the margin; new ID, closest candidate recorded).
   * Tracks shorter than `min_track_frames` are **SHORT**, tracks without any usable crop **NO_EMBEDDING** -
     neither gets a Global ID.
3. Identities that are visible *at the same time elsewhere in the same camera* are excluded as candidates
   (a person cannot be in two places at once). The number of such rejections is shown as a calibration hint.
4. Galleries are quality-gated: near duplicates are skipped, low-quality samples are replaced by better ones,
   a single track cannot flood a gallery, and later samples that disagree with the identity are not added.

## 11. Re-ID threshold calibration

> Re-ID similarity thresholds are model- and environment-dependent. The threshold should be calibrated using
> representative footage from the actual cameras.

Procedure:

1. Process a representative slice (e.g. `max_frames = 1500`, 2-3 cameras) with the default 0.70.
2. Open **Track Timeline** / **Person Gallery** and inspect identities with many tracks: crops of different
   people under one ID -> raise the threshold; the same person split into many IDs -> lower it.
3. Watch the dashboard's **calibration hint** ("N impossible same-camera matches above the threshold"): a high
   number means the threshold is too permissive for this footage.
4. `tracks.csv` contains `assignment_similarity` (MATCHED) and `candidate_similarity` (NEW/UNCERTAIN); plot both
   distributions - the threshold should sit in the gap between them.
5. Re-run; repeat per site. Different OSNet weights produce different similarity scales (AIN weights score
   lower), so recalibrate after changing the model.

On the crowded sample footage 0.65 produced "hub" identities that absorbed several dark-jacket people;
0.70 with the `top_k_tracks` strategy and co-visibility exclusion gave visibly cleaner galleries.

## 12. Output structure

```
Output/
├── processed/      cam1_reid.mp4, cam2_reid.mp4, ...   (H.264 via ffmpeg; OpenCV mp4v fallback)
├── crops/          P001/representative.jpg, P001/cam1/best.jpg, P001/cam1/cam1_T03_f3000.jpg ... (save_crops)
├── embeddings/     embeddings.npz  (gallery_embeddings, gallery_owner, gallery_source, mean_embeddings,
│                                    track_embeddings, track_keys, track_global_ids)
└── reports/
    ├── tracks.csv        global_id, camera_id, local_track_id, start/end frame & time, duration,
    │                     max_detection_confidence, average_reid_similarity, assignment_similarity, status,
    │                     candidate_id, candidate_similarity, n_embeddings, frames_observed
    ├── identities.csv    global_id, cameras_seen, first_seen, last_seen, total_duration, number_of_tracks, ...
    ├── summary.json      total_cameras, total_unique_persons, total_tracks, processing_time_seconds,
    │                     average_fps, status breakdown, per-camera stats, identity manager stats
    ├── config_used.json  the exact configuration of the run
    └── last_run.pkl      full session state (reload in the UI)
```

Overlay on the videos: bounding box, `P001 | C1-T03`, `Person 0.91`, `ReID 0.83 MATCHED` (or `pending` /
`UNCERTAIN`), a small Global-ID tag at the bottom-right corner of each box and a HUD with camera, frame,
time, active tracks and number of Global IDs. Colours are derived from the Global ID but the text is the primary cue.

## 13. Performance optimization

* **Frame skip** (`performance.frame_skip`): 1 = every second frame -> 2x faster; the sample 60 fps footage is
  perfectly trackable at 30 or 20 fps.
* **Detection interval**: run YOLO/OC-SORT every N processed frames and freeze boxes in between (cheap, less accurate).
* **Image size**: 640 is fastest; 1280 recovers small people on 1080p footage (GPU cost is small, CPU cost is 4x).
* **Embedding interval / gallery size / batch size** control OSNet cost; all crops of a frame are embedded in
  one batched call, stale tracks are never embedded, crops below the minimum size are skipped.
* **FP16** on CUDA for YOLO and OSNet, `torch.inference_mode()`, a single CPU->GPU transfer per batch, no
  per-frame copies (crops are views), preview JPEGs only every N frames.
* **Video writing**: ffmpeg `h264_nvenc` when available, else `libx264 -preset veryfast`; `video_scale` shrinks
  the output; disable videos for pure analytics runs.
* **Parallel mode** processes several cameras in threads that share the same YOLO/OSNet instances (no extra
  GPU memory); useful when decoding/encoding dominates. Sequential mode is the memory-friendly default.
* Measured on an RTX 2050 (4 GB), 1080p input, `imgsz=1280`, `frame_skip=1`, H.264 output: ~17-20 processed
  frames/s per camera including video encoding; CPU-only machines should use `imgsz=640`, `frame_skip>=2`.

## 14. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Device: CPU` although you have an NVIDIA GPU | install the CUDA torch build (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`); check `nvidia-smi` |
| OSNet download fails (Google Drive quota / blocked) | download the `.pt` manually into `models/` (names in `models/README.md`) |
| YOLO download fails | check internet / proxy, or copy `yolo11n.pt` into `models/` |
| Video does not play in the browser | it was written with OpenCV `mp4v` (no ffmpeg found). `pip install imageio-ffmpeg` or set `output.video_codec: libx264`; the file still plays in VLC |
| "Cannot open video" / corrupt file | unsupported codec - re-encode with ffmpeg (`ffmpeg -i in.avi -c:v libx264 out.mp4`); the camera is reported as failed, other cameras continue |
| Too many / too few Global IDs | calibrate `similarity_threshold` (section 11), check `min_track_frames`, crop size limits |
| Many SHORT tracks | crowded scenes fragment tracks; raise `max_age`, lower `min_hits`, or lower `min_track_frames` |
| GPU out of memory | smaller `imgsz`, smaller OSNet (`osnet_x0_25`), lower `reid.batch_size`, sequential mode |
| Streamlit reruns reload models | they are cached; only changing model/device/image size triggers a reload |
| Tests are slow | `pytest -m "not slow"` skips model-loading tests |

## 15. Limitations

* Appearance-based Re-ID cannot guarantee identity: people in similar clothing, heavy occlusion, tiny or
  blurry crops, strong colour casts between cameras and cross-camera resolution differences all reduce
  accuracy. Statuses (MATCHED / NEW ID / UNCERTAIN) and similarity values are exposed everywhere so that
  results can be verified; uncertain identities are never merged automatically (a manual `merge()` exists).
* No spatial/temporal camera topology is used (except same-camera co-visibility); adding travel-time
  constraints between cameras would reduce false matches.
* OC-SORT is a motion-only tracker: in dense crowds tracks fragment, which increases the number of local
  tracks (Re-ID then re-links the fragments when appearance allows).
* Detection interval > 1 freezes boxes between detection frames; it trades accuracy for speed.
* Timestamps are video-relative unless `camera_start_times` are configured; the cameras are not synchronised
  by the application.
* The pipeline is offline (pre-recorded files). `VideoReader` is the only place that touches the source, so
  an RTSP reader (cv2.VideoCapture on a URL + reconnect logic) can be dropped in for live cameras later.

## Development log / verification on the sample footage

* Environment: Windows 11, Python 3.12.10, torch 2.13.0+cu130 on an NVIDIA RTX 2050 (4 GB), ultralytics 8.4.122,
  boxmot 22.0.0, opencv 4.14, streamlit 1.61.1.
* `python -m pytest` -> 39 tests pass (maths, identity manager, track manager, real OC-SORT, video I/O, real
  OSNet inference, Streamlit AppTest end-to-end smoke test). The Streamlit server itself was started and
  probed (`/healthz` 200).
* `python run_pipeline.py --max-frames 600 --start-frame 3000 --frame-skip 1 --save-crops` over all 7 sample
  cameras: 4m 36s total (~15-19 processed fps per camera incl. H.264 encoding), 752 local tracks, 252 global
  identities, 89 of them re-identified in 2+ cameras (e.g. `P002` linked across cam1/cam3/cam4/cam7 - verified
  visually against the saved crops). Outputs: `Output/processed/cam*_reid.mp4`, `tracks.csv`, `identities.csv`,
  `summary.json`, `embeddings.npz`, representative crops.
* Parallel mode (2 workers, shared models) and corrupt-video handling (one broken file -> camera marked
  `error`, remaining cameras finish) verified with dedicated runs.

## Next recommended improvements

1. Camera-topology constraints (minimum travel time between camera pairs) and a per-pair similarity threshold.
2. Deep-OC-SORT / BoT-SORT (appearance-aided tracking) to reduce fragmentation inside a camera.
3. Offline global re-clustering of track embeddings after the run (agglomerative clustering with the
   co-visibility constraints) for higher cross-camera consistency.
4. RTSP / live-stream reader with reconnect, ring buffers and per-camera worker processes.
5. Evaluation tooling: import ground-truth annotations (e.g. WILDTRACK) and compute IDF1 / HOTA.
