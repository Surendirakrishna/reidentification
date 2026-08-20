"""Headless command-line runner (same pipeline as the Streamlit app).

Examples:
    python run_pipeline.py                         # all videos in Input Data, settings from config.yaml
    python run_pipeline.py --cameras cam1 cam2 --max-frames 600 --frame-skip 1
    python run_pipeline.py --threshold 0.6 --no-video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import load_config
from src.detector import PersonDetector
from src.pipeline import VideoProcessor
from src.reid import OSNetReID
from src.results import human_summary_lines
from src.utils import format_duration, get_logger, resolve_device, setup_logging
from src.video_utils import discover_videos

log = get_logger("cli")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-camera person Re-ID (YOLO + OC-SORT + OSNet)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--input", default=None, help="Input directory (default: config input.directory)")
    p.add_argument("--output", default=None, help="Output directory (default: config output.directory)")
    p.add_argument("--cameras", nargs="*", default=None, help="Camera ids / file stems to process (default: all)")
    p.add_argument("--max-frames", type=int, default=None, help="Max processed frames per camera (0 = all)")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--frame-skip", type=int, default=None)
    p.add_argument("--detection-interval", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="Re-ID similarity threshold")
    p.add_argument("--embedding-interval", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda:0")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--yolo", default=None, help="YOLO weights, e.g. yolo11n.pt")
    p.add_argument("--osnet", default=None, help="OSNet weights, e.g. osnet_x1_0_msmt17.pt")
    p.add_argument("--mode", choices=["sequential", "parallel"], default=None)
    p.add_argument("--no-video", action="store_true", help="Do not write annotated videos")
    p.add_argument("--save-crops", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if args.input:
        cfg.input.directory = args.input
    if args.output:
        cfg.output.directory = args.output
    if args.max_frames is not None:
        cfg.performance.max_frames = args.max_frames
    if args.start_frame is not None:
        cfg.performance.start_frame = args.start_frame
    if args.frame_skip is not None:
        cfg.performance.frame_skip = args.frame_skip
    if args.detection_interval is not None:
        cfg.performance.detection_interval = args.detection_interval
    if args.threshold is not None:
        cfg.reid.similarity_threshold = args.threshold
    if args.embedding_interval is not None:
        cfg.reid.embedding_interval = args.embedding_interval
    if args.device:
        cfg.detector.device = args.device
        cfg.reid.device = args.device
    if args.imgsz:
        cfg.detector.imgsz = args.imgsz
    if args.yolo:
        cfg.detector.model = args.yolo
    if args.osnet:
        cfg.reid.model = args.osnet
    if args.mode:
        cfg.performance.mode = args.mode
    if args.no_video:
        cfg.output.save_videos = False
    if args.save_crops:
        cfg.output.save_crops = True

    cameras = discover_videos(cfg.input.directory, cfg.input.extensions)
    if args.cameras:
        wanted = {c.lower() for c in args.cameras}
        cameras = [c for c in cameras if c.camera_id.lower() in wanted or c.path.stem.lower() in wanted
                   or c.camera_name.lower() in wanted]
    if not cameras:
        log.error("No videos found in %s", Path(cfg.input.directory).resolve())
        return 2

    dev = resolve_device(cfg.detector.device)
    log.info("Device: %s", dev.summary())
    t0 = time.time()
    detector = PersonDetector.from_config(cfg.detector)
    reid = OSNetReID.from_config(cfg.reid)
    log.info("Models loaded in %.1fs", time.time() - t0)

    processor = VideoProcessor(cfg, detector, reid)
    result = processor.run(cameras)
    print("\n".join(human_summary_lines(result.summary)))
    for name, path in result.report_paths.items():
        print(f"  {name}: {path}")
    for cam in result.camera_results:
        print(f"  {cam.camera_name}: {cam.status} {cam.frames_processed} frames in {format_duration(cam.processing_time)}"
              f" ({cam.processing_fps:.1f} fps), tracks={cam.n_tracks}, video={cam.output_video}")
    return 0 if all(c.status != "error" for c in result.camera_results) else 1


if __name__ == "__main__":
    sys.exit(main())
