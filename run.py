#!/usr/bin/env python3
"""
run.py
------
CLI entry point for the video surveillance pipeline.

Supports processing one or multiple videos in a single command.

Examples:
    python run.py \
        --video data/people-detection.mp4 \
        --zones config/zones.json \
        --output results/

    python run.py \
        --video data/evaluation/clip1.mp4 data/evaluation/clip2.mp4 \
        --zones config/zones.json \
        --output results/evaluation/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Allow running as `python run.py` from the repo root without installation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import DetectorError, PersonDetector  # noqa: E402
from src.pipeline import PipelineError, SurveillancePipeline  # noqa: E402
from src.tracker import PersonTracker  # noqa: E402
from src.utils import setup_logging  # noqa: E402
from src.zone_manager import ZoneConfigError, ZoneManager  # noqa: E402


DEFAULT_MODEL_CFG = os.path.join("models", "yolov4-tiny.cfg")
DEFAULT_MODEL_WEIGHTS = os.path.join("models", "yolov4-tiny.weights")
DEFAULT_CLASS_NAMES = os.path.join("models", "coco.names")

logger = logging.getLogger("surveillance")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Video surveillance: person detection, tracking, and "
            "zone-based event recognition (intrusion + loitering)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video",
        nargs="+",
        required=True,
        help="One or more input video files.",
    )

    parser.add_argument(
        "--zones",
        required=True,
        help="Path to the zones JSON config file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path prefix for a custom Darknet model, e.g. "
            "'models/my_model' expecting 'my_model.cfg' and "
            "'my_model.weights' alongside it. Defaults to "
            "the bundled YOLOv4-tiny model."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for detection.",
    )

    parser.add_argument(
        "--tracker-iou",
        type=float,
        default=0.3,
        help="IoU threshold used for track-to-detection association.",
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help=(
            "Max frames a track survives without a matching detection "
            "before being dropped."
        ),
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help=(
            "Consecutive matched frames required before a new track "
            "is confirmed."
        ),
    )

    parser.add_argument(
        "--loitering-seconds",
        type=float,
        default=None,
        help=(
            "Override loitering_seconds for ALL zones. Otherwise, "
            "per-zone values from the zones config are used."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Inference device. 'auto' uses CUDA if supported and "
            "available, otherwise CPU."
        ),
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=416,
        help="Detector network input resolution (square).",
    )

    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Write annotated output videos.",
    )

    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Skip writing annotated videos.",
    )

    parser.add_argument(
        "--save-events",
        dest="save_events",
        action="store_true",
        default=True,
        help="Write JSON/CSV event logs.",
    )

    parser.add_argument(
        "--no-save-events",
        dest="save_events",
        action="store_false",
        help="Skip writing event logs.",
    )

    parser.add_argument(
        "--event-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Which event log format(s) to write.",
    )

    parser.add_argument(
        "--fps-log-interval",
        type=int,
        default=100,
        help="Log progress/FPS every N frames. 0 disables periodic logging.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to also write application logs to a file.",
    )

    return parser


def resolve_model_paths(model_prefix: str | None) -> tuple[str, str, str]:
    if model_prefix is None:
        return DEFAULT_MODEL_CFG, DEFAULT_MODEL_WEIGHTS, DEFAULT_CLASS_NAMES

    return (
        f"{model_prefix}.cfg",
        f"{model_prefix}.weights",
        DEFAULT_CLASS_NAMES,
    )


def build_pipeline(args, detector, zone_manager) -> SurveillancePipeline:
    if args.loitering_seconds is not None:
        for zone in zone_manager.zones:
            zone.loitering_seconds = args.loitering_seconds

    tracker = PersonTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.tracker_iou,
    )

    event_formats = (
        ("json", "csv")
        if args.event_format == "both"
        else (args.event_format,)
    )

    return SurveillancePipeline(
        detector=detector,
        tracker=tracker,
        zone_manager=zone_manager,
        output_dir=args.output,
        save_video=args.save_video,
        save_events=args.save_events,
        event_formats=event_formats,
        fps_log_interval=args.fps_log_interval,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(
        verbose=args.verbose,
        log_file=args.log_file,
    )

    if args.loitering_seconds is not None and args.loitering_seconds <= 0:
        logger.error(
            "--loitering-seconds must be > 0, got %s",
            args.loitering_seconds,
        )
        return 2

    try:
        model_cfg, model_weights, class_names = resolve_model_paths(args.model)

        detector = PersonDetector(
            model_cfg=model_cfg,
            model_weights=model_weights,
            class_names=class_names,
            confidence_threshold=args.confidence,
            nms_threshold=args.iou,
            input_size=args.input_size,
            device=args.device,
        )

    except DetectorError as exc:
        logger.error("Failed to initialize detector: %s", exc)
        return 1

    try:
        zone_manager = ZoneManager.from_file(args.zones)

    except (ZoneConfigError, FileNotFoundError, ValueError) as exc:
        logger.error(
            "Failed to load zone configuration: %s",
            exc,
        )
        return 1

    if args.loitering_seconds is not None:
        for zone in zone_manager.zones:
            zone.loitering_seconds = args.loitering_seconds

        logger.info(
            "Overriding loitering_seconds=%.1f for all %d zone(s)",
            args.loitering_seconds,
            len(zone_manager.zones),
        )

    total_videos = len(args.video)

    logger.info(
        "Starting surveillance processing for %d video(s)",
        total_videos,
    )

    failed_videos = 0

    for index, video_path in enumerate(args.video, start=1):
        logger.info(
            "========== Video %d/%d: %s ==========",
            index,
            total_videos,
            video_path,
        )

        try:
            # Create a fresh tracker/pipeline for every video.
            # This prevents track IDs and event state from leaking
            # between unrelated videos.
            pipeline = build_pipeline(
                args,
                detector,
                zone_manager,
            )

            stats = pipeline.run(video_path)

            logger.info(
                "Completed video %d/%d: %s | "
                "frames=%d | detections=%d | "
                "intrusions=%d | loitering=%d | FPS=%.2f",
                index,
                total_videos,
                video_path,
                stats.frames_processed,
                stats.total_detections,
                stats.intrusion_events,
                stats.loitering_events,
                stats.processing_fps,
            )

        except PipelineError as exc:
            failed_videos += 1
            logger.error(
                "Pipeline failed for '%s': %s",
                video_path,
                exc,
            )

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            return 130

        except Exception as exc:  # noqa: BLE001
            failed_videos += 1
            logger.exception(
                "Unexpected error while processing '%s': %s",
                video_path,
                exc,
            )

    if failed_videos:
        logger.error(
            "Processing finished with %d failed video(s) out of %d.",
            failed_videos,
            total_videos,
        )
        return 1

    logger.info(
        "All %d video(s) processed successfully.",
        total_videos,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
