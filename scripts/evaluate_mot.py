"""
Evaluate the surveillance tracker on MOT-Challenge formatted data.

The script runs the project's real detector + tracker on a video, exports
predictions in MOT format, and optionally computes standard MOT metrics
through motmetrics.

Expected ground-truth format:
    frame,id,x,y,w,h,conf,class,visibility

Prediction output:
    frame,id,x,y,w,h,confidence,-1,-1,-1
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List

import cv2

# Allow running this script directly from the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.detector import PersonDetector
from run import resolve_model_paths
from src.evaluation import (
    PredBox,
    evaluate,
    load_mot_ground_truth,
    to_motmetrics_accumulator,
)
from src.tracker import PersonTracker


logger = logging.getLogger("mot-evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the surveillance tracker on MOT-Challenge data."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the MOT video file.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to MOT ground-truth gt.txt.",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation",
        help="Directory for predictions and metric output.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detector NMS IoU threshold.",
    )
    parser.add_argument(
        "--tracker-iou",
        type=float,
        default=0.30,
        help="Tracker association IoU threshold.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum number of missed frames before a track is removed.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help="Number of successful matches required to confirm a track.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=416,
        help="YOLO input size.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used for the project's basic evaluation.",
    )
    parser.add_argument(
    "--mot17-pedestrians-only",
    action="store_true",
    help=(
        "For MOT17 ground truth, evaluate only valid pedestrian targets "
        "(confidence=1, class=1)."
    ),
)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def write_predictions(path: str, predictions: List[PredBox]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for p in predictions:
            x1, y1, x2, y2 = p.bbox
            writer.writerow(
                [
                    p.frame,
                    p.track_id,
                    f"{x1:.3f}",
                    f"{y1:.3f}",
                    f"{x2 - x1:.3f}",
                    f"{y2 - y1:.3f}",
                    f"{p.confidence:.6f}",
                    -1,
                    -1,
                    -1,
                ]
            )


def write_summary(path: str, result, mot_summary=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("MOT Tracking Evaluation\n")
        f.write("=======================\n\n")

        f.write(f"Frames evaluated: {result.frames_evaluated}\n")
        f.write(f"Ground-truth boxes: {result.total_gt_boxes}\n")
        f.write(f"Predicted boxes: {result.total_pred_boxes}\n")
        f.write(f"True positives: {result.true_positives}\n")
        f.write(f"False positives: {result.false_positives}\n")
        f.write(f"False negatives: {result.false_negatives}\n")
        f.write(f"Precision: {result.precision:.4f}\n")
        f.write(f"Recall: {result.recall:.4f}\n")
        f.write(f"F1: {result.f1:.4f}\n")
        f.write(f"ID switches: {result.id_switches}\n")

        if mot_summary is not None:
            f.write("\nStandard MOTMetrics\n")
            f.write("===================\n")
            f.write(str(mot_summary))
            f.write("\n")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not os.path.isfile(args.video):
        logger.error("Video not found: %s", args.video)
        return 1

    if not os.path.isfile(args.gt):
        logger.error("Ground truth not found: %s", args.gt)
        return 1

    os.makedirs(args.output, exist_ok=True)

    logger.info("Loading detector...")
    model_cfg, model_weights, class_names = resolve_model_paths(None)

    detector = PersonDetector(
        model_cfg=model_cfg,
        model_weights=model_weights,
        class_names=class_names,
        confidence_threshold=args.confidence,
        nms_threshold=args.iou,
        input_size=args.input_size,
        device=args.device,
    )

    tracker = PersonTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.tracker_iou,
    )

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        logger.error("Could not open video: %s", args.video)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Opened MOT video: %.2f FPS, approximately %d frames",
        fps,
        total_frames,
    )

    predictions: List[PredBox] = []
    frame_number = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                break

            frame_number += 1

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            for track in tracks:
                predictions.append(
                    PredBox(
                        frame=frame_number,
                        track_id=track.track_id,
                        bbox=track.bbox,
                        confidence=track.confidence,
                    )
                )

            if frame_number % 100 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                logger.info(
                    "Processed %d/%d frames | %.2f FPS | predictions=%d",
                    frame_number,
                    total_frames,
                    frame_number / elapsed,
                    len(predictions),
                )

    finally:
        cap.release()

    elapsed = max(time.time() - start_time, 1e-6)

    logger.info(
        "Finished inference: %d frames in %.2fs (%.2f FPS)",
        frame_number,
        elapsed,
        frame_number / elapsed,
    )

    prediction_path = os.path.join(
        args.output,
        "predictions.txt",
    )

    write_predictions(prediction_path, predictions)

    logger.info("Predictions written to: %s", prediction_path)

    ground_truth = load_mot_ground_truth(
    args.gt,
    mot17_pedestrians_only=args.mot17_pedestrians_only,
)
    basic_result = evaluate(
        ground_truth,
        predictions,
        iou_threshold=args.match_iou,
    )

    logger.info("\n%s", basic_result.summary())

    mot_summary = None

    accumulator = to_motmetrics_accumulator(
        ground_truth,
        predictions,
    )

    if accumulator is not None:
        import motmetrics as mm

        metrics = [
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects",
        ]

        try:
            mh=mm.metrics.create()
            summary=mh.compute(
                accumulator,
                metrics=metrics,
                name="M0T17-02-FRCNN",
            )

            mot_summary = summary.to_string()

            logger.info(
                "\nStandard MOT metrics:\n%s",
                mot_summary,
            )

        except Exception as exc:
            logger.warning(
                "Could not compute standard MOT metrics: %s",
                exc,
            )

    summary_path = os.path.join(
        args.output,
        "metrics.txt",
    )

    write_summary(
        summary_path,
        basic_result,
        mot_summary,
    )

    logger.info("Metrics written to: %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
