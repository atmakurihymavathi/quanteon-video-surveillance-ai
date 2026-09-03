"""
utils.py
--------
Shared utility functions used across the surveillance pipeline:
bounding-box geometry, timestamp formatting, and small helpers that
don't belong to any single module.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("surveillance")

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection-over-Union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_bottom_center(box: BBox) -> Tuple[float, float]:
    """Return the bottom-center point of a bounding box.

    The bottom-center (feet position for a standing person) is used as the
    reference point for ground-plane zone membership because it is a much
    better proxy for "where a person is standing" than the box centroid,
    which shifts upward as a person's bounding box grows/shrinks with
    proximity to the camera.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def bbox_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clip_bbox(box: BBox, frame_w: int, frame_h: int) -> BBox:
    """Clamp a bounding box to valid frame coordinates."""
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, frame_w - 1))
    y1 = float(np.clip(y1, 0, frame_h - 1))
    x2 = float(np.clip(x2, 0, frame_w - 1))
    y2 = float(np.clip(y2, 0, frame_h - 1))
    return (x1, y1, x2, y2)


def frame_to_timestamp(frame_number: int, fps: float) -> str:
    """Convert a frame number to an MM:SS.ss timestamp string."""
    if fps <= 0:
        fps = 1.0
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist.

    Raises a clear, actionable error if the directory cannot be created
    (e.g. permission problems) instead of letting a raw OSError propagate.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create output directory '{path}': {exc}. "
            "Check that the path is valid and that you have write permission."
        ) from exc


@dataclass
class Polygon:
    """A simple polygon wrapper with validation."""

    points: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(
                f"Polygon must have at least 3 points, got {len(self.points)}."
            )
        for p in self.points:
            if len(p) != 2:
                raise ValueError(f"Polygon point must be an (x, y) pair, got {p}.")

    def as_np(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)


def load_json_config(path: str) -> dict:
    """Load and parse a JSON configuration file with clear error messages."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Check the --zones path or create a config file (see config/zones.json)."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in configuration file '{path}': {exc}"
        ) from exc


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure the root 'surveillance' logger with console + optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        ensure_dir(os.path.dirname(log_file) or ".")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
