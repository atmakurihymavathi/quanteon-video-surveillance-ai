"""
tracker.py
----------
A lightweight multi-object tracker that assigns persistent IDs to person
detections across frames using constant-velocity motion prediction,
IoU-based association, and the Hungarian algorithm.

This tracker provides short-term motion-based re-identification during
brief occlusions. It is not an appearance-based long-term re-identification
system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection
from .utils import BBox, iou

logger = logging.getLogger("surveillance")


@dataclass
class Track:
    """State for a single tracked person."""

    track_id: int
    bbox: BBox
    confidence: float
    velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False

    def predict(self) -> BBox:
        """Project the bounding box forward one frame."""
        x1, y1, x2, y2 = self.bbox
        vx1, vy1, vx2, vy2 = self.velocity

        return (
            x1 + vx1,
            y1 + vy1,
            x2 + vx2,
            y2 + vy2,
        )

    def update(self, detection: Detection) -> None:
        """Update this track using a matched detection."""
        old_bbox = self.bbox
        new_bbox = detection.bbox

        self.velocity = tuple(
            new_bbox[i] - old_bbox[i] for i in range(4)
        )  # type: ignore[assignment]

        self.bbox = new_bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self) -> None:
        """Advance a track when no detection was associated."""
        self.bbox = self.predict()

        self.velocity = tuple(
            v * 0.8 for v in self.velocity
        )  # type: ignore[assignment]

        self.time_since_update += 1
        self.age += 1


class PersonTracker:
    """Maintain persistent person IDs across video frames."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        if max_age < 0:
            raise ValueError("max_age must be >= 0")

        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Advance the tracker by one frame.

        Existing tracks are associated against their predicted positions.
        Unmatched tracks are retained for up to max_age frames, allowing
        brief occlusions without immediately creating a new ID.
        """
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        # Update matched tracks.
        for track_id, det_idx in matches:
            track = self.tracks[track_id]
            track.update(detections[det_idx])

            if track.hits >= self.min_hits:
                track.confirmed = True

        # Age unmatched tracks and move them using motion prediction.
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()

        # Create tracks for unmatched detections.
        for det_idx in unmatched_dets:
            det = detections[det_idx]

            new_track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                confidence=det.confidence,
            )

            if self.min_hits <= 1:
                new_track.confirmed = True

            self.tracks[self._next_id] = new_track
            self._next_id += 1

        # Remove tracks that have been missing for too long.
        stale_ids = [
            tid
            for tid, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]

        for tid in stale_ids:
            del self.tracks[tid]

        if stale_ids:
            logger.debug(
                "Removed %d stale track(s): %s",
                len(stale_ids),
                stale_ids,
            )

        return [
            track
            for track in self.tracks.values()
            if track.confirmed
        ]

    def _associate(
        self,
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate existing tracks with detections.

        Matching is performed against each track's predicted bounding box,
        not its stale last-observed bounding box.
        """
        track_ids = list(self.tracks.keys())

        if not track_ids or not detections:
            return (
                [],
                track_ids,
                list(range(len(detections))),
            )

        cost_matrix = np.ones(
            (len(track_ids), len(detections)),
            dtype=np.float32,
        )

        for row, track_id in enumerate(track_ids):
            predicted_box = self.tracks[track_id].predict()

            for col, detection in enumerate(detections):
                cost_matrix[row, col] = (
                    1.0 - iou(predicted_box, detection.bbox)
                )

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_detections = set()

        max_cost = 1.0 - self.iou_threshold

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= max_cost:
                track_id = track_ids[row]

                matches.append((track_id, col))
                matched_tracks.add(track_id)
                matched_detections.add(col)

        unmatched_tracks = [
            track_id
            for track_id in track_ids
            if track_id not in matched_tracks
        ]

        unmatched_detections = [
            idx
            for idx in range(len(detections))
            if idx not in matched_detections
        ]

        return (
            matches,
            unmatched_tracks,
            unmatched_detections,
        )
