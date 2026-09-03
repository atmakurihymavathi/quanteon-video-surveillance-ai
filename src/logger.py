"""
logger.py
---------
Structured event log persistence (JSON and CSV) for detected events.
Kept separate from Python's `logging` (see utils.setup_logging) so that
"application logs" (for developers) and "event logs" (the product output,
for security operators) don't get mixed together.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import List

from .events import Event
from .utils import ensure_dir

logger = logging.getLogger("surveillance")

EVENT_CSV_FIELDS = [
    "event_id",
    "event_type",
    "track_id",
    "zone",
    "frame_number",
    "timestamp",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "duration_seconds",
]


class EventLogWriter:
    """Writes accumulated events to JSON and/or CSV on demand."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def write_json(self, events: List[Event], filename: str = "events.json") -> str:
        path = os.path.join(self.output_dir, filename)
        payload = {
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            raise RuntimeError(f"Failed to write JSON event log to '{path}': {exc}") from exc
        logger.info("Wrote %d event(s) to %s", len(events), path)
        return path

    def write_csv(self, events: List[Event], filename: str = "events.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EVENT_CSV_FIELDS)
                writer.writeheader()
                for e in events:
                    row = e.to_dict()
                    bbox = row.pop("bbox")
                    row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"] = bbox
                    row.setdefault("duration_seconds", "")
                    writer.writerow(row)
        except OSError as exc:
            raise RuntimeError(f"Failed to write CSV event log to '{path}': {exc}") from exc
        logger.info("Wrote %d event(s) to %s", len(events), path)
        return path
