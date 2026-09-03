"""
zone_manager.py
----------------
Loads polygon-based zone definitions from JSON and provides robust
point-in-polygon membership testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from .utils import BBox, bbox_bottom_center, load_json_config

logger = logging.getLogger("surveillance")


class ZoneConfigError(ValueError):
    """Raised for malformed or invalid zone configuration."""


@dataclass
class Zone:
    """A single named polygon zone with per-zone feature flags."""

    name: str
    polygon: List[Tuple[float, float]]
    intrusion_enabled: bool = True
    loitering_enabled: bool = True
    loitering_seconds: float = 10.0
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 255))

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ZoneConfigError(f"Zone name must be a non-empty string, got: {self.name!r}")
        if len(self.polygon) < 3:
            raise ZoneConfigError(
                f"Zone '{self.name}' polygon must have at least 3 points "
                f"(got {len(self.polygon)}). A zone needs at least a triangle."
            )
        for pt in self.polygon:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                raise ZoneConfigError(
                    f"Zone '{self.name}' has an invalid point {pt!r}; expected [x, y]."
                )
        if self.loitering_seconds <= 0:
            raise ZoneConfigError(
                f"Zone '{self.name}' loitering_seconds must be > 0, got {self.loitering_seconds}."
            )

        # Degenerate polygon check: zero area means every membership test
        # would be meaningless, so fail fast with a clear message instead
        # of silently never triggering events.
        area = cv2.contourArea(np.array(self.polygon, dtype=np.float32))
        if area <= 0:
            raise ZoneConfigError(
                f"Zone '{self.name}' polygon has zero or negative area "
                "(points may be collinear or duplicated). Check the coordinates."
            )

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Return True if `point` lies inside or on the boundary of the zone.

        Uses cv2.pointPolygonTest, which correctly supports both convex and
        concave polygons. Points exactly on the boundary (return value 0)
        are treated as *inside* the zone -- a conservative choice for a
        security system, where a person straddling the boundary should be
        flagged rather than missed.
        """
        contour = np.array(self.polygon, dtype=np.float32)
        result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        return result >= 0

    def contains_bbox(self, bbox: BBox) -> bool:
        """Zone membership test for a detection box using its bottom-center point."""
        return self.contains_point(bbox_bottom_center(bbox))

    def as_np(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)


class ZoneManager:
    """Loads and holds all configured zones."""

    def __init__(self, zones: List[Zone]) -> None:
        if not zones:
            logger.warning(
                "No zones configured; the pipeline will run detection/tracking "
                "only, without zone intrusion or loitering event generation."
            )
        names = [z.name for z in zones]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ZoneConfigError(f"Duplicate zone name(s) found: {sorted(duplicates)}")
        self.zones = zones

    @classmethod
    def from_file(cls, path: str) -> "ZoneManager":
        config = load_json_config(path)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ZoneManager":
        if "zones" not in config:
            raise ZoneConfigError("Zone config must contain a top-level 'zones' key.")
        if not isinstance(config["zones"], list):
            raise ZoneConfigError("'zones' must be a list of zone objects.")

        zones: List[Zone] = []
        # A small, readable color palette cycled across zones for annotation.
        palette = [
            (0, 0, 255), (0, 165, 255), (255, 0, 255),
            (255, 255, 0), (0, 255, 255), (128, 0, 255),
        ]
        for i, raw in enumerate(config["zones"]):
            if not isinstance(raw, dict) or "name" not in raw or "polygon" not in raw:
                raise ZoneConfigError(
                    f"Zone entry #{i} is missing required 'name' or 'polygon' field: {raw!r}"
                )
            zone = Zone(
                name=raw["name"],
                polygon=[tuple(p) for p in raw["polygon"]],
                intrusion_enabled=raw.get("intrusion_enabled", True),
                loitering_enabled=raw.get("loitering_enabled", True),
                loitering_seconds=float(raw.get("loitering_seconds", 10.0)),
                color=tuple(raw.get("color", palette[i % len(palette)])),
            )
            zones.append(zone)
        return cls(zones)

    def zones_containing(self, bbox: BBox) -> List[Zone]:
        """Return all zones a bounding box currently falls within."""
        return [z for z in self.zones if z.contains_bbox(bbox)]

    def get(self, name: str) -> Zone:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(f"No such zone: '{name}'")
