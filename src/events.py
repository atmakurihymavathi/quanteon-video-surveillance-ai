"""
events.py
---------
Per-track, per-zone state machines that turn raw (track, zone) membership
into deduplicated, timestamped events: zone_intrusion and loitering.

State machine (per track_id, per zone name)
--------------------------------------------
    OUTSIDE --(bbox enters zone)--> INSIDE
    INSIDE  --(bbox leaves zone, or track lost > grace period)--> OUTSIDE

An 'intrusion' event fires exactly once on the OUTSIDE -> INSIDE
transition (not every frame the person remains inside).

While INSIDE, residence time accumulates. A 'loitering' event fires once
residence time first exceeds the zone's configured threshold. To avoid
spamming one event per frame thereafter, loitering re-fires only every
`loitering_refire_seconds` (configurable) while the person remains, which
keeps a security operator informed of a still-ongoing loiter without
flooding the log.

Brief detection loss (occlusion, missed frame) does not immediately reset
INSIDE state: a small grace period (in frames) tolerates short gaps so a
person is not incorrectly considered to have left and re-entered.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .utils import BBox, frame_to_timestamp

logger = logging.getLogger("surveillance")

_event_id_counter = itertools.count(1)


def _next_event_id(prefix: str) -> str:
    return f"{prefix}-{next(_event_id_counter):06d}"


class ZoneState(Enum):
    OUTSIDE = "outside"
    INSIDE = "inside"


@dataclass
class Event:
    event_id: str
    event_type: str
    track_id: int
    zone: str
    frame_number: int
    timestamp: str
    confidence: float
    bbox: BBox
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "zone": self.zone,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
        }
        if self.duration_seconds is not None:
            d["duration_seconds"] = round(self.duration_seconds, 2)
        return d


@dataclass
class _TrackZoneState:
    state: ZoneState = ZoneState.OUTSIDE
    entered_frame: Optional[int] = None
    frames_missing: int = 0
    loitering_triggered: bool = False
    last_loiter_refire_frame: Optional[int] = None


class EventManager:
    """Tracks per-(track_id, zone) state and emits deduplicated events.

    Parameters
    ----------
    fps:
        Video frame rate, used to convert frame counts into seconds and
        timestamps.
    exit_grace_frames:
        Number of consecutive frames a track may be absent from a zone
        (due to a missed detection, brief occlusion, etc.) before it is
        considered to have actually left. Prevents flapping intrusion/
        loitering events from single-frame detection noise.
    loitering_refire_seconds:
        Minimum seconds between repeated loitering events for the same
        ongoing loiter, so operators get periodic updates without a
        flood of near-duplicate events.
    """

    def __init__(
        self,
        fps: float,
        exit_grace_frames: int = 5,
        loitering_refire_seconds: float = 15.0,
    ) -> None:
        self.fps = max(fps, 1e-3)
        self.exit_grace_frames = exit_grace_frames
        self.loitering_refire_seconds = loitering_refire_seconds
        self._states: Dict[Tuple[int, str], _TrackZoneState] = {}
        self.events: List[Event] = []

    def process_frame(
        self,
        frame_number: int,
        tracks_in_zones: List[Tuple[int, BBox, float, List[str]]],
        zone_lookup: Dict[str, "object"],
    ) -> List[Event]:
        """Update state for all tracks in this frame and return new events.

        Parameters
        ----------
        tracks_in_zones:
            List of (track_id, bbox, confidence, [zone_names_bbox_is_in]).
            Must include every *currently confirmed* track, even those in
            zero zones, so we can correctly detect zone-exit transitions.
        zone_lookup:
            Mapping of zone name -> Zone object (for intrusion/loitering
            enabled flags and thresholds).
        """
        new_events: List[Event] = []
        seen_pairs = set()

        for track_id, bbox, confidence, zone_names in tracks_in_zones:
            for zone_name in zone_names:
                zone = zone_lookup.get(zone_name)
                if zone is None:
                    continue
                key = (track_id, zone_name)
                seen_pairs.add(key)
                st = self._states.setdefault(key, _TrackZoneState())
                st.frames_missing = 0

                if st.state == ZoneState.OUTSIDE:
                    st.state = ZoneState.INSIDE
                    st.entered_frame = frame_number
                    st.loitering_triggered = False
                    st.last_loiter_refire_frame = None
                    if getattr(zone, "intrusion_enabled", True):
                        new_events.append(
                            self._make_intrusion_event(
                                track_id, zone_name, frame_number, confidence, bbox
                            )
                        )
                else:
                    if getattr(zone, "loitering_enabled", True) and st.entered_frame is not None:
                        duration = (frame_number - st.entered_frame) / self.fps
                        threshold = getattr(zone, "loitering_seconds", 10.0)
                        if duration >= threshold:
                            should_fire = not st.loitering_triggered
                            if not should_fire and st.last_loiter_refire_frame is not None:
                                since_refire = (
                                    frame_number - st.last_loiter_refire_frame
                                ) / self.fps
                                should_fire = since_refire >= self.loitering_refire_seconds
                            if should_fire:
                                st.loitering_triggered = True
                                st.last_loiter_refire_frame = frame_number
                                new_events.append(
                                    self._make_loitering_event(
                                        track_id,
                                        zone_name,
                                        frame_number,
                                        confidence,
                                        bbox,
                                        duration,
                                    )
                                )

        # Any (track, zone) pair that was INSIDE but not observed this frame
        # either left the zone or its track disappeared -- apply the grace
        # period before resetting to OUTSIDE.
        for key, st in list(self._states.items()):
            if key in seen_pairs or st.state == ZoneState.OUTSIDE:
                continue
            st.frames_missing += 1
            if st.frames_missing > self.exit_grace_frames:
                st.state = ZoneState.OUTSIDE
                st.entered_frame = None
                st.loitering_triggered = False
                st.last_loiter_refire_frame = None

        self.events.extend(new_events)
        return new_events

    def handle_lost_tracks(self, active_track_ids: List[int]) -> None:
        """Purge state for tracks the tracker has permanently dropped.

        Called once per frame with the full set of currently-known track
        IDs (confirmed or not) so we don't leak state for tracks that will
        never come back.
        """
        active = set(active_track_ids)
        stale_keys = [k for k in self._states if k[0] not in active]
        for k in stale_keys:
            del self._states[k]

    def _make_intrusion_event(
        self, track_id: int, zone: str, frame_number: int, confidence: float, bbox: BBox
    ) -> Event:
        return Event(
            event_id=_next_event_id("evt"),
            event_type="zone_intrusion",
            track_id=track_id,
            zone=zone,
            frame_number=frame_number,
            timestamp=frame_to_timestamp(frame_number, self.fps),
            confidence=confidence,
            bbox=bbox,
        )

    def _make_loitering_event(
        self,
        track_id: int,
        zone: str,
        frame_number: int,
        confidence: float,
        bbox: BBox,
        duration: float,
    ) -> Event:
        return Event(
            event_id=_next_event_id("evt"),
            event_type="loitering",
            track_id=track_id,
            zone=zone,
            frame_number=frame_number,
            timestamp=frame_to_timestamp(frame_number, self.fps),
            confidence=confidence,
            bbox=bbox,
            duration_seconds=duration,
        )
