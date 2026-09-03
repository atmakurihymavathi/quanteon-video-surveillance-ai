"""Unit tests for src.events: intrusion state transitions, loitering timing,
and event deduplication."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.events import EventManager
from src.zone_manager import Zone


def make_zone(name="zone_a", loitering_seconds=2.0, intrusion_enabled=True, loitering_enabled=True):
    return Zone(
        name=name,
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        intrusion_enabled=intrusion_enabled,
        loitering_enabled=loitering_enabled,
        loitering_seconds=loitering_seconds,
    )


BBOX = (10, 10, 30, 30)


class TestIntrusionEvents:
    def test_intrusion_fires_once_on_entry(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone()
        lookup = {zone.name: zone}

        # Track 1 enters zone_a on frame 1.
        events = em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)
        assert len(events) == 1
        assert events[0].event_type == "zone_intrusion"
        assert events[0].track_id == 1

    def test_intrusion_does_not_refire_every_frame(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(loitering_seconds=1000)  # effectively disable loitering firing
        lookup = {zone.name: zone}

        total_new_events = 0
        for frame in range(1, 21):  # 20 frames, still inside
            events = em.process_frame(frame, [(1, BBOX, 0.9, ["zone_a"])], lookup)
            total_new_events += len(events)
        assert total_new_events == 1  # only the initial intrusion event

    def test_intrusion_refires_after_exit_and_reentry(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(loitering_seconds=1000)
        lookup = {zone.name: zone}

        em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)  # enter -> 1 event
        # Leave for longer than the exit grace period (default 5 frames).
        for frame in range(2, 10):
            em.process_frame(frame, [(1, BBOX, 0.9, [])], lookup)
        events = em.process_frame(10, [(1, BBOX, 0.9, ["zone_a"])], lookup)  # re-enter
        assert len(events) == 1
        assert events[0].event_type == "zone_intrusion"

    def test_disabled_intrusion_produces_no_event(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(intrusion_enabled=False, loitering_enabled=False)
        lookup = {zone.name: zone}
        events = em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)
        assert events == []

    def test_brief_detection_loss_does_not_reset_state(self):
        """A single missed frame (occlusion/detector miss) should not be
        treated as the person leaving the zone (grace period)."""
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(loitering_seconds=1000)
        lookup = {zone.name: zone}

        em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)  # enter
        em.process_frame(2, [], lookup)  # missed detection this frame
        events = em.process_frame(3, [(1, BBOX, 0.9, ["zone_a"])], lookup)  # back
        # Should NOT be treated as a new intrusion (still within grace period).
        assert events == []


class TestLoiteringEvents:
    def test_loitering_fires_after_threshold(self):
        fps = 10.0  # 10 frames = 1 second
        em = EventManager(fps=fps, loitering_refire_seconds=1000)
        zone = make_zone(loitering_seconds=1.0)
        lookup = {zone.name: zone}

        events_all = []
        for frame in range(1, 16):  # 1.5 seconds of dwell time
            events_all.extend(em.process_frame(frame, [(1, BBOX, 0.9, ["zone_a"])], lookup))

        event_types = [e.event_type for e in events_all]
        assert "zone_intrusion" in event_types
        assert "loitering" in event_types
        loiter_events = [e for e in events_all if e.event_type == "loitering"]
        assert len(loiter_events) == 1  # dedup: only fires once given long refire interval
        assert loiter_events[0].duration_seconds >= 1.0

    def test_loitering_does_not_fire_before_threshold(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(loitering_seconds=5.0)
        lookup = {zone.name: zone}

        events_all = []
        for frame in range(1, 21):  # only 2 seconds of dwell, threshold is 5
            events_all.extend(em.process_frame(frame, [(1, BBOX, 0.9, ["zone_a"])], lookup))
        assert all(e.event_type != "loitering" for e in events_all)

    def test_loitering_refires_periodically(self):
        fps = 10.0
        em = EventManager(fps=fps, loitering_refire_seconds=1.0)
        zone = make_zone(loitering_seconds=0.5)
        lookup = {zone.name: zone}

        events_all = []
        for frame in range(1, 41):  # 4 seconds total dwell
            events_all.extend(em.process_frame(frame, [(1, BBOX, 0.9, ["zone_a"])], lookup))
        loiter_events = [e for e in events_all if e.event_type == "loitering"]
        # Should fire multiple times (roughly every ~1s after the initial threshold), not once.
        assert len(loiter_events) >= 2

    def test_loitering_disabled_flag_respected(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone(loitering_seconds=0.1, loitering_enabled=False)
        lookup = {zone.name: zone}
        events_all = []
        for frame in range(1, 21):
            events_all.extend(em.process_frame(frame, [(1, BBOX, 0.9, ["zone_a"])], lookup))
        assert all(e.event_type != "loitering" for e in events_all)


class TestEventIdsAndDedup:
    def test_event_ids_are_unique(self):
        fps = 10.0
        em = EventManager(fps=fps, loitering_refire_seconds=1000)
        zone = make_zone(loitering_seconds=100)
        lookup = {zone.name: zone}
        # Two different tracks entering the same zone should each get a unique event id.
        e1 = em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)
        e2 = em.process_frame(1, [(2, BBOX, 0.9, ["zone_a"])], lookup)
        assert e1[0].event_id != e2[0].event_id

    def test_multiple_people_multiple_zones_independent_state(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone_a = make_zone(name="zone_a", loitering_seconds=1000)
        zone_b = make_zone(name="zone_b", loitering_seconds=1000)
        lookup = {zone_a.name: zone_a, zone_b.name: zone_b}

        events = em.process_frame(
            1,
            [
                (1, BBOX, 0.9, ["zone_a"]),
                (2, BBOX, 0.9, ["zone_b"]),
                (3, BBOX, 0.9, ["zone_a", "zone_b"]),
            ],
            lookup,
        )
        # Track 1 -> 1 event, track 2 -> 1 event, track 3 -> 2 events (both zones).
        assert len(events) == 4

    def test_handle_lost_tracks_purges_state(self):
        fps = 10.0
        em = EventManager(fps=fps, loitering_refire_seconds=1000)
        zone = make_zone(loitering_seconds=1000)
        lookup = {zone.name: zone}
        em.process_frame(1, [(1, BBOX, 0.9, ["zone_a"])], lookup)
        assert (1, "zone_a") in em._states
        em.handle_lost_tracks(active_track_ids=[])  # track 1 is gone for good
        assert (1, "zone_a") not in em._states


class TestEmptyFrames:
    def test_empty_frame_produces_no_events_and_no_crash(self):
        fps = 10.0
        em = EventManager(fps=fps)
        zone = make_zone()
        lookup = {zone.name: zone}
        events = em.process_frame(1, [], lookup)
        assert events == []
