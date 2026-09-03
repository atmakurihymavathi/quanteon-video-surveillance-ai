"""Unit tests for src.tracker: association, ID persistence, occlusion handling."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import Detection
from src.tracker import PersonTracker


def det(x1, y1, x2, y2, conf=0.9):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


class TestBasicTracking:
    def test_single_person_gets_consistent_id(self):
        tracker = PersonTracker(min_hits=1)
        ids_seen = set()
        x = 0
        for _ in range(10):
            tracks = tracker.update([det(x, 0, x + 20, 40)])
            ids_seen.update(t.track_id for t in tracks)
            x += 5
        assert len(ids_seen) == 1  # same ID kept across smooth motion

    def test_new_track_not_confirmed_until_min_hits(self):
        tracker = PersonTracker(min_hits=3)
        tracks = tracker.update([det(0, 0, 20, 40)])
        assert tracks == []  # not confirmed yet (only 1 hit)
        tracks = tracker.update([det(2, 0, 22, 40)])
        assert tracks == []  # 2 hits, still not confirmed
        tracks = tracker.update([det(4, 0, 24, 40)])
        assert len(tracks) == 1  # 3rd hit -> confirmed

    def test_empty_frame_does_not_crash_and_ages_tracks(self):
        tracker = PersonTracker(min_hits=1, max_age=2)
        tracker.update([det(0, 0, 20, 40)])
        tracks = tracker.update([])  # empty frame (no detections)
        assert isinstance(tracks, list)


class TestOcclusionHandling:
    def test_id_persists_through_brief_occlusion(self):
        tracker = PersonTracker(min_hits=1, max_age=5)
        t1 = tracker.update([det(0, 0, 20, 40)])
        original_id = t1[0].track_id

        # Person briefly disappears (occluded) for 2 frames.
        tracker.update([])
        tracker.update([])

        # Reappears close to where it was predicted to be.
        t2 = tracker.update([det(6, 0, 26, 40)])
        assert len(t2) == 1
        assert t2[0].track_id == original_id

    def test_track_dropped_after_max_age_exceeded(self):
        tracker = PersonTracker(min_hits=1, max_age=3)
        tracker.update([det(0, 0, 20, 40)])
        for _ in range(5):  # exceed max_age with no detections
            tracker.update([])
        assert len(tracker.tracks) == 0


class TestMultiplePeople:
    def test_two_people_get_different_ids(self):
        tracker = PersonTracker(min_hits=1)
        tracks = tracker.update([det(0, 0, 20, 40), det(200, 0, 220, 40)])
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_two_people_tracked_independently_over_time(self):
        tracker = PersonTracker(min_hits=1)
        xa, xb = 0, 300
        ids_a, ids_b = set(), set()
        for _ in range(8):
            tracks = tracker.update([det(xa, 0, xa + 20, 40), det(xb, 0, xb + 20, 40)])
            by_x = sorted(tracks, key=lambda t: t.bbox[0])
            ids_a.add(by_x[0].track_id)
            ids_b.add(by_x[1].track_id)
            xa += 5
            xb += 5
        assert len(ids_a) == 1
        assert len(ids_b) == 1
        assert ids_a != ids_b
