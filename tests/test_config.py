"""Unit tests for config loading/validation (src.utils, src.zone_manager)
and small geometry/formatting helpers."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.utils import ensure_dir, frame_to_timestamp, iou, load_json_config
from src.zone_manager import ZoneConfigError, ZoneManager


class TestLoadJsonConfig:
    def test_missing_file_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_json_config(str(missing))

    def test_invalid_json_raises_valueerror(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not valid json ")
        with pytest.raises(ValueError):
            load_json_config(str(bad))

    def test_valid_json_loads(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"zones": []}))
        config = load_json_config(str(good))
        assert config == {"zones": []}


class TestRealZonesConfigFile:
    def test_bundled_zones_config_is_valid(self):
        """The shipped config/zones.json must load without error -- this
        catches accidental breakage of the example config reviewers rely on."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "config", "zones.json")
        zm = ZoneManager.from_file(path)
        assert len(zm.zones) >= 1
        for zone in zm.zones:
            assert zone.loitering_seconds > 0
            assert len(zone.polygon) >= 3


class TestZoneConfigEdgeCases:
    def test_non_list_zones_raises(self):
        with pytest.raises(ZoneConfigError):
            ZoneManager.from_dict({"zones": "not-a-list"})

    def test_zone_entry_not_a_dict_raises(self):
        with pytest.raises(ZoneConfigError):
            ZoneManager.from_dict({"zones": ["not-a-dict"]})


class TestIoU:
    def test_identical_boxes_iou_is_one(self):
        box = (0, 0, 10, 10)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        # Two 10x10 boxes overlapping in a 5x10 region -> intersection 50,
        # union = 100 + 100 - 50 = 150 -> IoU = 1/3.
        box_a = (0, 0, 10, 10)
        box_b = (5, 0, 15, 10)
        assert iou(box_a, box_b) == pytest.approx(50 / 150)

    def test_degenerate_zero_area_box(self):
        assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


class TestTimestampFormatting:
    def test_zero_frame(self):
        assert frame_to_timestamp(0, 30.0) == "00:00.00"

    def test_matches_assignment_example_format(self):
        # e.g. "00:11.33" style from the assignment's sample event schema.
        result = frame_to_timestamp(340, 30.0)
        assert result.count(":") == 1
        minutes, seconds = result.split(":")
        assert len(minutes) == 2
        assert "." in seconds

    def test_handles_zero_or_invalid_fps_gracefully(self):
        # Must not raise a ZeroDivisionError.
        result = frame_to_timestamp(100, 0)
        assert isinstance(result, str)


class TestEnsureDir:
    def test_creates_nested_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        ensure_dir(str(target))
        assert target.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path):
        ensure_dir(str(tmp_path))  # already exists; should not raise
        ensure_dir(str(tmp_path))
