"""Unit tests for src.zone_manager: point-in-polygon and zone membership."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.zone_manager import Zone, ZoneConfigError, ZoneManager


SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def make_square_zone(**overrides):
    defaults = dict(name="test_zone", polygon=SQUARE)
    defaults.update(overrides)
    return Zone(**defaults)


class TestPointInPolygon:
    def test_point_clearly_inside(self):
        zone = make_square_zone()
        assert zone.contains_point((200, 200)) is True

    def test_point_clearly_outside(self):
        zone = make_square_zone()
        assert zone.contains_point((50, 50)) is False
        assert zone.contains_point((400, 400)) is False

    def test_point_on_boundary_is_inside(self):
        # Boundary points are treated as inside (conservative for security use).
        zone = make_square_zone()
        assert zone.contains_point((100, 200)) is True  # on left edge
        assert zone.contains_point((300, 300)) is True  # on corner

    def test_point_just_outside_boundary(self):
        zone = make_square_zone()
        assert zone.contains_point((99, 200)) is False

    def test_concave_polygon(self):
        # An L-shaped (concave) polygon: point in the "notch" must be outside.
        l_shape = [(0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)]
        zone = Zone(name="l_zone", polygon=l_shape)
        assert zone.contains_point((150, 150)) is False  # in the cut-out notch
        assert zone.contains_point((50, 50)) is True  # in the solid part


class TestBBoxMembership:
    def test_bbox_bottom_center_used_for_membership(self):
        zone = make_square_zone()
        # A tall box whose TOP is inside the zone but whose bottom-center
        # (feet) is outside must be considered OUTSIDE the zone.
        bbox_feet_outside = (150, 50, 250, 90)  # bottom y=90 is above zone top (100)
        assert zone.contains_bbox(bbox_feet_outside) is False

        bbox_feet_inside = (150, 50, 250, 150)  # bottom y=150 is inside zone
        assert zone.contains_bbox(bbox_feet_inside) is True

    def test_zones_containing_multiple_zones(self):
        z1 = Zone(name="a", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        z2 = Zone(name="b", polygon=[(50, 50), (200, 50), (200, 200), (50, 200)])
        zm = ZoneManager([z1, z2])
        # Point in the overlap of both zones.
        overlap_bbox = (60, 30, 80, 70)
        found = {z.name for z in zm.zones_containing(overlap_bbox)}
        assert found == {"a", "b"}


class TestZoneValidation:
    def test_polygon_too_few_points_raises(self):
        with pytest.raises(ZoneConfigError):
            Zone(name="bad", polygon=[(0, 0), (10, 10)])

    def test_empty_name_raises(self):
        with pytest.raises(ZoneConfigError):
            Zone(name="", polygon=SQUARE)

    def test_zero_area_polygon_raises(self):
        # All three points collinear -> zero area.
        with pytest.raises(ZoneConfigError):
            Zone(name="degenerate", polygon=[(0, 0), (50, 0), (100, 0)])

    def test_invalid_point_shape_raises(self):
        with pytest.raises(ZoneConfigError):
            Zone(name="bad_point", polygon=[(0, 0), (10, 10, 10), (0, 10)])

    def test_negative_loitering_seconds_raises(self):
        with pytest.raises(ZoneConfigError):
            make_square_zone(loitering_seconds=-1)

    def test_duplicate_zone_names_raise(self):
        z1 = make_square_zone(name="dup")
        z2 = Zone(name="dup", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
        with pytest.raises(ZoneConfigError):
            ZoneManager([z1, z2])


class TestZoneManagerFromDict:
    def test_from_dict_basic(self):
        config = {
            "zones": [
                {"name": "z1", "polygon": SQUARE, "loitering_seconds": 5}
            ]
        }
        zm = ZoneManager.from_dict(config)
        assert len(zm.zones) == 1
        assert zm.get("z1").loitering_seconds == 5

    def test_missing_zones_key_raises(self):
        with pytest.raises(ZoneConfigError):
            ZoneManager.from_dict({"not_zones": []})

    def test_missing_required_fields_raises(self):
        with pytest.raises(ZoneConfigError):
            ZoneManager.from_dict({"zones": [{"name": "z1"}]})  # missing polygon

    def test_empty_zones_list_is_allowed(self):
        # An empty zone list should not raise -- pipeline should still run
        # (detection/tracking only, no zone events).
        zm = ZoneManager.from_dict({"zones": []})
        assert zm.zones == []

    def test_unknown_zone_lookup_raises_keyerror(self):
        zm = ZoneManager.from_dict({"zones": [{"name": "z1", "polygon": SQUARE}]})
        with pytest.raises(KeyError):
            zm.get("does_not_exist")
