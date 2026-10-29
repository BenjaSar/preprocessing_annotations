#!/usr/bin/env python3
"""
AV-2 Object Type Registry Tests — validate the registry that replaced
map_detections_to_rooms's inline if/elif type dispatch.

Covers:
  - default registry reproduces the pre-AV-2 dispatch table exactly
  - non-strict flag_for() falls back to has_windows (old `else` branch)
  - strict flag_for() returns None for an unregistered type
  - register() rejects an unknown flag field (fails at registration, not
    silently at mapping time)
  - map_detections_to_rooms default output is unchanged by the refactor
  - map_detections_to_rooms strict mode counts unregistered types instead
    of folding them into has_windows
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from preprocessing_annotations.detection.object_types import (
    ObjectTypeRegistry,
    VALID_FLAG_FIELDS,
)
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
    map_detections_to_rooms,
)


def test_default_registry_matches_pre_av2_dispatch():
    registry = ObjectTypeRegistry()
    expected = {
        "skylight": "has_skylights",
        "side_opening": "has_openings",
        "opening": "has_openings",
        "door": "has_door",
        "toilet": "has_toilet",
        "bathtub": "has_bathtub",
        "sink": "has_sink",
    }
    for type_name, flag_field in expected.items():
        assert registry.flag_for(type_name) == flag_field


def test_non_strict_unknown_falls_back_to_has_windows():
    registry = ObjectTypeRegistry()
    assert registry.flag_for("double_door", strict=False) == "has_windows"
    assert registry.flag_for("wall", strict=False) == "has_windows"


def test_strict_unknown_returns_none():
    registry = ObjectTypeRegistry()
    assert registry.flag_for("double_door", strict=True) is None


def test_register_rejects_unknown_flag_field():
    registry = ObjectTypeRegistry()
    with pytest.raises(ValueError):
        registry.register("wall", "has_wall")  # not in VALID_FLAG_FIELDS


def test_register_new_class_is_used():
    registry = ObjectTypeRegistry()
    registry.register("double_door", "has_door")
    assert registry.flag_for("double_door") == "has_door"
    assert "has_door" in VALID_FLAG_FIELDS


def _room(bbox):
    return {"id": "room_1", "bbox": bbox}


def _window(window_type, bbox=(10, 10, 20, 20)):
    return WindowDetection(
        bbox=bbox, confidence=0.9,
        source_tier=WindowDetectionTier.YOLO,
        metadata={"type": window_type},
    )


def test_map_detections_default_behavior_unchanged():
    """A door + an unregistered class ('double_door') intersecting the
    same room: default (non-strict) dispatch must still set has_door AND
    fold the unregistered class into has_windows -- byte-identical to
    the pre-AV-2 if/elif chain's `else` branch."""
    rooms = [_room([0, 0, 100, 100])]
    windows = [_window("door"), _window("double_door")]

    mappings = map_detections_to_rooms(windows, rooms)

    assert len(mappings) == 1
    m = mappings[0]
    assert m.has_door is True
    assert m.has_windows is True  # unregistered type folded in, old behavior
    assert m.unknown_type_count == 0  # non-strict never counts unknowns
    assert m.window_count == 2


def test_map_detections_strict_counts_unknown_type():
    rooms = [_room([0, 0, 100, 100])]
    windows = [_window("door"), _window("double_door")]

    mappings = map_detections_to_rooms(windows, rooms, strict_type_dispatch=True)

    assert len(mappings) == 1
    m = mappings[0]
    assert m.has_door is True
    assert m.has_windows is False  # not folded in under strict mode
    assert m.unknown_type_count == 1
    assert m.window_count == 2
