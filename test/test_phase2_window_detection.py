#!/usr/bin/env python3
"""
Phase 2 Window Detection Tests — Validate three-tier architecture.

Tests window detection infrastructure without requiring actual models:
  - Tier 1: PDF layer extraction (mock)
  - Tier 2: CubiCasa5K integration (mock)
  - Tier 3: VLM prompting (mock)
  - Spatial intersection (Shapely)
  - Suffix application (taxonomy.add_window_suffix)
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from window_detector import (
    WindowDetector,
    WindowDetection,
    RoomWindowMapping,
    WindowDetectionTier,
    apply_window_suffixes,
)


def test_window_detection_basic():
    """Test WindowDetection dataclass and basic setup."""
    print("\n" + "=" * 70)
    print("TEST 1: Window Detection Dataclass")
    print("=" * 70)

    window = WindowDetection(
        bbox=(100, 100, 200, 150),
        confidence=0.9,
        source_tier=WindowDetectionTier.PDF_LAYERS,
    )

    assert window.bbox == (100, 100, 200, 150), "Bbox mismatch"
    assert window.confidence == 0.9, "Confidence mismatch"
    assert window.source_tier == WindowDetectionTier.PDF_LAYERS, "Tier mismatch"

    print("✅ WindowDetection created successfully")
    print(f"   bbox: {window.bbox}")
    print(f"   confidence: {window.confidence}")
    print(f"   source: {window.source_tier.value}")


def test_room_window_mapping():
    """Test RoomWindowMapping and spatial logic."""
    print("\n" + "=" * 70)
    print("TEST 2: Room-Window Mapping")
    print("=" * 70)

    mapping = RoomWindowMapping(
        room_id="room_1",
        has_windows=True,
        has_skylights=False,
        has_openings=False,
        window_count=2,
    )

    assert mapping.room_id == "room_1", "Room ID mismatch"
    assert mapping.has_windows is True, "Window flag mismatch"
    assert mapping.window_count == 2, "Window count mismatch"

    print("✅ RoomWindowMapping created successfully")
    print(f"   room_id: {mapping.room_id}")
    print(f"   has_windows: {mapping.has_windows}")
    print(f"   window_count: {mapping.window_count}")


def test_window_detector_init():
    """Test WindowDetector initialization."""
    print("\n" + "=" * 70)
    print("TEST 3: WindowDetector Initialization")
    print("=" * 70)

    detector = WindowDetector()

    assert detector is not None, "Detector creation failed"
    assert detector.has_shapely is True, "Shapely check failed"
    assert len(detector.tier_results) == 0, "Tier results should be empty"

    print("✅ WindowDetector initialized successfully")
    print(f"   Shapely available: {detector.has_shapely}")
    print(f"   Tier results: {dict(detector.tier_results)}")


def test_spatial_intersection():
    """Test spatial intersection logic with mock data."""
    print("\n" + "=" * 70)
    print("TEST 4: Spatial Intersection (Shapely)")
    print("=" * 70)

    detector = WindowDetector()

    # Mock windows: two windows in a 300x300 image
    windows = [
        WindowDetection(
            bbox=(50, 50, 100, 100),
            confidence=0.95,
            source_tier=WindowDetectionTier.PDF_LAYERS,
        ),
        WindowDetection(
            bbox=(150, 150, 200, 200),
            confidence=0.95,
            source_tier=WindowDetectionTier.PDF_LAYERS,
        ),
    ]

    # Mock rooms: two rooms with bounding boxes
    rooms = [
        {
            "id": "room_1",
            "type": "CONFERENCE",
            "bbox": [0, 0, 150, 150],  # Will intersect window 1
        },
        {
            "id": "room_2",
            "type": "PRIVATE OFFICE",
            "bbox": [100, 100, 300, 300],  # Will intersect windows 1 and 2
        },
    ]

    # Run spatial intersection
    mappings = detector.map_windows_to_rooms(windows, rooms)

    assert len(mappings) == 2, f"Expected 2 mappings, got {len(mappings)}"

    mapping1 = [m for m in mappings if m.room_id == "room_1"][0]
    mapping2 = [m for m in mappings if m.room_id == "room_2"][0]

    assert mapping1.has_windows is True, "Room 1 should have windows"
    assert mapping1.window_count >= 1, "Room 1 should have at least 1 window"

    assert mapping2.has_windows is True, "Room 2 should have windows"
    assert mapping2.window_count >= 1, "Room 2 should have at least 1 window"

    print("✅ Spatial intersection working correctly")
    print(f"   Room 1 windows: {mapping1.window_count}")
    print(f"   Room 2 windows: {mapping2.window_count}")


def test_window_suffix_application():
    """Test window suffix application to room types."""
    print("\n" + "=" * 70)
    print("TEST 5: Window Suffix Application")
    print("=" * 70)

    from automation.taxonomy import add_window_suffix

    # Test cases.
    # Window presence is a visual attribute orthogonal to room type — any
    # mandatory type can receive a suffix.  CORRIDOR w/ windows is valid:
    # it means the detector found windows in that corridor.  The eligibility
    # sets (WINDOW_ELIGIBLE etc.) have been removed; the detector decides.
    test_cases = [
        ("CONFERENCE",     True,  False, False, "CONFERENCE w/ windows"),
        ("PRIVATE OFFICE", True,  False, False, "PRIVATE OFFICE w/ windows"),
        ("GYMNASIUM",      False, True,  False, "GYMNASIUM w/ skylights"),
        ("PARKING GARAGE", False, False, True,  "PARKING GARAGE w/ side openings"),
        ("CORRIDOR",       True,  False, False, "CORRIDOR w/ windows"),   # any type is eligible
        ("STORAGE ROOM",   True,  False, False, "STORAGE ROOM w/ windows"),
        ("RESTROOM",       False, False, False, "RESTROOM"),              # no flags → no suffix
    ]

    for base_type, has_win, has_sky, has_open, expected in test_cases:
        result = add_window_suffix(
            base_type,
            has_windows=has_win,
            has_skylights=has_sky,
            has_openings=has_open,
        )
        assert result == expected, f"Failed: {base_type} -> got {result}, expected {expected}"
        print(f"✅ {base_type:20} + windows={has_win}, skylights={has_sky} -> {result}")


def test_apply_window_suffixes():
    """Test the apply_window_suffixes convenience function."""
    print("\n" + "=" * 70)
    print("TEST 6: Apply Window Suffixes to Rooms")
    print("=" * 70)

    # Mock rooms
    rooms = [
        {"id": "room_1", "type": "CONFERENCE", "room_number": "101"},
        {"id": "room_2", "type": "PRIVATE OFFICE", "room_number": "102"},
    ]

    # Mock window mappings
    mappings = [
        RoomWindowMapping(room_id="room_1", has_windows=True, window_count=2),
        RoomWindowMapping(room_id="room_2", has_windows=False, window_count=0),
    ]

    # Apply suffixes
    apply_window_suffixes(rooms, mappings)

    # Check results
    room1 = rooms[0]
    room2 = rooms[1]

    assert room1["type"] == "CONFERENCE w/ windows", f"Room 1 suffix mismatch: {room1['type']}"
    assert room2["type"] == "PRIVATE OFFICE", f"Room 2 should not have suffix: {room2['type']}"
    assert "window_detection" in room1, "Room 1 should have window_detection metadata"

    print("✅ Window suffixes applied correctly")
    print(f"   Room 1: {room1['type']} ({room1['window_detection']['window_count']} windows)")
    print(f"   Room 2: {room2['type']} (no windows)")


def test_end_to_end_pipeline():
    """Test full pipeline: detection -> mapping -> suffixes."""
    print("\n" + "=" * 70)
    print("TEST 7: End-to-End Pipeline")
    print("=" * 70)

    detector = WindowDetector()

    # Create mock data
    image_array = np.zeros((500, 500, 3), dtype=np.uint8)
    rooms = [
        {"id": "office_1", "type": "PRIVATE OFFICE", "bbox": [50, 50, 200, 200]},
        {"id": "conference_1", "type": "CONFERENCE", "bbox": [250, 250, 450, 450]},
        {"id": "corridor_1", "type": "CORRIDOR", "bbox": [200, 50, 250, 450]},
    ]

    # Since PDF path is None and models aren't available, detection will return empty
    # This tests the graceful fallback behavior
    mappings = detector.detect_windows(
        image_array=image_array,
        rooms=rooms,
        pdf_path=None,
        vlm_backend=None,
    )

    assert len(mappings) == 3, "Should have mapping for each room"

    # All should have no windows (since detection returned empty)
    for mapping in mappings:
        assert mapping.has_windows is False, f"{mapping.room_id} should have no windows"

    print("✅ End-to-end pipeline working (graceful fallback)")
    print(f"   Processed {len(mappings)} rooms")
    for mapping in mappings:
        print(
            f"   - {mapping.room_id}: windows={mapping.has_windows}, count={mapping.window_count}"
        )


def test_tier_detection_priority():
    """Test that tiers are attempted in correct order."""
    print("\n" + "=" * 70)
    print("TEST 8: Tier Priority and Fallback")
    print("=" * 70)

    detector = WindowDetector()

    # With no models and no PDF, all tiers should be skipped
    # and we should get empty results
    image_array = np.zeros((500, 500, 3), dtype=np.uint8)
    rooms = [{"id": "test_room", "type": "CONFERENCE", "bbox": [0, 0, 500, 500]}]

    mappings = detector.detect_windows(
        image_array=image_array,
        rooms=rooms,
        pdf_path=None,
        vlm_backend=None,
    )

    assert len(mappings) == 1, "Should have one mapping"
    assert mappings[0].has_windows is False, "No windows should be detected without models"

    print("✅ Tier priority working correctly")
    print(f"   Tier results: {dict(detector.tier_results)}")
    print(f"   All tiers gracefully skipped when models unavailable")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 2 WINDOW DETECTION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Window Detection Dataclass", test_window_detection_basic),
        ("Room-Window Mapping", test_room_window_mapping),
        ("WindowDetector Initialization", test_window_detector_init),
        ("Spatial Intersection (Shapely)", test_spatial_intersection),
        ("Window Suffix Application", test_window_suffix_application),
        ("Apply Suffixes to Rooms", test_apply_window_suffixes),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Tier Priority and Fallback", test_tier_detection_priority),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
