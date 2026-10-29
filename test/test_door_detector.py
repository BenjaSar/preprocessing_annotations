#!/usr/bin/env python3
"""
T-B0 Door Detector Tests — validate the CubiCasa door tier and its interface
without loading any model (CubiCasa5KDetector is injected as a fake).

Covers:
  - door bboxes extracted and tagged with the CubiCasa tier + type metadata
  - non-door icons (window/toilet/sink) are filtered out
  - pre-computed icons path runs no inference (no detector needed)
  - graceful empty result when no door class / no detector is available
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add the package dir (parent of test/) so `door_detector` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.detection.door_detector import DoorDetector, DoorDetection, DoorDetectionTier


@dataclass
class _FakeIconMask:
    """Mimics cubicasa5k_detector.IconMask (only .bboxes is used here)."""
    bboxes: List[Tuple[float, float, float, float]]


class _FakeCubiCasa:
    """Fake CubiCasa5KDetector: returns a fixed icon dict, counts inference calls."""

    def __init__(self, icons: Dict[str, Any]):
        self._icons = icons
        self.calls = 0

    def detect_icons(self, image_array, threshold: float = 0.3) -> Dict[str, Any]:
        self.calls += 1
        return self._icons


def test_extracts_doors_and_filters_non_doors():
    icons = {
        "door": _FakeIconMask(bboxes=[(10, 10, 40, 60), (100, 20, 130, 70)]),
        "window": _FakeIconMask(bboxes=[(5, 5, 8, 30)]),
        "toilet": _FakeIconMask(bboxes=[(200, 200, 230, 240)]),
    }
    fake = _FakeCubiCasa(icons)
    detector = DoorDetector(detector=fake)

    doors = detector.detect_doors(image_array=np.zeros((300, 300, 3), dtype=np.uint8))

    assert len(doors) == 2, f"expected 2 doors, got {len(doors)}"
    assert all(isinstance(d, DoorDetection) for d in doors)
    assert all(d.source_tier == DoorDetectionTier.CUBICASA5K for d in doors)
    assert all(d.metadata == {"type": "door"} for d in doors)
    assert doors[0].bbox == (10, 10, 40, 60)
    print("OK extracts doors, filters window/toilet")


def test_precomputed_icons_run_no_inference():
    icons = {"door": _FakeIconMask(bboxes=[(1, 1, 2, 2)])}
    fake = _FakeCubiCasa({"door": _FakeIconMask(bboxes=[])})  # would give 0 if called
    detector = DoorDetector(detector=fake)

    doors = detector.detect_doors(icons=icons)

    assert len(doors) == 1, "pre-computed icons should be used verbatim"
    assert fake.calls == 0, "detect_icons must not run when icons are supplied"
    print("OK pre-computed icons path skips inference")


def test_no_door_class_returns_empty():
    fake = _FakeCubiCasa({"window": _FakeIconMask(bboxes=[(0, 0, 1, 1)])})
    detector = DoorDetector(detector=fake)
    assert detector.detect_doors(image_array=np.zeros((4, 4, 3), np.uint8)) == []
    print("OK no door class -> empty")


def test_no_detector_and_no_image_returns_empty():
    # No injected detector, no image, no icons -> nothing to do, no crash.
    detector = DoorDetector(detector=None)
    assert detector.detect_doors() == []
    print("OK no detector/image -> empty")


class _FakeVLMBackend:
    """Fake VLM backend: returns fixed door dicts (pixel bbox), like detect_doors."""

    def __init__(self, doors: List[dict]):
        self._doors = doors

    def detect_doors(self, image_path):
        return self._doors


def test_vlm_tier_union_with_cubicasa():
    cubi_icons = {"door": _FakeIconMask(bboxes=[(10, 10, 40, 60)])}
    fake_cubi = _FakeCubiCasa(cubi_icons)
    fake_vlm = _FakeVLMBackend(
        [{"bbox": [100, 100, 140, 160], "type": "swing", "confidence": 0.8}]
    )
    detector = DoorDetector(detector=fake_cubi)

    # gate off -> CubiCasa only
    only_cubi = detector.detect_doors(
        icons=cubi_icons, vlm_backend=fake_vlm, use_vlm_tier=False
    )
    assert len(only_cubi) == 1 and only_cubi[0].source_tier == DoorDetectionTier.CUBICASA5K

    # gate on -> union of both tiers, each tagged by source
    # (img_path must resolve to a real file; the fake backend ignores its content)
    both = detector.detect_doors(
        icons=cubi_icons, img_path=Path(__file__), vlm_backend=fake_vlm, use_vlm_tier=True
    )
    tiers = sorted(d.source_tier.value for d in both)
    assert len(both) == 2, f"expected union of 2, got {len(both)}"
    assert tiers == ["cubicasa5k", "vlm_prompt"], tiers
    vlm_door = [d for d in both if d.source_tier == DoorDetectionTier.VLM_PROMPT][0]
    assert vlm_door.bbox == (100.0, 100.0, 140.0, 160.0)
    assert vlm_door.metadata == {"type": "swing"}
    print("OK VLM tier: gate off=CubiCasa only, gate on=union tagged by source")


def test_vlm_tier_rejects_degenerate_bbox():
    fake_vlm = _FakeVLMBackend(
        [{"bbox": [50, 50, 50, 60]}, {"bbox": [1, 2, 3]}]  # zero-width, wrong length
    )
    detector = DoorDetector(detector=None)
    doors = detector.detect_doors_from_vlm_prompt(fake_vlm, img_path=Path(__file__))
    assert doors == [], "degenerate/malformed bboxes must be dropped"
    print("OK VLM tier drops degenerate/malformed bboxes")


if __name__ == "__main__":
    test_extracts_doors_and_filters_non_doors()
    test_precomputed_icons_run_no_inference()
    test_no_door_class_returns_empty()
    test_no_detector_and_no_image_returns_empty()
    test_vlm_tier_union_with_cubicasa()
    test_vlm_tier_rejects_degenerate_bbox()
    print("\nALL DOOR DETECTOR TESTS PASSED")
