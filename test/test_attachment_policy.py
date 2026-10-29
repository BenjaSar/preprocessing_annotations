#!/usr/bin/env python3
"""
AV-3 Attachment Policy Tests — validate the AttachmentPolicy port,
its default (IntersectionAttachmentPolicy), the Prototype-First
alternative (NearestWithCutoffAttachmentPolicy), and the factory.

Covers:
  - IntersectionAttachmentPolicy output is byte-identical to a direct
    map_detections_to_rooms call (the pre-AV-3 behavior)
  - AttachmentPolicyFactory resolves "intersection" (default) and
    "nearest_with_cutoff", and rejects an unknown policy name
  - NearestWithCutoffAttachmentPolicy requires an explicit cutoff
  - NearestWithCutoffAttachmentPolicy: an intersecting detection is
    handled by intersection alone (no double count); a non-intersecting
    detection within cutoff attaches to its nearest room; a
    non-intersecting detection beyond cutoff stays unattached
    (identical to the default policy's behavior for it)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from preprocessing_annotations.config import AttachmentConfig
from preprocessing_annotations.detection.attachment import (
    AttachmentPolicyFactory,
    IntersectionAttachmentPolicy,
    NearestWithCutoffAttachmentPolicy,
)
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
    map_detections_to_rooms,
)


def _room(room_id, bbox):
    return {"id": room_id, "bbox": bbox}


def _detection(window_type, bbox):
    return WindowDetection(
        bbox=bbox, confidence=0.9,
        source_tier=WindowDetectionTier.YOLO,
        metadata={"type": window_type},
    )


def test_intersection_policy_matches_direct_call():
    rooms = [_room("r1", [0, 0, 100, 100])]
    detections = [_detection("door", (10, 10, 20, 20))]

    policy = IntersectionAttachmentPolicy(AttachmentConfig())
    via_policy = policy.attach(detections, rooms)
    via_direct = map_detections_to_rooms(detections, rooms)

    assert len(via_policy) == len(via_direct) == 1
    assert via_policy[0].has_door == via_direct[0].has_door is True
    assert via_policy[0].window_count == via_direct[0].window_count == 1


def test_factory_default_is_intersection():
    policy = AttachmentPolicyFactory.create(AttachmentConfig())
    assert isinstance(policy, IntersectionAttachmentPolicy)


def test_factory_creates_nearest_with_cutoff():
    policy = AttachmentPolicyFactory.create(
        AttachmentConfig(policy="nearest_with_cutoff", nearest_cutoff_px=300)
    )
    assert isinstance(policy, NearestWithCutoffAttachmentPolicy)


def test_factory_rejects_unknown_policy():
    with pytest.raises(ValueError):
        AttachmentPolicyFactory.create(AttachmentConfig(policy="does_not_exist"))


def test_nearest_cutoff_requires_explicit_cutoff():
    with pytest.raises(ValueError):
        NearestWithCutoffAttachmentPolicy(
            AttachmentConfig(policy="nearest_with_cutoff")
        )


def test_nearest_cutoff_attaches_within_range():
    """Detection is outside r1's bbox (no intersection), but its
    centroid is close to r1's centroid and within cutoff -- should
    attach to r1, not r2 (which is far away)."""
    rooms = [
        _room("r1", [0, 0, 100, 100]),
        _room("r2", [5000, 5000, 5100, 5100]),
    ]
    detections = [_detection("window", (150, 150, 160, 160))]

    policy = NearestWithCutoffAttachmentPolicy(
        AttachmentConfig(policy="nearest_with_cutoff", nearest_cutoff_px=300)
    )
    mappings = policy.attach(detections, rooms)
    by_id = {m.room_id: m for m in mappings}

    assert by_id["r1"].has_windows is True
    assert by_id["r1"].window_count == 1
    assert by_id["r2"].has_windows is False
    assert by_id["r2"].window_count == 0


def test_nearest_cutoff_leaves_far_detection_unattached():
    """Same geometry as above but cutoff too small to reach r1 -- must
    behave like the default policy: unattached, not force-attached."""
    rooms = [_room("r1", [0, 0, 100, 100])]
    detections = [_detection("window", (150, 150, 160, 160))]

    policy = NearestWithCutoffAttachmentPolicy(
        AttachmentConfig(policy="nearest_with_cutoff", nearest_cutoff_px=10)
    )
    mappings = policy.attach(detections, rooms)

    assert mappings[0].has_windows is False
    assert mappings[0].window_count == 0


def test_nearest_cutoff_does_not_double_count_intersecting_detection():
    """A detection that already intersects a room must not also be
    counted by the nearest-fallback pass."""
    rooms = [_room("r1", [0, 0, 100, 100])]
    detections = [_detection("door", (10, 10, 20, 20))]

    policy = NearestWithCutoffAttachmentPolicy(
        AttachmentConfig(policy="nearest_with_cutoff", nearest_cutoff_px=1000)
    )
    mappings = policy.attach(detections, rooms)

    assert mappings[0].window_count == 1
    assert mappings[0].has_door is True
