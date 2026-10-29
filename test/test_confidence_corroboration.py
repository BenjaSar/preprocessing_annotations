"""T-1 (technology-evaluation-2026-08-28.md, G-1): object-detection
corroboration term in room confidence.

Covers the hard constraints from that plan's risk table:
  - default weight (0.0) is byte-identical to pre-T-1 behavior
  - the term is additive, not a reallocated weighted average -- a room
    with zero evidence is never penalized relative to the term being
    absent entirely
  - _apply_object_attributes never changes room count (A2) and leaves
    rooms with no mapping match completely untouched
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.annotation_schema import ConfidenceComputer
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.detection.window_detector import (
    RoomWindowMapping, WindowDetection, WindowDetectionTier,
    map_detections_to_rooms,
)


def test_compute_default_matches_original_3factor_formula():
    combined, detail = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
    )
    expected = round(0.3 * 0.9 + 0.4 * 0.7 + 0.3 * 0.85, 4)
    assert combined == expected
    assert detail.corroboration == 0.0


def test_zero_corroboration_weight_is_a_noop_even_with_evidence():
    """w_corroboration=0.0 (the shipped default): a room with full
    evidence (corroboration_score=1.0) must score identically to one
    with none -- the weight, not the evidence, gates any effect."""
    with_evidence, _ = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
        corroboration_score=1.0, w_corroboration=0.0,
    )
    without_evidence, _ = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
        corroboration_score=0.0, w_corroboration=0.0,
    )
    assert with_evidence == without_evidence


def test_absence_of_evidence_never_penalizes():
    """Asymmetric requirement: enabling the term (w_corroboration>0) for
    a room with NO evidence must not lower its score vs the term being
    off entirely."""
    baseline, _ = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
    )
    no_evidence_term_on, _ = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
        corroboration_score=0.0, w_corroboration=0.2,
    )
    assert no_evidence_term_on == baseline


def test_positive_evidence_boosts_and_clamps():
    base, _ = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
    )
    boosted, detail = ConfidenceComputer.compute(
        detection_score=0.9, classification_score=0.7, ocr_score=0.85,
        corroboration_score=1.0, w_corroboration=0.2,
    )
    assert boosted > base
    assert boosted <= 1.0
    assert detail.corroboration == 1.0

    # Clamp: a near-1.0 base plus a large bonus must not exceed 1.0.
    clamped, _ = ConfidenceComputer.compute(
        detection_score=1.0, classification_score=1.0, ocr_score=1.0,
        corroboration_score=1.0, w_corroboration=0.5,
    )
    assert clamped == 1.0


def test_apply_corroboration_recomputes_from_stored_detail():
    stored_confidence = round(0.3 * 0.9 + 0.4 * 0.7 + 0.3 * 0.75, 4)
    room = {
        "confidence": stored_confidence,
        "confidence_detail": {
            "detection": 0.9, "classification": 0.7, "ocr": 0.75,
        },
    }
    # Default weights: reproduces the same value (no-op).
    ConfidenceComputer.apply_corroboration(room, corroboration_score=1.0)
    assert room["confidence"] == stored_confidence

    # Nonzero weight: now moves, and detail records the bonus applied.
    room2 = {
        "confidence": stored_confidence,
        "confidence_detail": {
            "detection": 0.9, "classification": 0.7, "ocr": 0.75,
        },
    }
    ConfidenceComputer.apply_corroboration(
        room2, corroboration_score=1.0, w_corroboration=0.1,
    )
    assert room2["confidence"] > stored_confidence
    assert room2["confidence_detail"]["corroboration"] == 1.0


_ROOM_CONFIDENCE = round(0.3 * 0.9 + 0.4 * 0.7 + 0.3 * 0.75, 4)


def _room_dict(confidence=_ROOM_CONFIDENCE, det=0.9, cls=0.7, ocr=0.75):
    return {
        "confidence": confidence,
        "confidence_detail": {"detection": det, "classification": cls, "ocr": ocr},
        "coordinates": {"bbox": [0, 0, 100, 100]},
    }


def test_apply_object_attributes_default_weights_preserve_confidence():
    """Default call (no weight kwargs, matching every pre-T-1 call
    site) must not change confidence even for a room with a full-evidence
    mapping match."""
    rooms = [_room_dict()]
    mapping = RoomWindowMapping(
        room_id=0, has_door=True, has_windows=True, window_count=1,
    )
    result = AnnotationPipeline._apply_object_attributes(rooms, [mapping])
    assert len(result) == len(rooms)  # A2: room count invariant
    assert result[0]["confidence"] == _ROOM_CONFIDENCE
    assert result[0]["coordinates"]["attributes"]["evidence_score"] == 2 / 5


def test_apply_object_attributes_nonzero_weight_boosts_matched_only():
    """Two rooms: one matched (evidence), one not. Only the matched
    room's confidence should move; the unmatched room must be untouched
    (A2 + asymmetric-absence requirement) and room count must be
    preserved either way."""
    rooms = [_room_dict(), _room_dict()]
    mapping = RoomWindowMapping(room_id=0, has_door=True, window_count=1)

    result = AnnotationPipeline._apply_object_attributes(
        rooms, [mapping], w_corroboration=0.15,
    )
    assert len(result) == 2
    assert result[0]["confidence"] > _ROOM_CONFIDENCE  # matched room: boosted
    assert result[1]["confidence"] == _ROOM_CONFIDENCE  # no mapping: untouched
    assert "attributes" not in result[1].get("coordinates", {})


_REAL_FIXTURE = Path(
    "/home/ubuntu/floorplan_classifier/VLM/t_q1q5_verify_test3/annotations/"
    "326 ROCKAWAY - AVI-ON LAYOUT_page001.json"
)


def _real_rooms_and_mappings():
    """Real production rooms + real, already-persisted YOLO detections
    from the same page/same coordinate space (no image file or live
    detector needed -- both were computed together in one real run and
    saved to this fixture, so they are guaranteed self-consistent,
    unlike substituting a different render of "the same" page)."""
    import json
    data = json.loads(_REAL_FIXTURE.read_text())
    rooms = data["rooms"]
    detections = [
        WindowDetection(
            bbox=tuple(det["bbox"]),
            confidence=det.get("confidence", 1.0),
            source_tier=WindowDetectionTier(det.get("source_tier", "none")),
            metadata={"type": det.get("category")},
        )
        for det in data["objectDetections"]
    ]
    rooms_for_mapping = [
        {"id": idx, "bbox": r["bbox"]} for idx, r in enumerate(rooms)
    ]
    mappings = map_detections_to_rooms(detections, rooms_for_mapping)
    return rooms, mappings


def _skip_if_real_fixture_missing():
    if not _REAL_FIXTURE.exists():
        pytest.skip(f"real project fixture not present ({_REAL_FIXTURE})")


def test_corroboration_default_weight_is_noop_on_real_production_data():
    """T-1 spike-lite: real page, real persisted YOLO detections (31),
    real rooms (39) -- not synthetic. Default weights must not move
    confidence even though this page has genuine object evidence."""
    _skip_if_real_fixture_missing()
    rooms, mappings = _real_rooms_and_mappings()
    assert mappings, "sanity: this real fixture must produce mappings"

    sft_rooms = [_room_dict() for _ in rooms]
    result = AnnotationPipeline._apply_object_attributes(sft_rooms, mappings)

    assert len(result) == len(rooms), "A2: room count invariant"
    assert all(r["confidence"] == _ROOM_CONFIDENCE for r in result)


def test_corroboration_nonzero_weight_boosts_and_never_penalizes_real_production_data():
    _skip_if_real_fixture_missing()
    rooms, mappings = _real_rooms_and_mappings()

    baseline = AnnotationPipeline._apply_object_attributes(
        [_room_dict() for _ in rooms], mappings,
    )
    boosted = AnnotationPipeline._apply_object_attributes(
        [_room_dict() for _ in rooms], mappings, w_corroboration=0.15,
    )

    assert len(boosted) == len(baseline) == len(rooms), "A2: room count invariant"
    baseline_conf = [r["confidence"] for r in baseline]
    boosted_conf = [r["confidence"] for r in boosted]
    assert any(b > a for a, b in zip(baseline_conf, boosted_conf)), (
        "expected at least one room to gain corroboration on this real, "
        "evidence-bearing page"
    )
    assert all(b >= a for a, b in zip(baseline_conf, boosted_conf)), (
        "no room may score lower with the term on than with it off"
    )
