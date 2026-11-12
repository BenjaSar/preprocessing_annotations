#!/usr/bin/env python3
"""
Y1 (2026-08-20 audit): MEPTextExtractor.extract_and_find_rooms's
extra_detections merge -- the corrected integration point for the PDF
text-layer prototype (see test_pdf_text_layer_merge.py for the fetch-only
half, and PipelineConfig.use_pdf_text_layer's docstring for why the
original integration point was wrong).

The contract this file exists to lock down:
  1. extra_detections=None -> byte-identical to pre-Y1 behavior.
  2. extra_detections items reach find_room_candidates (this is the
     actual fix -- the original design appended to raw_detections
     AFTER find_room_candidates had already run on it).
  3. The RETURNED raw_detections stays OCR-only, even when
     extra_detections contributed candidates -- this is what keeps
     compute_exclusion_zones safe.
  4. OCR wins every collision: an extra_detections item within
     _EXTRA_DETECTIONS_MERGE_TOL_PX of an existing OCR detection is
     dropped, not added.

Stubs extract_text() to avoid a real OCR/model call -- these tests
check the merge logic, not OCR itself.
"""

from unittest.mock import MagicMock, patch

from preprocessing_annotations.config import OCRConfig
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor, TextDetection


def _det(x1, y1, x2, y2, text="X", conf=0.9):
    return TextDetection(bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text=text, confidence=conf)


def _extractor_with_stubbed_ocr(ocr_detections):
    ex = MEPTextExtractor(OCRConfig())
    ex.extract_text = MagicMock(return_value=ocr_detections)
    return ex


def _fake_image(tmp_path):
    from PIL import Image
    p = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(p)
    return p


def test_extra_detections_none_is_byte_identical_to_before(tmp_path):
    ocr_dets = [_det(10, 10, 50, 30, "OCR TEXT")]
    ex = _extractor_with_stubbed_ocr(ocr_dets)
    img = _fake_image(tmp_path)

    candidates_a, raw_a = ex.extract_and_find_rooms(img)
    candidates_b, raw_b = ex.extract_and_find_rooms(img, extra_detections=None)

    assert raw_a == raw_b == ocr_dets
    assert [c.room_name for c in candidates_a] == [c.room_name for c in candidates_b]


def test_extra_detection_far_from_ocr_becomes_a_candidate(tmp_path):
    """The JANITOR case: reaches find_room_candidates, the actual fix."""
    ocr_dets = [_det(10, 10, 50, 30, "UNRELATED")]
    ex = _extractor_with_stubbed_ocr(ocr_dets)
    img = _fake_image(tmp_path)
    janitor = _det(1118, 1312, 1167, 1328, "JANITOR", conf=1.0)

    candidates, raw = ex.extract_and_find_rooms(img, extra_detections=[janitor])

    names = [c.room_name for c in candidates]
    assert "JANITOR" in names, f"JANITOR must reach find_room_candidates, got {names}"


def test_raw_detections_stays_ocr_only_even_when_extra_contributes(tmp_path):
    """The actual fix: raw_detections (which feeds compute_exclusion_zones
    in pipeline.py) must NOT contain extra_detections, even when they
    successfully became room candidates."""
    ocr_dets = [_det(10, 10, 50, 30, "UNRELATED")]
    ex = _extractor_with_stubbed_ocr(ocr_dets)
    img = _fake_image(tmp_path)
    janitor = _det(1118, 1312, 1167, 1328, "JANITOR", conf=1.0)

    candidates, raw = ex.extract_and_find_rooms(img, extra_detections=[janitor])

    assert raw == ocr_dets, "raw_detections must stay OCR-only regardless of extra_detections"
    assert janitor not in raw


def test_colliding_extra_detection_is_dropped_ocr_wins(tmp_path):
    """An extra_detections item within tolerance of an existing OCR
    detection must be dropped -- OCR wins the collision, matching
    _dedupe_detections's own centroid-tolerance convention."""
    ocr_det = _det(10, 10, 50, 30, "OCR TEXT")
    ex = _extractor_with_stubbed_ocr([ocr_det])
    img = _fake_image(tmp_path)
    # Centroid (30,20) -- well within tolerance of ocr_det's centroid (30,20).
    colliding = _det(15, 12, 45, 28, "GARBLED", conf=1.0)

    candidates, raw = ex.extract_and_find_rooms(img, extra_detections=[colliding])

    names = [c.room_name for c in candidates]
    assert "GARBLED" not in names, "colliding extra_detections item must not reach candidates"
    assert raw == [ocr_det]


def test_empty_extra_detections_list_is_a_noop(tmp_path):
    ocr_dets = [_det(10, 10, 50, 30, "OCR TEXT")]
    ex = _extractor_with_stubbed_ocr(ocr_dets)
    img = _fake_image(tmp_path)

    candidates, raw = ex.extract_and_find_rooms(img, extra_detections=[])

    assert raw == ocr_dets
