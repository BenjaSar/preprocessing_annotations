#!/usr/bin/env python3
"""
Prototype (2026-08-20 audit, corrected same day -- Y1): PDF text-layer
recall, additive-only against OCR. Confirmed real: Lake Shore Electrical
p002's PDF text layer contains 'JANITOR' at image coords (1118,1312)
where OCR found nothing.

Y1 corrects the original integration point. The first version merged
PDF detections into raw_detections in pipeline.py -- but raw_detections
is returned from find_rooms_pass1 BEFORE candidate classification
happens inside it, so nothing merged there could ever become a room
candidate, AND raw_detections also feeds compute_exclusion_zones, which
is confirmed unsafe with PDF-sourced input (see
test_pdf_text_layer_exclusion_zone_hazard.py). Corrected design, split
across two responsibilities:
  - AnnotationPipeline._pdf_text_layer_detections: FETCH ONLY. Returns
    PDF-sourced detections or None. No merging, no collision logic.
  - MEPTextExtractor.extract_and_find_rooms(extra_detections=...): does
    the OCR-wins collision filter and merges ONLY into the list passed
    to find_room_candidates. raw_detections it returns stays OCR-only.
    Covered in test_ocr_extractor_extra_detections.py.

This file covers the fetch-only half.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.ingestion.ocr_extractor import TextDetection


def _det(x1, y1, x2, y2, text="X", conf=0.9):
    return TextDetection(bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text=text, confidence=conf)


def _pipeline(use_pdf_text_layer: bool, input_path: Path) -> AnnotationPipeline:
    config = PipelineConfig()
    config.use_pdf_text_layer = use_pdf_text_layer
    pipeline = AnnotationPipeline(config)
    pipeline._input_path = input_path
    return pipeline


def test_flag_off_returns_none(tmp_path):
    pipeline = _pipeline(use_pdf_text_layer=False, input_path=tmp_path)
    pipeline._pdf_extractor = MagicMock()  # must never be touched

    result = pipeline._pdf_text_layer_detections(Path("building_page001.png"))

    assert result is None
    pipeline._pdf_extractor.extract_text_layer.assert_not_called()


def test_flag_on_no_source_pdf_locatable_returns_none(tmp_path):
    # No matching PDF exists in input_path -> _source_pdf_for returns None.
    pipeline = _pipeline(use_pdf_text_layer=True, input_path=tmp_path)
    pipeline._pdf_extractor = MagicMock()

    result = pipeline._pdf_text_layer_detections(Path("building_page001.png"))

    assert result is None
    pipeline._pdf_extractor.extract_text_layer.assert_not_called()


def test_flag_on_unparseable_page_index_returns_none(tmp_path):
    (tmp_path / "building.pdf").write_bytes(b"%PDF-1.4\n")
    pipeline = _pipeline(use_pdf_text_layer=True, input_path=tmp_path)
    pipeline._pdf_extractor = MagicMock()

    # No "_pageNNN" suffix -> _source_pdf_for itself returns None for this
    # stem (matches its own "not a page-extracted image" contract), so
    # this also never reaches extract_text_layer.
    result = pipeline._pdf_text_layer_detections(Path("building.png"))

    assert result is None
    pipeline._pdf_extractor.extract_text_layer.assert_not_called()


def test_flag_on_empty_pdf_text_layer_returns_none(tmp_path):
    """The '...FLAT' PDF case: text outlined to curves, zero words."""
    (tmp_path / "building.pdf").write_bytes(b"%PDF-1.4\n")
    pipeline = _pipeline(use_pdf_text_layer=True, input_path=tmp_path)
    pipeline._pdf_extractor = MagicMock()
    pipeline._pdf_extractor.extract_text_layer.return_value = []

    result = pipeline._pdf_text_layer_detections(Path("building_page001.png"))

    assert result is None
    pipeline._pdf_extractor.extract_text_layer.assert_called_once_with(
        tmp_path / "building.pdf", 1
    )


def test_flag_on_returns_pdf_detections_unfiltered(tmp_path):
    """No collision filtering here -- that happens downstream in
    extract_and_find_rooms, which is the only place OCR's own
    raw_detections exists to filter against."""
    (tmp_path / "building.pdf").write_bytes(b"%PDF-1.4\n")
    pipeline = _pipeline(use_pdf_text_layer=True, input_path=tmp_path)
    pipeline._pdf_extractor = MagicMock()
    dets = [_det(1118, 1312, 1167, 1328, "JANITOR", conf=1.0)]
    pipeline._pdf_extractor.extract_text_layer.return_value = dets

    result = pipeline._pdf_text_layer_detections(Path("building_page002.png"))

    assert result == dets
    pipeline._pdf_extractor.extract_text_layer.assert_called_once_with(
        tmp_path / "building.pdf", 2
    )
