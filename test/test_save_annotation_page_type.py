#!/usr/bin/env python3
"""
AV-6 integration audit-fix: _save_annotation's page_type_result merge
was wired into pipeline.py's STEP 3 call site but had never actually
been exercised -- prior W6 tests covered the classifier layer only
(UnslothQwenBackend.classify_page_type, UnslothPageClassifier,
AnnotationPipeline.page_classifier), never the write path itself.

Covers:
  - page_type_result=None (the default, and every pre-AV-6 call shape)
    writes no page_type/page_type_rationale keys at all
  - a real page_type_result is written verbatim into the saved JSON
  - a run with use_windows/use_yolo_objects both off (page_type is
    independent of the object-detection block) still writes it
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_annotator import VLMAnnotationResult


def _minimal_result(image_file: str = "page000.png") -> VLMAnnotationResult:
    return VLMAnnotationResult(
        image_file=image_file,
        image_size={"width": 1000, "height": 1000},
        rooms=[],
        panels=[],
        electrical_counts={},
    )


def test_no_page_type_key_when_result_is_none(tmp_path):
    pipeline = AnnotationPipeline(PipelineConfig())
    output_path = tmp_path / "page000.json"

    pipeline._save_annotation(
        _minimal_result(), output_path, ocr_rooms=[], img_path=None,
        page_type_result=None,
    )

    data = json.loads(output_path.read_text())
    assert "page_type" not in data
    assert "page_type_rationale" not in data


def test_page_type_result_written_verbatim(tmp_path):
    pipeline = AnnotationPipeline(PipelineConfig())
    output_path = tmp_path / "page000.json"

    pipeline._save_annotation(
        _minimal_result(), output_path, ocr_rooms=[], img_path=None,
        page_type_result={"page_type": "floor_plan", "rationale": "shows rooms"},
    )

    data = json.loads(output_path.read_text())
    assert data["page_type"] == "floor_plan"
    assert data["page_type_rationale"] == "shows rooms"


def test_page_type_written_with_object_detection_block_disabled(tmp_path):
    """use_windows/use_yolo_objects both default False -- the object-
    detection block (which page_type_result's merge sits right before)
    never runs; page_type must still be written regardless."""
    config = PipelineConfig()
    assert config.use_windows is False
    assert config.use_yolo_objects is False
    pipeline = AnnotationPipeline(config)
    output_path = tmp_path / "page000.json"

    pipeline._save_annotation(
        _minimal_result(), output_path, ocr_rooms=[], img_path=None,
        page_type_result={"page_type": "other", "rationale": "unclear"},
    )

    data = json.loads(output_path.read_text())
    assert data["page_type"] == "other"
