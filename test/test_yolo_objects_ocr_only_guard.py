#!/usr/bin/env python3
"""
P-4: warn when --use-yolo-objects runs without --use-vlm.

Gap: OCR-only room boxes are text-label bboxes (e.g. 71x14px), not room
extents, so map_detections_to_rooms' geometric intersection almost never
attaches a detection to a room -- confirmed on a real page (P-1,
2026-09-03): 26 real rooms + 44 real detections -> 0 attached. See
memory use-yolo-objects-ocr-only-nearly-noop. This does not affect any
of the 14 real runs on disk (all use_vlm=True), but the flag
combination itself silently burns a full tiled-detector pass with
~zero attached signal if run OCR-only, so warn rather than let that
go unnoticed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import (
    _warn_if_yolo_objects_ocr_only,
)


def test_warns_when_yolo_objects_without_vlm(caplog):
    config = PipelineConfig()
    config.use_yolo_objects = True
    config.use_vlm = False

    with caplog.at_level("WARNING"):
        _warn_if_yolo_objects_ocr_only(config)

    assert any(
        "text-label bboxes, not room extents" in r.message for r in caplog.records
    )


def test_no_warning_when_use_vlm_enabled(caplog):
    config = PipelineConfig()
    config.use_yolo_objects = True
    config.use_vlm = True

    with caplog.at_level("WARNING"):
        _warn_if_yolo_objects_ocr_only(config)

    assert caplog.records == []


def test_no_warning_when_yolo_objects_disabled(caplog):
    config = PipelineConfig()  # both default False
    with caplog.at_level("WARNING"):
        _warn_if_yolo_objects_ocr_only(config)

    assert caplog.records == []
