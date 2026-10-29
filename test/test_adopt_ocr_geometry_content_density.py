#!/usr/bin/env python3
"""
B1 (2026-08-21 audit): _adopt_ocr_geometry must carry content_density
with the rest of the geometry provenance.

Root cause: pipeline.py runs SAM refine_annotations separately on VLM
rooms and OCR rooms *before* _merge_vlm_and_ocr -- each gets its own
content_density describing its own (pre-merge) bbox. _adopt_ocr_geometry
then swaps bbox/original_bbox/sam_*/segmentation from ocr_room onto the
VLM-sourced target dict, but content_density was left off that list --
the shipped room described OCR's box while reporting the VLM box's own,
unrelated density. Measured 16/16 (100%) of vlm_ocr_merged rooms stale
on a real 3-page run (max drift 0.2848) before this fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.sft_validator import _adopt_ocr_geometry


def test_content_density_follows_ocr_geometry():
    target = {
        "bbox": [10, 10, 500, 500],
        "content_density": 0.02,  # VLM room's own (giant, blank) box
        "room_name": "GENERIC ROOM",
    }
    ocr_room = {
        "bbox": [15, 15, 90, 20],
        "original_bbox": [15, 15, 90, 20],
        "sam_expanded": True,
        "sam_skip_reason": None,
        "sam_confidence": 0.9,
        "content_density": 0.14,  # OCR room's own (real, room-scale) box
    }

    _adopt_ocr_geometry(target, ocr_room, "CLASSROOM", ocr_room["bbox"])

    assert target["content_density"] == 0.14
    assert target["bbox"] == ocr_room["bbox"]


def test_content_density_absent_on_ocr_room_clears_target():
    """If ocr_room never got a density (SAM off / expansion skipped), the
    target must not keep reporting the discarded VLM box's own value --
    None is the documented 'unmeasured, skip the check' convention."""
    target = {"bbox": [10, 10, 500, 500], "content_density": 0.02}
    ocr_room = {"bbox": [15, 15, 90, 20]}

    _adopt_ocr_geometry(target, ocr_room, "CLASSROOM", ocr_room["bbox"])

    assert target["content_density"] is None
