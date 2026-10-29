#!/usr/bin/env python3
"""
Regression guard for the exclusion-zone guardrail (Q1, sprint1_verify47
audit, 2026-08-19): RoomSegmenter.refine_annotations accepted
exclusion_zones and the pipeline dutifully threaded BOM/title-block
zones into it, but the parameter was only ever consumed inside the
use_density_prompts branch (default off) -- the main label-centroid
expansion loop never checked it at all.

Confirmed failure (visually verified crop): a seed on the word
"ELECTRICAL" inside a sheet title ("ELECTRICAL LIGHTING FIRST FLOOR
PLAN") got expanded by SAM into the entire title block -- revision
table, seal, sheet number -- 3.90% of the page. No GPU needed to test
this: segment_from_point is patched directly, and the guardrail fires
BEFORE it would even be called.

Covers:
  - a seed centroid inside an exclusion zone is kept original, tagged,
    and never reaches segment_from_point at all (no wasted SAM call)
  - a seed centroid outside every exclusion zone is unaffected
  - exclusion_zones=None (the default, and every pre-existing call
    site/test) behaves byte-identically to before this guardrail existed
  - exclusion_zones=[] (empty list, distinct from None) also changes
    nothing
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.detection.sam_segmenter import RoomSegmenter, SegmentationResult


def _annotation(bbox):
    return [{"bbox": bbox, "room_name": "TEST"}]


def _expanding_segmenter():
    """A segmenter whose segment_from_point always returns a large,
    otherwise-valid expansion -- used to prove the exclusion check fires
    BEFORE this would ever be called, and does nothing when it's not
    supposed to fire."""
    segmenter = RoomSegmenter()
    calls = []

    def fake_segment(image_path, point, include_mask=True, label_bbox=None):
        calls.append(point)
        return SegmentationResult(prompt_point=point, bbox=[point[0] - 50, point[1] - 50, 100, 100], confidence=0.9)

    segmenter.segment_from_point = fake_segment
    segmenter._calls = calls
    return segmenter


def test_seed_inside_exclusion_zone_is_skipped_before_sam_call():
    segmenter = _expanding_segmenter()
    # Label centroid at (110, 60); zone covers x[0,200] y[0,100].
    bbox = [100, 50, 20, 20]

    result = segmenter.refine_annotations(
        "nonexistent.png", _annotation(bbox),
        img_width=4500, img_height=3375,
        exclusion_zones=[(0, 0, 200, 100)],
    )

    assert len(result) == 1
    assert result[0]["sam_expanded"] is False
    assert result[0]["sam_skip_reason"] == "exclusion_zone"
    assert result[0]["bbox"] == bbox, "original bbox must be preserved, not SAM's"
    assert segmenter._calls == [], "SAM must never be called for an excluded seed"


def test_seed_outside_every_exclusion_zone_is_unaffected():
    segmenter = _expanding_segmenter()
    bbox = [1000, 1000, 20, 20]  # centroid (1010,1010), far from the zone below

    result = segmenter.refine_annotations(
        "nonexistent.png", _annotation(bbox),
        img_width=4500, img_height=3375,
        exclusion_zones=[(0, 0, 200, 100)],
    )

    assert len(result) == 1
    assert result[0]["sam_expanded"] is True
    assert result[0].get("sam_skip_reason") is None
    assert segmenter._calls == [(1010, 1010)]


def test_exclusion_zones_none_is_byte_identical_to_no_guardrail():
    """The default (None) -- every pre-existing call site -- must behave
    exactly as it did before this guardrail was added."""
    segmenter = _expanding_segmenter()
    bbox = [100, 50, 20, 20]

    result = segmenter.refine_annotations(
        "nonexistent.png", _annotation(bbox),
        img_width=4500, img_height=3375,
        exclusion_zones=None,
    )

    assert result[0]["sam_expanded"] is True
    assert segmenter._calls == [(110, 60)]


def test_exclusion_zones_empty_list_changes_nothing():
    segmenter = _expanding_segmenter()
    bbox = [100, 50, 20, 20]

    result = segmenter.refine_annotations(
        "nonexistent.png", _annotation(bbox),
        img_width=4500, img_height=3375,
        exclusion_zones=[],
    )

    assert result[0]["sam_expanded"] is True
    assert segmenter._calls == [(110, 60)]


def test_multiple_zones_only_the_containing_one_matters():
    segmenter = _expanding_segmenter()
    bbox = [1000, 1000, 20, 20]  # centroid (1010,1010)

    result = segmenter.refine_annotations(
        "nonexistent.png", _annotation(bbox),
        img_width=4500, img_height=3375,
        exclusion_zones=[(0, 0, 200, 100), (900, 900, 1100, 1100)],
    )

    assert result[0]["sam_expanded"] is False
    assert result[0]["sam_skip_reason"] == "exclusion_zone"
    assert segmenter._calls == []
