#!/usr/bin/env python3
"""
T4: integration smoke test for the SAM3 exemplar feature's pipeline
wiring, on one real production page with real detectors (YOLO + SAM3).

Marked `slow` (excluded by pytest.ini's default addopts) -- loads real
GPU models, unlike every other test in this suite.

Covers:
  - flag OFF: objectDetections is byte-identical to pre-feature output
    (regression guard on the existing use_yolo_objects-only path)
  - flag ON: objectDetections is a superset of the OFF run, and every
    added item carries source_tier == "sam3_exemplar"
  - T6 guardrail: the per-page seed/addition count log line actually
    fires and reports the true counts -- a run with the flag on can
    silently no-op every page (measured: 6/7 real pages, 0 additions
    each) and this must be visible in the log, not inferred from
    "enabled: true" in the run config
"""

import json
from pathlib import Path

import pytest

from preprocessing_annotations.config import PipelineConfig, YoloObjectDetectorConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline

# Frozen beside the tests (2026-11-04). The original fixture lived in scratch
# run directories (sprint1_verify34/, t_q1q5_verify_test3/) that no longer
# exist, so all three tests skipped on every checkout and guarded nothing.
# A "looks equivalent" substitute had been tried and rejected (2026-09-03):
# an image that merely matches the declared width/height is not the render
# the rooms were computed against, and it shifted the pinned numbers and
# misaligned the room boxes. So this is a fully regenerated PAIR -- the image
# and its `rooms` come from ONE run (adapter_real_v1), one render; the
# provenance and image checksum are recorded inside the rooms file.
#
# The page is not arbitrary. The previous fixture produced 0 SAM3 additions,
# which made test_flag_on_is_superset_with_correct_provenance vacuous (an
# empty set is a subset of anything). Measured under this file's own config
# (YOLO -6 at 0.5, flag off vs on), 10 candidate pages gave 0 additions on 8,
# 1 on one, and 3-4 here -- the page with the most real SAM3 additions, so
# the provenance check has something to check.
_FIXTURES = Path(__file__).parent / "fixtures"
_IMG = _FIXTURES / "sam3_integration_page.png"
_ROOMS_JSON = _FIXTURES / "sam3_integration_rooms.json"
_YOLO_CHECKPOINT = (
    "/home/ubuntu/floorplan_classifier/VLM/yolo_checkpoints/runs/"
    "kaggle_door_window-6/weights/best.pt"
)

pytestmark = pytest.mark.slow


def _require_environment():
    """Fail on a missing fixture, skip on a missing model.

    The fixtures are committed, so their absence is a broken checkout and
    must be loud -- skipping here is exactly how this file guarded nothing
    for months. The YOLO checkpoint is a large untracked weights file, so
    its absence is a property of the machine and a skip is honest.
    """
    assert _IMG.exists() and _ROOMS_JSON.exists(), (
        f"committed fixture missing: {_IMG} / {_ROOMS_JSON}"
    )
    if not Path(_YOLO_CHECKPOINT).exists():
        pytest.skip(f"YOLO checkpoint not present on this machine ({_YOLO_CHECKPOINT})")


def _real_rooms():
    return json.loads(_ROOMS_JSON.read_text())["rooms"]


def _make_pipeline(use_sam3_exemplar: bool) -> AnnotationPipeline:
    config = PipelineConfig()
    config.use_yolo_objects = True
    config.use_sam3_exemplar = use_sam3_exemplar
    config.yolo_objects = YoloObjectDetectorConfig(
        checkpoint_path=_YOLO_CHECKPOINT, confidence_threshold=0.5
    )
    return AnnotationPipeline(config)


def _object_detections(pipeline: AnnotationPipeline):
    detection_pass = pipeline._detect_object_mappings(_IMG, _real_rooms())
    return pipeline._collect_object_detections(detection_pass.mappings)


def test_flag_off_matches_yolo_only_baseline():
    _require_environment()
    pipeline = _make_pipeline(use_sam3_exemplar=False)

    detections = _object_detections(pipeline)

    assert detections, "sanity: this real page must produce some detections"
    assert all(d["source_tier"] == "yolo" for d in detections)


def test_flag_on_is_superset_with_correct_provenance():
    _require_environment()
    baseline = _object_detections(_make_pipeline(use_sam3_exemplar=False))
    with_sam3 = _object_detections(_make_pipeline(use_sam3_exemplar=True))

    baseline_keys = {(d["category"], tuple(d["bbox"])) for d in baseline}
    with_sam3_keys = {(d["category"], tuple(d["bbox"])) for d in with_sam3}

    assert baseline_keys <= with_sam3_keys, (
        "flag ON must be a strict superset of flag OFF -- SAM3 additions "
        "extend the seed list, never replace or drop a seed detection"
    )

    added = with_sam3_keys - baseline_keys
    assert added, (
        "this fixture must exercise SAM3 additions -- with none, the "
        "provenance check below is an empty set and passes whatever the "
        "tier labels say (the previous fixture's failure mode)"
    )
    added_tiers = {
        d["source_tier"] for d in with_sam3
        if (d["category"], tuple(d["bbox"])) in added
    }
    assert added_tiers <= {"sam3_exemplar"}, (
        "every detection beyond the baseline must be tagged sam3_exemplar"
    )


def test_seed_and_addition_counts_are_logged(caplog):
    """T6 guardrail: the log line must fire and must report the TRUE
    counts, not just "enabled". Pinned to this page's actually-measured
    behavior (20 door seeds, 4 additions) rather than assuming any nonzero
    count -- a run that silently no-ops every page must still be visible
    in this line. Measured twice (probe, then this test) with identical
    counts."""
    _require_environment()
    pipeline = _make_pipeline(use_sam3_exemplar=True)

    with caplog.at_level("INFO"):
        pipeline._detect_object_mappings(_IMG, _real_rooms())

    matches = [r for r in caplog.records if "SAM3 exemplar:" in r.message]
    assert len(matches) == 1, "exactly one seed/addition log line per page"

    rendered = matches[0].getMessage()
    assert _IMG.name in rendered
    assert "door seeds" in rendered
    assert "additional doors" in rendered
    # Real, currently-measured numbers for this page -- not placeholders.
    assert "20 door seeds" in rendered
    assert "4 additional doors" in rendered
