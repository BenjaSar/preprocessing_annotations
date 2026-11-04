#!/usr/bin/env python3
"""
Regression guard for P2 (sprint1_verify47 audit, 2026-08-19): a page
where SAM's label->room expansion failed on EVERY box due to a resource
error (measured: CUDA OOM, 3,093 times across one real run) produced
only a logger.info "0 expanded, N kept original" line -- identical in
severity to a healthy page where 0 expansions happen because nothing
needed expanding. 57/57 pages in that run silently failed this way and
it was found only by grepping the raw log by hand.

RoomSegmenter.refine_annotations now tracks per-box error causes and,
when EVERY box failed with an error (not a guardrail rejection), emits
one WARNING with a cause breakdown. No GPU/model needed to test this --
segment_from_point is patched directly on the instance; refine_annotations
takes img_width/img_height explicitly so it never touches a real file.
"""

import logging

from preprocessing_annotations.detection.sam_segmenter import RoomSegmenter


def _annotations(n=3):
    return [
        {"bbox": [100 + i * 50, 100, 40, 30], "room_name": f"ROOM{i}"}
        for i in range(n)
    ]


def test_all_boxes_erroring_logs_one_loud_warning(caplog):
    segmenter = RoomSegmenter()
    segmenter.segment_from_point = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory. Tried to allocate 1024.00 MiB.")
    )

    with caplog.at_level(logging.WARNING, logger="preprocessing_annotations.detection.sam_segmenter"):
        result = segmenter.refine_annotations(
            "nonexistent.png", _annotations(3), img_width=4500, img_height=3375,
        )

    assert len(result) == 3
    assert all(r["sam_expanded"] is False for r in result)

    loud = [r for r in caplog.records if "ALL 3 boxes failed" in r.message]
    assert len(loud) == 1, f"expected exactly one loud summary, got {len(loud)}"
    assert "CUDA out of memory" in loud[0].message


def test_partial_success_does_not_trigger_loud_warning(caplog):
    """One success among failures must NOT fire the all-failed alarm --
    this line means to isolate the total-failure case, not general
    imperfection."""
    segmenter = RoomSegmenter()
    calls = {"n": 0}

    def fake_segment(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            from preprocessing_annotations.detection.sam_segmenter import SegmentationResult
            return SegmentationResult(prompt_point=(0, 0), bbox=[90, 90, 200, 200], confidence=0.9)
        raise RuntimeError("CUDA out of memory.")

    segmenter.segment_from_point = fake_segment

    with caplog.at_level(logging.WARNING, logger="preprocessing_annotations.detection.sam_segmenter"):
        segmenter.refine_annotations(
            "nonexistent.png", _annotations(3), img_width=4500, img_height=3375,
        )

    loud = [r for r in caplog.records if "boxes failed" in r.message]
    assert loud == [], "partial success must not trigger the all-failed summary"


def test_healthy_zero_expansions_does_not_trigger_loud_warning(caplog):
    """0 expanded with 0 errors (every box legitimately guardrail-
    rejected, e.g. label_scale_noop) is a normal outcome -- must stay at
    the existing INFO level, not escalate to the resource-failure alarm."""
    segmenter = RoomSegmenter()

    def fake_segment(*a, **k):
        from preprocessing_annotations.detection.sam_segmenter import SegmentationResult
        # Same area as the label itself -> triggers the label_scale_noop
        # guardrail (kept original, no exception, no expansion).
        return SegmentationResult(prompt_point=(0, 0), bbox=[100, 100, 5, 5], confidence=0.9)

    segmenter.segment_from_point = fake_segment

    with caplog.at_level(logging.WARNING, logger="preprocessing_annotations.detection.sam_segmenter"):
        segmenter.refine_annotations(
            "nonexistent.png", _annotations(3), img_width=4500, img_height=3375,
        )

    loud = [r for r in caplog.records if "boxes failed" in r.message]
    assert loud == [], "guardrail-rejected boxes are not a resource failure"


def test_no_annotations_does_not_crash_or_warn(caplog):
    segmenter = RoomSegmenter()
    with caplog.at_level(logging.WARNING, logger="preprocessing_annotations.detection.sam_segmenter"):
        result = segmenter.refine_annotations(
            "nonexistent.png", [], img_width=4500, img_height=3375,
        )
    assert result == []
    assert not any("boxes failed" in r.message for r in caplog.records)
