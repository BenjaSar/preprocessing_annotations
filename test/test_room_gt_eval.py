#!/usr/bin/env python3
"""
AV-8 room-GT harness tests (eval/room_gt_eval.py) on synthetic fixtures.

Covers:
  - load_persisted_detections: attached + unattached unioned per page;
    pre-N2 annotations (no unattachedObjectDetections) rejected
  - page_key: Label Studio upload prefix + punctuation rewrite and the
    pipeline's own file names normalize to the same key
  - load_logged_detections: per-page grouping from the debug log format
    pipeline._detect_object_mappings writes
  - score_localization: center-in-polygon vs bbox-IoU extent counts
  - score_attachment: a door touching the room's GT polygon is correct;
    one touching only a neighbor's polygon is wrong
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from shapely.geometry import box

from preprocessing_annotations.config import AttachmentConfig
from preprocessing_annotations.detection.attachment import AttachmentPolicyFactory
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
)
from room_gt_eval import (
    AttachmentScore,
    LocalizationScore,
    load_logged_detections,
    load_persisted_detections,
    page_key,
    score_attachment,
    score_localization,
)


def _door(bbox):
    return WindowDetection(
        bbox=bbox, confidence=0.9, source_tier=WindowDetectionTier.YOLO,
        metadata={"type": "door"},
    )


def test_page_key_matches_across_sources():
    gt_name = "C:\\x\\upload\\4\\f1272381-326_ROCKAWAY_-_AVI-ON_LAYOUT_page000.png"
    run_name = "326 ROCKAWAY - AVI-ON LAYOUT_page000.json"
    assert page_key(gt_name) == page_key(run_name)


def test_load_logged_detections_groups_by_page(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text(
        "12:00 | INFO | p | SAM3 exemplar: 2 door seeds (conf>=0.75) -> 0 additional doors for A_page000.png\n"
        "12:00 | DEBUG | p | YOLO object door: bbox=(1, 2, 3, 4) conf=0.80\n"
        "12:00 | INFO | p | SAM3 exemplar: 0 door seeds (conf>=0.75) -> 0 additional doors for B_page001.png\n"
        "12:00 | DEBUG | p | YOLO object window: bbox=(5, 6, 7, 8) conf=0.50\n"
        "12:00 | DEBUG | p | YOLO object door: bbox=(9, 9, 10, 10) conf=0.30\n"
        "12:00 | DEBUG | p | YOLO object door: bbox=(11.5, 12.25, 13.0, 14.75) conf=0.40\n"
    )

    detections = load_logged_detections(log)

    assert [d.bbox for d in detections[page_key("A_page000.png")]] == [(1, 2, 3, 4)]
    page_b = detections[page_key("B_page001.png")]
    assert [d.metadata["type"] for d in page_b] == ["window", "door", "door"]
    assert page_b[2].bbox == (11.5, 12.25, 13.0, 14.75)


def test_score_localization_label_vs_extent():
    gts = [box(0, 0, 100, 100), box(200, 0, 300, 100)]
    label_box = (40, 40, 60, 60)        # center inside gt 0, tiny -> IoU < 0.5
    extent_box = (200, 0, 300, 100)     # exact gt 1 extent
    score = LocalizationScore()

    score_localization([label_box, extent_box], gts, 0.5, score)

    assert score.predictions == 2
    assert score.predictions_center_in_gt == 2
    assert score.gt_rooms_located == 2
    assert score.predictions_iou_match == 1
    assert score.gt_rooms_iou_matched == 1


def test_score_localization_mean_iou_and_derived_rates():
    gts = [box(0, 0, 100, 100), box(200, 0, 300, 100)]
    score = LocalizationScore()

    score_localization([(40, 40, 60, 60), (200, 0, 300, 100)], gts, 0.5, score)
    report = score.to_dict()

    assert report["mean_iou_per_prediction"] == pytest.approx((0.04 + 1.0) / 2)
    assert report["mean_iou_per_gt_room"] == pytest.approx((0.04 + 1.0) / 2)
    assert report["label_f1"] == 1.0 and report["label_fnr"] == 0.0
    assert report["extent_f1"] == pytest.approx(0.5)


def test_score_attachment_correct_and_wrong():
    gts = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    rooms = [(40, 40, 60, 60), (140, 40, 160, 60)]   # one label per GT room
    on_room0 = _door((45, 45, 55, 55))                # inside room 0 only
    policy = AttachmentPolicyFactory.create(AttachmentConfig())
    score = AttachmentScore()

    score_attachment(policy, [on_room0], rooms, gts, score)

    assert score.correct == 1
    assert score.wrong == 0
    assert score.attachable == 1


def test_score_attachment_counts_wrong_room():
    gts = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    # Room 1's label sits at the far side of its room: center (187.5, 50).
    rooms = [(40, 40, 60, 60), (180, 40, 195, 60)]
    # Door touches only GT room 1 (center (103, 50)), but is nearer room
    # 0's label center (~53px) than room 1's (~84.5px).
    door = _door((101, 45, 105, 55))
    policy = AttachmentPolicyFactory.create(
        AttachmentConfig(policy="nearest_with_cutoff", nearest_cutoff_px=60)
    )
    score = AttachmentScore()

    score_attachment(policy, [door], rooms, gts, score)

    assert score.wrong == 1
    assert score.correct == 0


def _write_annotation(run_dir, name, **keys):
    annotations = run_dir / "annotations"
    annotations.mkdir(exist_ok=True)
    (annotations / name).write_text(json.dumps(keys))


def _record(bbox, category="door", tier="yolo"):
    return {"category": category, "bbox": bbox, "confidence": 0.8, "source_tier": tier}


def test_load_persisted_detections_unions_attached_and_unattached(tmp_path):
    _write_annotation(
        tmp_path, "A_page000.json",
        objectDetections=[_record([1, 2, 3, 4])],
        unattachedObjectDetections=[_record([5, 6, 7, 8], "window", "sam3_exemplar")],
    )

    detections = load_persisted_detections(tmp_path)[page_key("A_page000.json")]

    assert [(d.bbox, d.metadata["type"], d.source_tier.value) for d in detections] == [
        ((1, 2, 3, 4), "door", "yolo"),
        ((5, 6, 7, 8), "window", "sam3_exemplar"),
    ]
    assert detections[0].confidence == 0.8


def test_load_persisted_detections_rejects_pre_n2_annotation(tmp_path):
    _write_annotation(tmp_path, "A_page000.json", objectDetections=[_record([1, 2, 3, 4])])

    with pytest.raises(ValueError, match="A_page000.json"):
        load_persisted_detections(tmp_path)
