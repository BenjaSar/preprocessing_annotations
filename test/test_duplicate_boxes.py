"""automation/duplicate_boxes.py: one room reported twice — a room-extent box
around a label box of the same room type — keeps only the label box."""

import dataclasses

from preprocessing_annotations.automation.duplicate_boxes import split_redundant_room_extents
from preprocessing_annotations.config import PipelineConfig

EXTENT_SOURCE, LABEL_SOURCE = "vlm_only", "ocr_only"


def _room(source, bbox, room_type="CLASSROOM"):
    return {"source": source, "bbox": list(bbox), "type": room_type, "room_name": room_type}


def test_room_extent_box_holding_a_label_box_of_the_same_type_is_dropped():
    extent = _room(EXTENT_SOURCE, (100, 100, 400, 300))
    label = _room(LABEL_SOURCE, (200, 180, 60, 20))
    kept, dropped = split_redundant_room_extents([extent, label])
    assert kept == [label]
    assert dropped == [extent]


def test_label_box_of_a_different_type_does_not_make_it_redundant():
    extent = _room(EXTENT_SOURCE, (100, 100, 400, 300))
    label = _room(LABEL_SOURCE, (200, 180, 60, 20), room_type="CORRIDOR")
    kept, dropped = split_redundant_room_extents([extent, label])
    assert dropped == []
    assert kept == [extent, label]


def test_sole_room_extent_box_is_kept():
    """The case option B got wrong: no label box inside, so this box is the
    only record of that room."""
    extent = _room(EXTENT_SOURCE, (100, 100, 400, 300))
    far_label = _room(LABEL_SOURCE, (900, 900, 60, 20))
    kept, dropped = split_redundant_room_extents([extent, far_label])
    assert dropped == []
    assert kept == [extent, far_label]


def test_label_boxes_are_never_dropped_even_inside_another_label_box():
    outer = _room(LABEL_SOURCE, (100, 100, 400, 300))
    inner = _room("vlm_ocr_merged", (200, 180, 60, 20))
    kept, dropped = split_redundant_room_extents([outer, inner])
    assert dropped == []


def test_rooms_without_a_usable_bbox_are_left_alone():
    broken = {"source": EXTENT_SOURCE, "bbox": [1, 2], "type": "CLASSROOM"}
    label = _room(LABEL_SOURCE, (0, 0, 10, 10))
    kept, dropped = split_redundant_room_extents([broken, label])
    assert dropped == []
    assert kept == [broken, label]


def test_unknown_source_has_no_convention_and_is_untouched():
    unknown = _room("some_new_source", (100, 100, 400, 300))
    label = _room(LABEL_SOURCE, (200, 180, 60, 20))
    assert split_redundant_room_extents([unknown, label])[1] == []


def test_config_default_is_on_and_snapshot_records_it():
    from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline

    config = PipelineConfig()
    assert config.room_filter.drop_redundant_room_extent_boxes is True
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["room_filter"]["drop_redundant_room_extent_boxes"] is True
    assert snapshot["ocr"]["room_name_placeholder_pattern"] == config.ocr.room_name_placeholder_pattern


def test_filter_reports_the_drop_and_the_flag_turns_it_off():
    from preprocessing_annotations.automation.sft_validator import prepare_sft_annotation

    config = PipelineConfig()
    annotation = {
        "image_file": "p.png", "image_size": {"width": 4000, "height": 3000}, "rooms": [], "ocr_rooms": [],
    }

    def _prepare(room_filter):
        payload = dict(annotation)
        payload["rooms"] = [
            dict(_room(EXTENT_SOURCE, (100, 100, 400, 300)), room_name="CLASSROOM"),
            dict(_room(LABEL_SOURCE, (200, 180, 60, 20)), room_name="CLASSROOM"),
        ]
        return prepare_sft_annotation(payload, min_rooms_for_sft=config.min_rooms_for_sft, room_filter=room_filter)

    on = _prepare(config.room_filter)
    assert on["drop_attribution"].get("redundant_room_extent_box") == 1
    assert [r["source"] for r in on["rooms"]] == [LABEL_SOURCE]

    off = _prepare(dataclasses.replace(config.room_filter, drop_redundant_room_extent_boxes=False))
    assert "redundant_room_extent_box" not in off["drop_attribution"]
    assert len(off["rooms"]) == 2
