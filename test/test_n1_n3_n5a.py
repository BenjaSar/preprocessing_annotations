"""N1/N3/N5a regression tests (2026-10-10).

N1: drop_detail populated per filter step that drops rooms.
N3: CAFETERIA, MEDIA CENTER, MDF, IDF pass the semantic gate and map in taxonomy.
N5a: RoomFilterConfig drives prepare_sft_annotation / _merge_vlm_and_ocr;
     defaults equal the former literals.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.sft_validator import (
    SemanticRoomValidator,
    _merge_vlm_and_ocr,
    prepare_sft_annotation,
)
from preprocessing_annotations.automation.taxonomy import normalize_to_mandatory
from preprocessing_annotations.config import RoomFilterConfig


def _annotation(rooms):
    return {
        "image_file": "test.png",
        "image_size": {"width": 2000, "height": 2000},
        "rooms": rooms,
        "ocr_rooms": [],
    }


def _room(name, bbox, source):
    return {"room_name": name, "bbox": list(bbox), "source": source,
            "confidence": 0.9, "content_density": 0.15}


@pytest.mark.parametrize("name, expected", [
    ("CAFETERIA", "CAFETERIA"),
    ("MEDIA CENTER", "CLASSROOM"),
    ("MDF", "ELECTRICAL ROOM"),
    ("IDF", "ELECTRICAL ROOM"),
])
def test_n3_term_passes_semantic_gate_and_maps(name, expected):
    kept = SemanticRoomValidator().filter_rooms([_room(name, (0, 0, 300, 300), "ocr_only")])
    assert len(kept) == 1
    assert normalize_to_mandatory(name) == expected


def test_n5a_room_min_area_drives_room_scale_gate():
    room = (_room("OFFICE", (100, 100, 250, 250), "vlm_only"),)  # 62,500 px²
    default = prepare_sft_annotation(_annotation([dict(r) for r in room]))
    raised = prepare_sft_annotation(
        _annotation([dict(r) for r in room]),
        room_filter=RoomFilterConfig(room_min_area=62_501),
    )
    assert len(default["rooms"]) == 1
    assert raised["rooms"] == []
    assert raised["drop_attribution"]["not_room_scale"] == 1


def test_n5a_merge_proximity_drives_vlm_ocr_match():
    vlm = [{"room_name": "OFFICE", "bbox": [0, 0, 100, 100]}]
    ocr = [{"name": "OFFICE", "bbox": [150, 0, 50, 50]}]  # 50 px edge gap
    assert len(_merge_vlm_and_ocr(vlm, ocr, RoomFilterConfig().merge_proximity_px).rooms) == 1
    assert len(_merge_vlm_and_ocr(vlm, ocr, 0).rooms) == 2


def test_n1_semantic_drop_detail_populated():
    result = prepare_sft_annotation(
        _annotation([_room("XYZZY UNKNOWN", (100, 100, 500, 500), "vlm_ocr_merged")])
    )
    assert result["drop_detail"]["semantic"] == [
        {"room_name": "XYZZY UNKNOWN", "bbox": [100, 100, 500, 500], "source": "vlm_ocr_merged"}
    ]


def test_n1_not_room_scale_drop_detail_populated():
    # 14,400 px²: passes geometric floor (10,000), fails room-scale floor (40,000).
    result = prepare_sft_annotation(
        _annotation([_room("OFFICE", (100, 100, 120, 120), "vlm_only")])
    )
    assert [d["room_name"] for d in result["drop_detail"]["not_room_scale"]] == ["OFFICE"]


def test_n1_no_drop_detail_when_nothing_dropped():
    result = prepare_sft_annotation(
        _annotation([_room("OFFICE", (100, 100, 250, 250), "vlm_only")])
    )
    assert result["rooms"]
    assert result.get("drop_detail", {}) == {}


def test_n1b_merge_fragment_skip_recorded():
    vlm = [{"room_name": "TYPE-D2 3BR", "bbox": [0, 0, 100, 100]}]
    ocr = [{"name": "BR", "bbox": [150, 0, 50, 50]}]
    result = _merge_vlm_and_ocr(vlm, ocr, 100)
    assert len(result.rooms) == 1
    assert result.discards == {
        "merge_fragment_skip": [{"room_name": "BR", "bbox": [150, 0, 50, 50], "source": None}]
    }


def test_n1b_merge_geometry_replacement_recorded_with_original_vlm_box():
    vlm = [{"room_name": "OFF", "bbox": [0, 0, 100, 100]}]
    ocr = [{"name": "OFFICE", "bbox": [10, 10, 50, 20]}]
    result = _merge_vlm_and_ocr(vlm, ocr, 100)
    assert result.rooms[0]["bbox"] == [10, 10, 50, 20]
    assert result.discards == {
        "merge_vlm_geometry_replaced": [
            {"room_name": "OFF", "bbox": [0, 0, 100, 100], "source": "vlm_only"}
        ]
    }


def test_n1b_merge_discards_reach_drop_detail_and_attribution():
    ann = _annotation([{"room_name": "TYPE-D2 3BR", "bbox": [100, 100, 300, 300]}])
    ann["ocr_rooms"] = [{"name": "BR", "bbox": [450, 100, 50, 50]}]
    result = prepare_sft_annotation(ann)
    assert result["drop_attribution"]["merge_fragment_skip"] == 1
    assert result["drop_detail"]["merge_fragment_skip"][0]["room_name"] == "BR"


def _echo_annotation():
    """One VLM room plus an OCR candidate that is the same detection (the
    two-pass VLM fallback's output): same box, truncated to int."""
    return {
        "image_file": "test.png",
        "image_size": {"width": 2000, "height": 2000},
        "rooms": [{"room_name": "OFFICE", "bbox": [100.0, 100.0, 250.0, 250.0]}],
        "ocr_rooms": [{"name": "Office", "bbox": [100, 100, 250, 249],
                       "confidence": 0.95, "source": "ocr_only"}],
    }


def test_echo_candidate_corroborates_when_skip_disabled():
    result = prepare_sft_annotation(
        _echo_annotation(), room_filter=RoomFilterConfig(skip_vlm_echo_candidates=False)
    )
    assert [r["source"] for r in result["rooms"]] == ["vlm_ocr_merged"]


def test_echo_candidate_skipped_by_default_leaving_room_vlm_only():
    result = prepare_sft_annotation(_echo_annotation())
    assert [r["source"] for r in result["rooms"]] == ["vlm_only"]
    assert result["drop_attribution"]["vlm_echo_skipped"] == 1
    assert result["drop_detail"]["vlm_echo_skipped"][0]["room_name"] == "Office"


def test_skip_vlm_echo_matches_box_even_when_label_differs():
    annotation = _echo_annotation()
    annotation["ocr_rooms"][0]["name"] = "STORAGE"
    result = prepare_sft_annotation(
        annotation, room_filter=RoomFilterConfig(skip_vlm_echo_candidates=True)
    )
    assert result["drop_attribution"]["vlm_echo_skipped"] == 1


def test_skip_vlm_echo_keeps_independent_ocr_candidate():
    annotation = _echo_annotation()
    annotation["ocr_rooms"][0]["bbox"] = [900, 900, 250, 249]
    result = prepare_sft_annotation(
        annotation, room_filter=RoomFilterConfig(skip_vlm_echo_candidates=True)
    )
    assert "vlm_echo_skipped" not in result.get("drop_attribution", {})


def test_vlm_echo_tolerance_is_configurable():
    annotation = _echo_annotation()
    annotation["ocr_rooms"][0]["bbox"] = [100, 100, 250, 245]  # 5 px off
    tight = RoomFilterConfig(skip_vlm_echo_candidates=True)
    loose = RoomFilterConfig(skip_vlm_echo_candidates=True, vlm_echo_bbox_tolerance_px=5)
    assert "vlm_echo_skipped" not in prepare_sft_annotation(
        dict(annotation), room_filter=tight).get("drop_attribution", {})
    assert prepare_sft_annotation(
        dict(annotation), room_filter=loose)["drop_attribution"]["vlm_echo_skipped"] == 1


def test_vlm_uncorroborated_area_frac_zero_drops_every_vlm_only_room():
    annotation = _annotation([_room("OFFICE", (100, 100, 250, 250), "vlm_only")])
    kept = prepare_sft_annotation(dict(annotation))
    dropped = prepare_sft_annotation(
        dict(annotation), room_filter=RoomFilterConfig(vlm_uncorroborated_area_frac=0.0)
    )
    assert len(kept["rooms"]) == 1
    assert dropped["rooms"] == []
    assert dropped["drop_attribution"]["vlm_uncorroborated"] == 1
