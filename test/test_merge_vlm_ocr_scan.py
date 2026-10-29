"""_merge_vlm_and_ocr matches OCR rooms against VLM rooms only, and against
the nearest proximal one.

The scan used to walk `merged`, which the same loop appends OCR rooms to, so
two neighbouring OCR labels could merge into each other. That stayed hidden
while the VLM emitted room-scale boxes (one always matched first); the
label-scale boxes of an SFT-tuned model exposed it by leaving no VLM match at
all. See the measured case in the function's own comment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.sft_validator import _merge_vlm_and_ocr

PROXIMITY = 100


def _room(name, bbox):
    return {"room_name": name, "name": name, "bbox": bbox}


def _names(rooms):
    return [(r.get("room_name"), r.get("source")) for r in rooms]


def test_neighbouring_ocr_labels_do_not_merge_into_each_other():
    """Four 16px labels 58px apart, no VLM room: all four must survive."""
    ocr = [_room("TOILET", [2523, 1125, 39, 16]), _room("TOILET", [2620, 1126, 40, 16]),
           _room("TOILET", [2713, 1126, 40, 16]), _room("TOILET", [2810, 1126, 41, 16])]

    merged = _merge_vlm_and_ocr([], ocr, PROXIMITY).rooms

    assert _names(merged) == [("TOILET", "ocr_only")] * 4


def test_identical_adjacent_labels_survive_even_with_a_distant_vlm_room():
    """A VLM room elsewhere must not change how the labels relate to each other."""
    vlm = [_room("CLASSROOM 12", [100, 100, 250, 280])]
    ocr = [_room("TOILET", [2523, 1125, 39, 16]), _room("TOILET", [2620, 1126, 40, 16])]

    merged = _merge_vlm_and_ocr(vlm, ocr, PROXIMITY).rooms

    assert _names(merged) == [("CLASSROOM 12", "vlm_only"),
                              ("TOILET", "ocr_only"), ("TOILET", "ocr_only")]


def test_ocr_label_still_corroborates_a_proximal_vlm_room():
    """The real merge still happens: same label, same place -> vlm_ocr_merged."""
    vlm = [_room("OFFICE", [500, 500, 200, 120])]
    ocr = [_room("OFFICE", [540, 540, 60, 16])]

    merged = _merge_vlm_and_ocr(vlm, ocr, PROXIMITY).rooms

    assert _names(merged) == [("OFFICE", "vlm_ocr_merged")]


def test_distinct_label_inside_a_vlm_room_is_kept_separately():
    """FM-1 behaviour is unchanged."""
    vlm = [_room("CLASSROOM 12", [2579, 750, 252, 289])]
    ocr = [_room("TOILET", [2620, 1126, 40, 16])]

    merged = _merge_vlm_and_ocr(vlm, ocr, PROXIMITY).rooms

    assert _names(merged) == [("CLASSROOM 12", "vlm_only"), ("TOILET", "ocr_only")]


def test_the_nearest_proximal_vlm_room_wins_not_the_first_listed():
    """Two proximal VLM boxes: list order is arbitrary, distance is not."""
    far = _room("FAR ROOM", [300, 1100, 60, 40])
    near = _room("NEAR ROOM", [520, 1120, 60, 40])
    ocr = [_room("STORAGE ROOM", [530, 1125, 50, 16])]

    merged = _merge_vlm_and_ocr([far, near], ocr, PROXIMITY).rooms

    # The nearest box is the one that takes the OCR geometry and label.
    adopted = [r for r in merged if r.get("source") == "vlm_ocr_merged"]
    assert len(adopted) == 1
    assert adopted[0]["vlm_bbox"] == near["bbox"]


def test_an_ocr_room_far_from_every_vlm_room_is_added_alone():
    vlm = [_room("OFFICE", [0, 0, 100, 100])]
    ocr = [_room("TOILET", [3000, 3000, 40, 16])]

    merged = _merge_vlm_and_ocr(vlm, ocr, PROXIMITY).rooms

    assert _names(merged) == [("OFFICE", "vlm_only"), ("TOILET", "ocr_only")]
