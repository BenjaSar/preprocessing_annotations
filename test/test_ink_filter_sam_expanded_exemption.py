#!/usr/bin/env python3
"""
A1 (2026-08-21): _filter_low_ink_rooms's exempt_sam_expanded param.

Confirmed gap (3 pages / 7 visually-verified sam_expanded boxes): ink
density has no discriminative power for sam_expanded rooms -- correct
(0.0053-0.31) and wrong (0.0869-0.12) ranges overlap. This flag skips the
ink check for sam_expanded rooms only.

Contract this file locks down:
  1. exempt_sam_expanded=False (default) -> byte-identical to pre-A1
     behavior, including for sam_expanded rooms.
  2. exempt_sam_expanded=True + sam_expanded room below min_frac -> kept,
     not dropped.
  3. exempt_sam_expanded=True + non-expanded room below min_frac -> still
     dropped. FIX-5's real, measured-effective use must be untouched.
"""


from PIL import Image

from preprocessing_annotations.automation.sft_validator import _filter_low_ink_rooms


def _blank_image(w=200, h=200):
    return Image.new("L", (w, h), color=255)  # all white -> ink=0.0 everywhere


def _room(name, bbox, sam_expanded=False):
    r = {"room_name": name, "bbox": bbox}
    if sam_expanded:
        r["sam_expanded"] = True
    return r


def test_default_false_drops_sam_expanded_low_ink_room():
    gray = _blank_image()
    rooms = [_room("CLASSROOM", [10, 10, 50, 50], sam_expanded=True)]

    kept, dropped = _filter_low_ink_rooms(rooms, gray, min_frac=0.09, binarise_max=200)

    assert dropped == 1
    assert kept == []


def test_exempt_true_keeps_sam_expanded_low_ink_room():
    gray = _blank_image()
    rooms = [_room("CLASSROOM", [10, 10, 50, 50], sam_expanded=True)]

    kept, dropped = _filter_low_ink_rooms(
        rooms, gray, min_frac=0.09, binarise_max=200, exempt_sam_expanded=True
    )

    assert dropped == 0
    assert kept == rooms


def test_exempt_true_still_drops_non_expanded_low_ink_room():
    """FIX-5's real, measured-effective use (blank label-scale boxes) must
    stay untouched by this flag."""
    gray = _blank_image()
    rooms = [_room("JUNK", [10, 10, 20, 10])]

    kept, dropped = _filter_low_ink_rooms(
        rooms, gray, min_frac=0.09, binarise_max=200, exempt_sam_expanded=True
    )

    assert dropped == 1
    assert kept == []


def test_exempt_true_mixed_rooms_only_expanded_survives():
    gray = _blank_image()
    rooms = [
        _room("JUNK", [10, 10, 20, 10]),
        _room("CLASSROOM", [50, 50, 50, 50], sam_expanded=True),
    ]

    kept, dropped = _filter_low_ink_rooms(
        rooms, gray, min_frac=0.09, binarise_max=200, exempt_sam_expanded=True
    )

    assert dropped == 1
    assert [r["room_name"] for r in kept] == ["CLASSROOM"]


def test_exempt_false_is_byte_identical_regardless_of_sam_expanded_flag():
    gray = _blank_image()
    rooms_a = [_room("CLASSROOM", [10, 10, 50, 50], sam_expanded=True)]
    rooms_b = [_room("CLASSROOM", [10, 10, 50, 50], sam_expanded=False)]

    result_a = _filter_low_ink_rooms(rooms_a, gray, min_frac=0.09, binarise_max=200)
    result_b = _filter_low_ink_rooms(rooms_b, gray, min_frac=0.09, binarise_max=200)

    assert result_a[1] == result_b[1] == 1
