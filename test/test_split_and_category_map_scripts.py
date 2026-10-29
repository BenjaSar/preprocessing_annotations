"""scripts/make_split_group_map.py and scripts/draft_gt_category_map.py:
the map-building logic behind the two human-reviewed artifacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collections import Counter

from draft_gt_category_map import draft_mapping
from make_split_group_map import build_group_map
from preprocessing_annotations.automation.taxonomy import normalize_to_mandatory

FALLBACK = normalize_to_mandatory("XYZZY QWERTY UNMAPPABLE")


def _gt_file(tmp_path, file_names):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": name} for i, name in enumerate(file_names)],
        "categories": [{"id": 1, "name": "office"}],
        "annotations": [
            {"image_id": i, "category_id": 1, "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]}
            for i in range(len(file_names))
        ],
    }))
    return path


def test_group_map_pins_only_documents_the_gt_annotates(tmp_path):
    run = tmp_path / "run"
    (run / "annotations").mkdir(parents=True)
    for stem in ("326 ROCKAWAY - AVI-ON LAYOUT_page000", "Untouched Doc_page000"):
        (run / "annotations" / f"{stem}.json").write_text("{}")
    # Label Studio rewrites punctuation and prefixes an upload id
    gt = _gt_file(tmp_path, ["a1b2c3d4-326_ROCKAWAY_-_AVI-ON_LAYOUT_page000.png"])

    group_map = build_group_map(run, gt)

    assert group_map["pinned_test_groups"] == ["326 ROCKAWAY - AVI-ON LAYOUT"]
    assert group_map["groups"] == {
        "326 ROCKAWAY - AVI-ON LAYOUT": "326 ROCKAWAY - AVI-ON LAYOUT",
        "Untouched Doc": "Untouched Doc",
    }


def test_category_draft_resolves_known_names_and_withholds_guesses():
    draft = draft_mapping(Counter({"toilet": 10, "choir": 2}))

    assert draft["mapping"]["toilet"] == normalize_to_mandatory("TOILET")
    assert draft["mapping"]["choir"] is None
    assert [e["gt_category"] for e in draft["needs_review"]] == ["choir"]
    assert draft["reviewed"] is False


def test_inferred_mapping_is_withheld_not_trusted():
    """A category the taxonomy does not declare must be withheld, however
    confident the resolver sounds -- that inference is where it conflates
    distinct kinds of room (`library` reached RESIDENTIAL UNIT that way before
    LIBRARY became a class)."""
    category = "sciences_classroom"          # resolved only by substring
    draft = draft_mapping(Counter({category: 15}))

    assert draft["mapping"][category] is None
    assert draft["needs_review"][0]["resolver_would_say"] == normalize_to_mandatory(
        category.replace("_", " ").upper()
    )


def test_category_whose_own_name_is_the_fallback_class_is_kept():
    draft = draft_mapping(Counter({FALLBACK.lower().replace(" ", "_"): 5}))

    assert draft["mapping"][FALLBACK.lower().replace(" ", "_")] == FALLBACK
    assert draft["needs_review"] == []


def test_review_queue_is_ordered_by_room_count():
    # names chosen to resolve to nothing, so the queue holds all three
    draft = draft_mapping(Counter({"xyzzy_a": 2, "xyzzy_b": 7, "xyzzy_c": 4}))

    assert [e["gt_rooms"] for e in draft["needs_review"]] == [7, 4, 2]
