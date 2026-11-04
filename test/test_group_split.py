"""Group-aware split (export/group_split.py): groups never straddle splits,
pinned groups land in test, and assignment is deterministic."""

import json

import pytest

from preprocessing_annotations.export.group_split import (
    SPLIT_NAMES,
    TEST,
    GroupMap,
    assign_groups,
    document_key,
    split_items,
)

RATIOS = (0.70, 0.15, 0.15)


def test_document_key_strips_page_index_and_extension():
    assert document_key("BP-1 Kennedy_page003.json") == "BP-1 Kennedy"
    assert document_key("326 ROCKAWAY_page012.png") == "326 ROCKAWAY"


def test_group_map_defaults_to_one_group_per_document():
    assert GroupMap().group_of("docA_page001.json") == "docA"


def test_group_map_reads_family_and_pins(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({
        "groups": {"masonic_avi": "masonic_violet", "violet_avi": "masonic_violet"},
        "pinned_test_groups": ["masonic_violet"],
    }))

    group_map = GroupMap.load(path)

    assert group_map.group_of("masonic_avi_page000.json") == "masonic_violet"
    assert group_map.group_of("violet_avi_page001.json") == "masonic_violet"
    assert group_map.group_of("other_page000.json") == "other"
    assert group_map.pinned_test_groups == frozenset({"masonic_violet"})


def test_load_without_path_is_the_default_map():
    assert GroupMap.load(None).groups == {}


def test_missing_map_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_split_group_map"):
        GroupMap.load(tmp_path / "absent.json")


def test_configured_default_map_pins_the_gt_documents():
    from preprocessing_annotations.config import PipelineConfig

    group_map = GroupMap.load(PipelineConfig().export.split_group_map_path)

    assert group_map.pinned_test_groups, "shipped map must pin the room-GT documents"
    assert group_map.group_of("326 ROCKAWAY - AVI-ON LAYOUT_page000.json") in group_map.pinned_test_groups


def test_every_group_lands_in_exactly_one_split():
    sizes = {f"g{i}": 1 + i % 4 for i in range(12)}
    assignment = assign_groups(sizes, RATIOS, seed=42)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) <= set(SPLIT_NAMES)


def test_pinned_groups_go_to_test():
    sizes = {"a": 5, "b": 5, "c": 5, "gt": 5}
    assignment = assign_groups(sizes, RATIOS, seed=42, pinned_test_groups=["gt"])
    assert assignment["gt"] == TEST


def test_pinned_group_absent_from_the_data_is_ignored():
    assignment = assign_groups({"a": 2, "b": 2}, RATIOS, seed=1, pinned_test_groups=["ghost"])
    assert "ghost" not in assignment


def test_assignment_is_deterministic_for_a_seed():
    sizes = {f"g{i}": 1 + i % 5 for i in range(15)}
    assert assign_groups(sizes, RATIOS, seed=7) == assign_groups(sizes, RATIOS, seed=7)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_groups({"a": 1}, (0.5, 0.2, 0.2), seed=1)


def test_item_split_follows_its_group():
    item_groups = {f"doc{d}_page{p}.json": f"doc{d}" for d in range(6) for p in range(3)}
    split_of = split_items(item_groups, RATIOS, seed=42)
    for document in {f"doc{d}" for d in range(6)}:
        pages = {s for item, s in split_of.items() if item.startswith(document + "_")}
        assert len(pages) == 1, f"{document} straddles splits"


def test_larger_split_gets_more_items_than_the_smaller_ones():
    item_groups = {f"doc{d}_page{p}.json": f"doc{d}" for d in range(10) for p in range(4)}
    split_of = split_items(item_groups, RATIOS, seed=42)
    counts = {name: sum(s == name for s in split_of.values()) for name in SPLIT_NAMES}
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]
    assert sum(counts.values()) == len(item_groups)


def test_empty_split_is_warned_not_silent(caplog):
    with caplog.at_level("WARNING"):
        assign_groups({"only": 10}, RATIOS, seed=1)
    assert "empty" in caplog.text
