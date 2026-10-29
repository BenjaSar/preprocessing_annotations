"""H2 (2026-10-11): eval/room_replay.py on synthetic fixtures."""

import json
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig, RoomFilterConfig, RoomGtEvalConfig
from room_gt_eval import LocalizationScore
from replay_fixtures import write_run_and_gt
from room_gt_eval import PredictedRoom
from room_replay import (
    document_key,
    evaluate_replay,
    parse_overrides,
    regressed_documents,
    replay_page,
    room_count_changes,
    score_replay,
)


def test_document_key_strips_page_index():
    assert document_key("326_rockaway_avi_on_layout_page012") == "326_rockaway_avi_on_layout"


def test_parse_overrides_types_values_after_defaults():
    parsed = parse_overrides(
        RoomFilterConfig(), ["merge_proximity_px=0", "max_room_frac=0.5"]
    )
    assert parsed.merge_proximity_px == 0 and isinstance(parsed.merge_proximity_px, int)
    assert parsed.max_room_frac == 0.5
    assert parsed.room_min_area == RoomFilterConfig().room_min_area


def test_parse_overrides_parses_bools_strictly():
    assert parse_overrides(
        RoomFilterConfig(), ["skip_vlm_echo_candidates=true"]
    ).skip_vlm_echo_candidates is True
    assert parse_overrides(
        RoomFilterConfig(skip_vlm_echo_candidates=True), ["skip_vlm_echo_candidates=false"]
    ).skip_vlm_echo_candidates is False
    with pytest.raises(ValueError, match="bad bool"):
        parse_overrides(RoomFilterConfig(), ["skip_vlm_echo_candidates=1"])


def test_room_count_changes_lists_only_differing_pages():
    box_ = (0, 0, 1, 1)
    assert room_count_changes(
        {"a": [box_], "b": [box_, box_], "c": []}, {"a": [box_], "b": [box_], "c": []}
    ) == {"b": [2, 1]}


@pytest.mark.parametrize("bad", ["nope=1", "merge_proximity_px", "merge_proximity_px="])
def test_parse_overrides_rejects_bad_override(bad):
    with pytest.raises(ValueError, match="bad override"):
        parse_overrides(RoomFilterConfig(), [bad])


def test_replay_page_applies_room_filter_and_returns_xyxy_and_drops(tmp_path):
    annotation = {
        "image_file": "p.png",
        "image_size": {"width": 2000, "height": 2000},
        "rooms": [{"room_name": "OFFICE", "bbox": [100, 100, 250, 250],
                   "source": "vlm_ocr_merged", "confidence": 0.9}],
        "ocr_rooms": [],
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(annotation))
    config = PipelineConfig()

    rooms, drops = replay_page(path, tmp_path, config, config.room_filter)
    assert [r.box for r in rooms] == [(100, 100, 350, 350)] and drops == {}
    assert rooms[0].room_type, "replayed rooms must carry their class, not just a box"

    strict = RoomFilterConfig(max_room_frac=0.001)
    rooms, drops = replay_page(path, tmp_path, config, strict)
    assert rooms == [] and drops == {"geometric": 1}


def test_score_replay_splits_per_document():
    gt = {"docA_page000": [box(0, 0, 100, 100)], "docB_page000": [box(0, 0, 100, 100)]}
    rooms = {
        "docA_page000": [PredictedRoom((40, 40, 60, 60), "PRIVATE OFFICE")],
        "docB_page000": [PredictedRoom((500, 500, 510, 510), "PRIVATE OFFICE")],
    }

    overall, per_document = score_replay(rooms, gt, 0.5)

    assert overall.gt_rooms_located == 1
    assert per_document["docA"].gt_rooms_located == 1
    assert per_document["docB"].gt_rooms_located == 0


def test_regressed_documents_lists_only_documents_that_lost_gt_rooms():
    def score(located):
        return LocalizationScore(gt_rooms_located=located)

    baseline = {"a": score(5), "b": score(5), "c": score(5)}
    variant = {"a": score(6), "b": score(4), "c": score(5)}

    assert regressed_documents(baseline, variant) == ["b"]


def test_evaluate_replay_reports_as_run_baseline_variant_and_regressions(tmp_path):
    gt_path, run = write_run_and_gt(tmp_path)
    config = PipelineConfig()
    variant = RoomFilterConfig(max_room_frac=0.001)

    report = evaluate_replay(RoomGtEvalConfig(gt_coco_path=gt_path), run, config, variant)

    assert report["pages_scored"] == 1 and report["gt_pages_missing_from_run"] == []
    assert report["as_run"]["localization"]["gt_rooms_located"] == 1
    assert report["baseline"]["localization"] == report["as_run"]["localization"]
    assert report["variant"]["localization"]["predictions"] == 0
    assert report["variant"]["drop_attribution"] == {"geometric": 1}
    assert report["documents_with_recall_drop"] == ["doca"]
    assert report["room_count_changes"] == {"doca_page000": [1, 0]}
