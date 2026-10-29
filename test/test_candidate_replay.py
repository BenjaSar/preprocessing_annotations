"""eval/candidate_replay.py: candidate generation replayed from saved raw OCR."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candidate_replay import (
    BASELINE,
    PatternVariant,
    check_recorded_patterns,
    candidate_dicts,
    evaluate_variants,
    extractor_with_patterns,
    load_text_detections,
    with_variant_candidates,
)
from preprocessing_annotations.config import PipelineConfig, RoomGtEvalConfig

LABEL = {"text": "BOOK CLOSET", "bbox": [[100, 100], [300, 100], [300, 130], [100, 130]], "confidence": 0.99}
OFFICE = {"text": "OFFICE", "bbox": [[400, 100], [500, 100], [500, 130], [400, 130]], "confidence": 0.99}
EXTRA = r"^BOOK\s+CLOSET$"   # a form no default pattern matches
PASS2 = {"room_number": "", "room_name": "KITCHEN", "name_expanded": None, "bbox": [900, 900, 50, 40], "confidence": 0.9}


def _annotation(**keys):
    return {"image_file": "docA_page000.png", "image_size": {"width": 2000, "height": 2000},
            "rooms": [], "ocr_rooms": [], **keys}


def _extractors(*extra):
    config = PipelineConfig()
    return (extractor_with_patterns(config, PatternVariant()),
            extractor_with_patterns(config, PatternVariant(added=extra)))


def test_load_text_detections_rejects_run_without_raw_ocr():
    with pytest.raises(ValueError, match="ocrTextDetections"):
        load_text_detections(_annotation())


def test_variant_adds_candidate_baseline_misses():
    baseline, variant = _extractors(EXTRA)
    annotation = _annotation(ocrTextDetections=[LABEL, OFFICE])
    annotation["ocr_rooms"] = candidate_dicts(baseline.find_room_candidates(load_text_detections(annotation)))

    rebuilt = with_variant_candidates(annotation, baseline, variant)

    assert [r["room_name"] for r in annotation["ocr_rooms"]] == ["OFFICE"]
    assert sorted(r["room_name"] for r in rebuilt["ocr_rooms"]) == ["BOOK CLOSET", "OFFICE"]


def test_pass2_candidates_are_kept_after_variant_candidates():
    baseline, variant = _extractors(EXTRA)
    annotation = _annotation(ocrTextDetections=[OFFICE])
    annotation["ocr_rooms"] = candidate_dicts(baseline.find_room_candidates(load_text_detections(annotation))) + [PASS2]

    rebuilt = with_variant_candidates(annotation, baseline, variant)

    assert [r["room_name"] for r in rebuilt["ocr_rooms"]] == ["OFFICE", "KITCHEN"]


def test_baseline_variant_reproduces_the_runs_ocr_rooms():
    baseline, _ = _extractors()
    annotation = _annotation(ocrTextDetections=[LABEL, OFFICE])
    annotation["ocr_rooms"] = candidate_dicts(baseline.find_room_candidates(load_text_detections(annotation))) + [PASS2]

    assert with_variant_candidates(annotation, baseline, baseline)["ocr_rooms"] == annotation["ocr_rooms"]


def test_evaluate_variants_scores_variant_against_baseline(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "docA_page000.png"}],
        "categories": [{"id": 1, "name": "janitor_closet"}],
        "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[90, 90, 320, 90, 320, 140, 90, 140]]}],
    }))
    run = tmp_path / "run"
    (run / "annotations").mkdir(parents=True)
    (run / "annotations" / "docA_page000.json").write_text(json.dumps(_annotation(ocrTextDetections=[LABEL])))

    report = evaluate_variants(
        RoomGtEvalConfig(gt_coco_path=gt_path), run, PipelineConfig(),
        {"closet": PatternVariant(added=(EXTRA,))},
    )

    assert report["candidates"][BASELINE]["localization"]["gt_rooms_located"] == 0
    assert report["candidates"]["closet"]["localization"]["gt_rooms_located"] == 1
    assert report["candidates"]["closet"]["documents_with_recall_drop"] == []
    assert report["candidates"]["closet"]["room_count_changes"] == {"doca_page000": [0, 1]}
    # one document: nothing to train on, so the fold falls back to the baseline
    assert [f["chosen"] for f in report["leave_one_document_out"]["folds"]] == [BASELINE]


def test_refuses_when_code_patterns_generate_candidates_the_run_lacks():
    baseline, variant = _extractors(EXTRA)           # "baseline" here is a changed pattern set
    annotation = _annotation(ocrTextDetections=[LABEL])   # run recorded no candidates
    with pytest.raises(ValueError, match="patterns differ"):
        with_variant_candidates(annotation, variant, variant)


def test_recorded_patterns_confirm_or_fall_back(tmp_path):
    config = PipelineConfig()
    assert check_recorded_patterns(tmp_path, config) is False      # no snapshot
    (tmp_path / "pipeline_config.json").write_text(json.dumps({"ocr": {}}))
    assert check_recorded_patterns(tmp_path, config) is False      # pre-2026-10-13 run
    (tmp_path / "pipeline_config.json").write_text(json.dumps(
        {"ocr": {"room_name_patterns": list(config.ocr.room_name_patterns)}}
    ))
    assert check_recorded_patterns(tmp_path, config) is True


def test_subset_guard_can_be_waived_when_patterns_are_verified():
    baseline, variant = _extractors(EXTRA)
    annotation = _annotation(ocrTextDetections=[LABEL])    # run recorded nothing
    with pytest.raises(ValueError, match="patterns differ"):
        with_variant_candidates(annotation, variant, variant)
    waived = with_variant_candidates(
        annotation, variant, variant, verify_against_recorded=False
    )
    assert [r["room_name"] for r in waived["ocr_rooms"]] == ["BOOK CLOSET"]


def test_recorded_patterns_must_equal_the_codes(tmp_path):
    config = PipelineConfig()
    (tmp_path / "pipeline_config.json").write_text(json.dumps(
        {"ocr": {"room_name_patterns": list(config.ocr.room_name_patterns) + [EXTRA]}}
    ))
    with pytest.raises(ValueError, match="differ"):
        check_recorded_patterns(tmp_path, config)




def test_removing_a_default_pattern_drops_its_candidates():
    config = PipelineConfig()
    janitor_pattern = next(p for p in config.ocr.room_name_patterns if "JANITORS?" in p)
    detection = load_text_detections(_annotation(ocrTextDetections=[
        {"text": "JANITORS CLOSET", "bbox": [[100, 100], [300, 100], [300, 130], [100, 130]], "confidence": 0.99},
    ]))
    without = extractor_with_patterns(config, PatternVariant(removed=(janitor_pattern,)))
    assert extractor_with_patterns(config, PatternVariant()).find_room_candidates(detection)
    assert without.find_room_candidates(detection) == []


def test_removing_an_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="cannot remove"):
        extractor_with_patterns(PipelineConfig(), PatternVariant(removed=("^NOT_A_PATTERN$",)))


def test_adding_a_pattern_already_in_code_is_rejected():
    """The trap this closes: a variant adding a pattern the code already has
    equals the baseline, so the report says "no change" for a pattern never
    actually tested. A working pattern was once measured inert and reverted
    that way."""
    config = PipelineConfig()
    present = config.ocr.room_name_patterns[0]
    with pytest.raises(ValueError, match="already in"):
        extractor_with_patterns(config, PatternVariant(added=(present,)))


def _janitor_run(tmp_path):
    """A run made before the JANITORS CLOSET pattern: it recorded no candidate."""
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "docA_page000.png"}],
        "categories": [{"id": 1, "name": "janitor_closet"}],
        "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[90, 90, 320, 90, 320, 140, 90, 140]]}],
    }))
    run = tmp_path / "run"
    (run / "annotations").mkdir(parents=True)
    janitor = {"text": "JANITORS CLOSET", "bbox": LABEL["bbox"], "confidence": 0.99}
    (run / "annotations" / "docA_page000.json").write_text(json.dumps(_annotation(ocrTextDetections=[janitor])))
    pattern = next(p for p in PipelineConfig().ocr.room_name_patterns if "JANITORS?" in p)
    return RoomGtEvalConfig(gt_coco_path=gt_path), run, pattern


def test_run_lacks_replays_an_older_run_on_the_codes_patterns(tmp_path):
    eval_config, run, pattern = _janitor_run(tmp_path)
    with pytest.raises(ValueError, match="patterns differ"):
        evaluate_variants(eval_config, run, PipelineConfig(), {})

    report = evaluate_variants(eval_config, run, PipelineConfig(), {}, run_lacks=(pattern,))

    assert report["run_lacks_patterns"] == [pattern]
    assert report["candidates"][BASELINE]["localization"]["gt_rooms_located"] == 1


def test_recorded_patterns_compare_against_code_minus_run_lacks(tmp_path):
    config = PipelineConfig()
    lacked = config.ocr.room_name_patterns[-1]
    (tmp_path / "pipeline_config.json").write_text(json.dumps(
        {"ocr": {"room_name_patterns": list(config.ocr.room_name_patterns[:-1])}}
    ))
    assert check_recorded_patterns(tmp_path, config, (lacked,)) is True
    with pytest.raises(ValueError, match="differ"):
        check_recorded_patterns(tmp_path, config)
