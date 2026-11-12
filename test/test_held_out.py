"""Leave-one-document-out selector (eval/held_out.py) and the room_replay sweep."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from held_out import leave_one_document_out
from room_gt_eval import LocalizationScore
from room_replay import evaluate_sweep
from replay_fixtures import write_run_and_gt

from preprocessing_annotations.config import PipelineConfig, RoomGtEvalConfig


def _score(located, predictions, in_gt):
    return LocalizationScore(
        gt_rooms_located=located, predictions=predictions, predictions_center_in_gt=in_gt
    )


def test_picks_candidate_that_locates_more_without_precision_loss():
    scores = {
        100: {"a": _score(5, 10, 9), "b": _score(5, 10, 9), "c": _score(5, 10, 9)},
        0: {"a": _score(7, 10, 9), "b": _score(7, 10, 9), "c": _score(7, 10, 9)},
    }
    result = leave_one_document_out(scores, default=100)
    assert [f.chosen for f in result.folds] == [0, 0, 0]
    assert result.located == 21 and result.default_located == 15


def test_rejects_candidate_whose_training_precision_is_below_default():
    scores = {
        100: {"a": _score(5, 10, 9), "b": _score(5, 10, 9)},
        0: {"a": _score(9, 20, 12), "b": _score(9, 20, 12)},
    }
    result = leave_one_document_out(scores, default=100)
    assert [f.chosen for f in result.folds] == [100, 100]
    assert result.located == result.default_located == 10


def test_tie_goes_to_earlier_candidate():
    scores = {
        100: {"a": _score(5, 10, 9), "b": _score(5, 10, 9)},
        50: {"a": _score(5, 10, 9), "b": _score(5, 10, 9)},
    }
    assert [f.chosen for f in leave_one_document_out(scores, default=100).folds] == [100, 100]


def test_choice_for_a_fold_ignores_the_held_out_document():
    # 0 only looks better because of document c; when c is held out it is
    # compared on a and b, where it is worse, so c is scored with the default.
    scores = {
        100: {"a": _score(5, 10, 9), "b": _score(5, 10, 9), "c": _score(5, 10, 9)},
        0: {"a": _score(4, 10, 9), "b": _score(4, 10, 9), "c": _score(20, 10, 9)},
    }
    folds = {f.held_out: f.chosen for f in leave_one_document_out(scores, default=100).folds}
    assert folds["c"] == 100 and folds["a"] == 0 and folds["b"] == 0


def test_to_dict_reports_precision():
    scores = {100: {"a": _score(5, 10, 9)}}
    out = leave_one_document_out(scores, default=100).to_dict()
    assert out["held_out"]["label_precision"] == 0.9
    assert out["default"]["gt_rooms_located"] == 5


def test_evaluate_sweep_replays_candidates_and_includes_default(tmp_path):
    gt_path, run = write_run_and_gt(tmp_path)

    report = evaluate_sweep(
        RoomGtEvalConfig(gt_coco_path=gt_path), run, PipelineConfig(),
        "max_room_frac=0.001,0.3",
    )

    assert report["field"] == "max_room_frac" and report["default"] == 0.3
    assert set(report["candidates"]) == {"0.3", "0.001"}
    assert report["candidates"]["0.001"]["localization"]["predictions"] == 0
    assert report["candidates"]["0.001"]["documents_with_recall_drop"] == ["doca"]
    assert report["leave_one_document_out"]["default"]["gt_rooms_located"] == 1


def test_evaluate_sweep_rejects_unknown_field(tmp_path):
    gt_path, run = write_run_and_gt(tmp_path)
    with pytest.raises(ValueError, match="bad override"):
        evaluate_sweep(
            RoomGtEvalConfig(gt_coco_path=gt_path), run, PipelineConfig(), "nope=1,2"
        )
