"""C2: category scoring (eval/room_gt_eval.py).

A room is scored on class only where it is scored on position: its centre
must lie in a GT polygon, since a room in the wrong place has no GT class to
be judged against. Categories the reviewed map leaves unmapped are excluded
and counted, never silently counted as right.
"""

import json
import sys
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import RoomGtEvalConfig
from room_gt_eval import (
    CategoryScore,
    PredictedRoom,
    boxes_of,
    load_gt_category_map,
    load_run_rooms,
    score_category,
)

GTS = [box(0, 0, 100, 100), box(200, 0, 300, 100)]
CATEGORIES = ["toilet", "choir"]
MAP = {"toilet": "TOILET", "choir": None}


def _room(x, room_type):
    return PredictedRoom((x, 40, x + 20, 60), room_type)


def test_correct_class_counts_as_correct():
    score = CategoryScore()
    score_category([_room(40, "TOILET")], GTS, CATEGORIES, MAP, score)
    assert (score.correct, score.wrong) == (1, 0)
    assert score.to_dict()["accuracy"] == 1.0


def test_wrong_class_is_recorded_with_its_confusion():
    score = CategoryScore()
    score_category([_room(40, "STORAGE ROOM")], GTS, CATEGORIES, MAP, score)
    assert (score.correct, score.wrong) == (0, 1)
    assert score.to_dict()["top_confusions"] == [
        {"gt_class": "TOILET", "predicted": "STORAGE ROOM", "rooms": 1}
    ]


def test_unmapped_category_is_excluded_not_counted_right():
    score = CategoryScore()
    score_category([_room(240, "ANYTHING")], GTS, CATEGORIES, MAP, score)
    assert (score.correct, score.wrong, score.excluded_unmapped) == (0, 0, 1)


def test_room_without_a_class_is_excluded():
    score = CategoryScore()
    score_category([_room(40, None)], GTS, CATEGORIES, MAP, score)
    assert (score.correct, score.wrong, score.excluded_untyped) == (0, 0, 1)


def test_room_outside_every_gt_polygon_is_not_scored():
    score = CategoryScore()
    score_category([_room(900, "TOILET")], GTS, CATEGORIES, MAP, score)
    assert (score.correct, score.wrong, score.excluded_unmapped) == (0, 0, 0)


def test_accuracy_is_zero_when_nothing_was_scored():
    assert CategoryScore().to_dict()["accuracy"] == 0.0


def test_loaded_rooms_carry_their_class(tmp_path):
    processed = tmp_path / "processed_annotations"
    processed.mkdir()
    (processed / "docA_page000.json").write_text(json.dumps({
        "roomsRecognized": [{"type": "TOILET", "coordinates": {"bbox": [1, 2, 3, 4]}}]
    }))

    rooms = load_run_rooms(tmp_path)["doca_page000"]

    assert rooms == [PredictedRoom((1, 2, 3, 4), "TOILET")]
    assert boxes_of(rooms) == [(1, 2, 3, 4)]


def test_shipped_category_map_is_loadable_and_complete():
    category_map = load_gt_category_map(RoomGtEvalConfig().gt_category_map_path)
    assert category_map["mapping"], "map must not be empty"
    assert "reviewed" in category_map
