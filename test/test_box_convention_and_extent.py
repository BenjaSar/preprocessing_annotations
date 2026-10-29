"""Oversize fix, part 1 (2026-10-14): make box size measurable and the box
convention explicit.

Two defects this locks down:
  - the run reported bbox IoU only, and IoU is undirected -- a label-scale
    box and a box that swallows its neighbours both score 0, so an oversize
    regression could not be seen
  - rooms carried two different box conventions in one list with nothing
    saying which was which (measured on the room GT, median box area over
    GT room area: ocr_only 0.013, vlm_only 0.348)
"""

import json
import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from preprocessing_annotations.automation import prepare_sft_annotation  # noqa: E402
from preprocessing_annotations.automation.annotation_schema import (  # noqa: E402
    LABEL_BOX,
    ROOM_EXTENT_BOX,
    box_convention_of,
)
from preprocessing_annotations.export.exporters import COCOExporter  # noqa: E402
from room_gt_eval import ExtentScore, score_extent  # noqa: E402


def _square(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


ROOM = _square(0, 0, 100, 100)  # area 10_000


class TestBoxConvention:
    def test_ocr_anchored_sources_are_label_boxes(self):
        for source in ("ocr_only", "vlm_ocr_merged", "ocr_abbreviation_recovered"):
            assert box_convention_of(source) == LABEL_BOX

    def test_vlm_only_is_a_room_extent_box(self):
        assert box_convention_of("vlm_only") == ROOM_EXTENT_BOX

    def test_unknown_source_gets_no_convention_rather_than_a_guess(self):
        # A new source must surface as unlabelled, not inherit a wrong
        # training target by default.
        assert box_convention_of("some_future_source") is None


def _annotation(rooms):
    return {
        "image_file": "test.png",
        "image_size": {"width": 2000, "height": 2000},
        "rooms": rooms,
        "ocr_rooms": [],
        "sft_ready": True,
    }


def _room(name, bbox, source):
    return {"room_name": name, "bbox": list(bbox), "source": source,
            "confidence": 0.9, "content_density": 0.15, "type": "OFFICE"}


class TestConventionIsStamped:
    """prepare_sft_annotation stamps every surviving room, so consumers read
    a field instead of each repeating the source -> convention mapping."""

    def test_surviving_rooms_carry_their_convention(self):
        processed = prepare_sft_annotation(_annotation([
            _room("OFFICE", (100, 100, 250, 250), "vlm_only"),
            _room("STORAGE", (800, 800, 250, 250), "ocr_only"),
        ]))
        carried = {r["room_name"]: r["box_convention"] for r in processed["rooms"]}
        assert carried == {"OFFICE": ROOM_EXTENT_BOX, "STORAGE": LABEL_BOX}

    def test_coco_export_carries_the_convention(self, tmp_path):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        processed = prepare_sft_annotation(_annotation([
            _room("OFFICE", (100, 100, 250, 250), "vlm_only"),
            _room("STORAGE", (800, 800, 250, 250), "ocr_only"),
        ]))
        (processed_dir / "page000.json").write_text(json.dumps(processed))

        out = tmp_path / "coco.json"
        COCOExporter().export(
            processed_dir, tmp_path, out, min_annotations_per_image=1,
        )
        coco = json.loads(out.read_text())
        carried = {
            a["attributes"]["room_name"]: a["attributes"]["box_convention"]
            for a in coco["annotations"]
        }
        assert carried == {"OFFICE": ROOM_EXTENT_BOX, "STORAGE": LABEL_BOX}


class TestScoreExtent:
    def test_label_scale_box_is_not_oversize(self):
        score = ExtentScore()
        score_extent([(40.0, 40.0, 50.0, 50.0)], [ROOM], 1.0, score)
        assert (score.located, score.oversize) == (1, 0)
        assert score.area_ratios == [pytest.approx(0.01)]

    def test_box_larger_than_its_room_is_oversize(self):
        score = ExtentScore()
        score_extent([(10.0, 10.0, 160.0, 160.0)], [ROOM], 1.0, score)
        assert (score.located, score.oversize) == (1, 1)

    def test_iou_cannot_separate_the_two_but_area_ratio_can(self):
        """Both boxes score IoU ~0 against the room; only the ratio tells
        the too-small one from the too-large one. This is the whole reason
        the measure exists."""
        score = ExtentScore()
        score_extent(
            [(49.0, 49.0, 51.0, 51.0), (-400.0, -400.0, 500.0, 500.0)],
            [ROOM], 1.0, score,
        )
        assert score.oversize == 1
        assert min(score.area_ratios) < 1.0 < max(score.area_ratios)

    def test_room_outside_every_gt_polygon_is_not_scored(self):
        # A room in the wrong place has no GT room whose size it could be
        # judged against; counting it would dilute the oversize rate.
        score = ExtentScore()
        score_extent([(500.0, 500.0, 600.0, 600.0)], [ROOM], 1.0, score)
        assert (score.located, score.oversize) == (0, 0)

    def test_empty_page_reports_zeroes_not_a_division_error(self):
        assert ExtentScore().to_dict()["oversize_rate"] == 0.0
