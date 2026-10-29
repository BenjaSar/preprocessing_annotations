"""PDF vector-wall room extents: ingestion/vector_walls.py,
orchestration/room_extent_step.py, the pipeline wiring and eval scoring.
Synthetic layered PDFs are built with PyMuPDF; no model is loaded."""

import json
import sys
from pathlib import Path

import fitz
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig, RoomExtentConfig, RoomGtEvalConfig
from preprocessing_annotations.ingestion.pdf_extractor import page_source
from preprocessing_annotations.ingestion.vector_walls import VectorWallError, VectorWallReader
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.orchestration.room_extent_step import (
    EXTENT_SOURCE,
    RoomExtentStep,
    assign_extents,
)
from room_extent_eval import evaluate

# 200 x 100 pt page: outer walls + one partition = two rooms, plus a hatch
# stroke on an excluded layer that would otherwise split the left room.
LEFT_ROOM = (10, 10, 100, 90)
RIGHT_ROOM = (100, 10, 190, 90)


def _layered_pdf(path: Path, rotation: int = 0) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    wall, hatch = doc.add_ocg("A-WALL"), doc.add_ocg("A-WALL-PATT")
    page.draw_rect(fitz.Rect(10, 10, 190, 90), oc=wall)
    page.draw_line((100, 10), (100, 90), oc=wall)
    page.draw_line((20, 20), (80, 80), oc=hatch)
    page.set_rotation(rotation)
    doc.save(path)
    return path


def _room(xyxy):
    return {"coordinates": {"bbox": list(xyxy)}}


def _bounds(face):
    return tuple(round(v) for v in face.bounds)


# --- page names ---------------------------------------------------------------

def test_page_source_parses_extracted_page_stems():
    assert page_source("326 ROCKAWAY - AVI-ON LAYOUT_page001") == ("326 ROCKAWAY - AVI-ON LAYOUT", 1)
    assert page_source("scan") is None
    assert page_source("_page001") is None


# --- reader ------------------------------------------------------------------

def test_reader_polygonizes_wall_layers_and_skips_excluded_ones(tmp_path):
    faces = VectorWallReader(RoomExtentConfig()).faces(_layered_pdf(tmp_path / "p.pdf"), 0, (200, 100))
    assert sorted(_bounds(f) for f in faces) == [LEFT_ROOM, RIGHT_ROOM]


def test_reader_scales_to_the_image_and_applies_page_rotation(tmp_path):
    pdf = _layered_pdf(tmp_path / "p.pdf", rotation=90)
    faces = VectorWallReader(RoomExtentConfig()).faces(pdf, 0, (200, 400))   # rendered 2x, portrait
    assert sorted(_bounds(f) for f in faces) == [(20, 20, 180, 200), (20, 200, 180, 380)]


def test_reader_returns_no_faces_without_matching_layers(tmp_path):
    config = RoomExtentConfig(layer_pattern="NO-SUCH-LAYER")
    assert VectorWallReader(config).faces(_layered_pdf(tmp_path / "p.pdf"), 0, (200, 100)) == []


def test_reader_reports_unreadable_pages_as_vector_wall_error(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(VectorWallError):
        VectorWallReader(RoomExtentConfig()).faces(broken, 0, (200, 100))


# --- assignment ------------------------------------------------------------------

def test_assign_extents_takes_the_smallest_containing_face():
    outer, inner = box(0, 0, 100, 100), box(10, 10, 40, 40)
    assert assign_extents([outer, inner], [(20, 20, 30, 30)]) == [inner]


def test_assign_extents_refuses_a_face_shared_by_two_labels():
    merged = box(0, 0, 100, 100)
    single = box(200, 0, 300, 100)
    labels = [(10, 10, 20, 20), (60, 60, 70, 70), (240, 40, 260, 60)]
    assert assign_extents([merged, single], labels) == [None, None, single]


def test_assign_extents_without_faces_or_containing_face():
    assert assign_extents([], [(0, 0, 1, 1)]) == [None]
    assert assign_extents([box(0, 0, 10, 10)], [(50, 50, 60, 60)]) == [None]


# --- step --------------------------------------------------------------------------

def test_step_adds_extent_beside_the_label_box(tmp_path):
    rooms = [_room((40, 40, 60, 50)), _room((140, 40, 160, 50)), _room((500, 500, 510, 510))]
    attached = RoomExtentStep(VectorWallReader(RoomExtentConfig())).attach(
        rooms, _layered_pdf(tmp_path / "p.pdf"), 0, (200, 100)
    )
    assert attached == 2
    assert rooms[0]["coordinates"]["bbox"] == [40, 40, 60, 50]          # label box untouched
    assert rooms[0]["coordinates"]["extent"]["bbox"] == list(LEFT_ROOM)
    assert rooms[0]["coordinates"]["extent"]["source"] == EXTENT_SOURCE
    assert rooms[1]["coordinates"]["extent"]["bbox"] == list(RIGHT_ROOM)
    assert "extent" not in rooms[2]["coordinates"]


# --- pipeline wiring ------------------------------------------------------------------

def test_pipeline_builds_no_extent_step_by_default():
    assert PipelineConfig().use_vector_room_extent is False
    assert AnnotationPipeline(PipelineConfig())._room_extent_step() is None


def test_pipeline_attaches_extents_and_tallies_outcomes(tmp_path):
    from collections import Counter

    _layered_pdf(tmp_path / "plan.pdf")
    config = PipelineConfig()
    config.use_vector_room_extent = True
    pipeline = AnnotationPipeline(config)
    pipeline._input_path = tmp_path
    step = pipeline._room_extent_step()
    counts = Counter()
    annotation = {"image_size": {"width": 200, "height": 100}, "roomsRecognized": [_room((40, 40, 60, 50))]}
    pipeline._attach_room_extents(step, tmp_path / "plan_page000.json", annotation, counts)
    pipeline._attach_room_extents(step, tmp_path / "other_page000.json", dict(annotation), counts)
    assert counts == {"rooms_with_extent": 1, "pages_with_extent": 1, "pages_without_source_pdf": 1}
    assert annotation["roomsRecognized"][0]["coordinates"]["extent"]["bbox"] == list(LEFT_ROOM)


# --- eval -------------------------------------------------------------------------------

def test_eval_scores_extents_against_gt(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _layered_pdf(pdf_dir / "plan.pdf")
    run = tmp_path / "run"
    (run / "processed_annotations").mkdir(parents=True)
    (run / "processed_annotations" / "plan_page000.json").write_text(json.dumps({
        "image_size": {"width": 200, "height": 100},
        "roomsRecognized": [_room((40, 40, 60, 50))],
    }))
    gt_path = tmp_path / "gt.json"
    x1, y1, x2, y2 = LEFT_ROOM
    gt_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "plan_page000.png", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "office"}],
        "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]}],
    }))

    report = evaluate(run, pdf_dir, RoomExtentConfig(), RoomGtEvalConfig(gt_coco_path=gt_path))

    assert report["coverage"] == {"pages": 1, "rooms": 1, "rooms_with_extent": 1, "pages_with_extent": 1}
    assert report["localization"]["label"]["gt_rooms_iou_matched"] == 0
    assert report["localization"]["extent_only"]["gt_rooms_iou_matched"] == 1
    assert report["localization"]["extent_or_label"]["gt_rooms_located"] == 1
    assert report["polygon_iou_matched"] == 1
