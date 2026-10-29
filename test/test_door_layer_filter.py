"""Door detections on wiring: detection/door_layer_filter.py and its pipeline
wiring. Synthetic layered PDFs are built with PyMuPDF; no model is loaded."""

import sys
from pathlib import Path

import fitz
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import DoorLayerFilterConfig, PipelineConfig
from preprocessing_annotations.detection.door_layer_filter import DoorLayerFilter
from preprocessing_annotations.detection.window_detector import WindowDetection
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline

# 200 x 100 pt page rendered at 2x (400 x 200 px): a door at (20, 20, 60, 60) px,
# a wiring arc at (100, 80, 160, 100) px.
IMAGE_SIZE = (400, 200)
ON_DOOR = (25, 25, 50, 50)
ON_WIRING = (110, 85, 150, 98)
ON_NOTHING = (300, 150, 350, 190)


def _pdf(path: Path, with_door_layer: bool = True, rotation: int = 0, wiring_layer: str = "E-WIRING") -> Path:
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    if with_door_layer:
        page.draw_rect(fitz.Rect(10, 10, 30, 30), oc=doc.add_ocg("A-DOOR"))
    page.draw_bezier((50, 50), (60, 40), (70, 40), (80, 50), oc=doc.add_ocg(wiring_layer))
    page.set_rotation(rotation)
    doc.save(path)
    return path


def _detection(bbox, kind="door"):
    return WindowDetection(bbox=bbox, metadata={"type": kind})


def _split(pdf_path, detections, image_size=IMAGE_SIZE):
    return DoorLayerFilter(DoorLayerFilterConfig()).split(detections, pdf_path, 0, image_size)


def test_door_on_wiring_only_is_dropped(tmp_path):
    on_door, on_wiring, on_nothing = _detection(ON_DOOR), _detection(ON_WIRING), _detection(ON_NOTHING)
    kept, dropped = _split(_pdf(tmp_path / "p.pdf"), [on_door, on_wiring, on_nothing])
    assert kept == [on_door, on_nothing]
    assert dropped == [on_wiring]


@pytest.mark.parametrize("wiring_layer", ["E-Wire-Std", "E_WIRE"])
def test_corpus_wiring_layer_names_match(tmp_path, wiring_layer):
    on_wiring = _detection(ON_WIRING)
    assert _split(_pdf(tmp_path / "p.pdf", wiring_layer=wiring_layer), [on_wiring]) == ([], [on_wiring])


def test_other_categories_on_wiring_are_kept(tmp_path):
    window = _detection(ON_WIRING, kind="window")
    assert _split(_pdf(tmp_path / "p.pdf"), [window]) == ([window], [])


def test_page_without_door_layer_keeps_everything(tmp_path):
    on_wiring = _detection(ON_WIRING)
    assert _split(_pdf(tmp_path / "p.pdf", with_door_layer=False), [on_wiring]) == ([on_wiring], [])


def test_rotated_page_maps_layers_into_image_space(tmp_path):
    # rotation 90: page shows 100 x 200 pt; the arc lands at (100, 100, 120, 160) px.
    rotated_wiring = _detection((102, 110, 118, 150))
    kept, dropped = _split(_pdf(tmp_path / "p.pdf", rotation=90), [rotated_wiring], image_size=(200, 400))
    assert dropped == [rotated_wiring]


@pytest.mark.parametrize("enabled,expected", [(False, 2), (True, 1)])
def test_pipeline_applies_the_filter_only_when_enabled(tmp_path, enabled, expected):
    _pdf(tmp_path / "plan.pdf")
    image_path = tmp_path / "plan_page000.png"
    Image.new("RGB", IMAGE_SIZE, "white").save(image_path)
    config = PipelineConfig()
    config.use_door_layer_filter = enabled
    pipeline = AnnotationPipeline(config)
    pipeline._input_path = tmp_path
    detections = [_detection(ON_DOOR), _detection(ON_WIRING)]
    assert len(pipeline._drop_wiring_doors(image_path, detections)) == expected


def test_default_is_on_and_snapshot_records_flag_and_patterns():
    config = PipelineConfig()
    assert config.use_door_layer_filter is True
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["pipeline"]["use_door_layer_filter"] is True
    assert snapshot["door_layer_filter"] == {
        "door_layer_pattern": config.door_layer_filter.door_layer_pattern,
        "wiring_layer_pattern": config.door_layer_filter.wiring_layer_pattern,
    }
