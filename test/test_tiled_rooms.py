"""Tiled room detection: one geometry, shared by the pipeline and Pass 2.

A whole-page VLM call does not ground small room labels (measured on
sprint1_verify55: 1 of 712 name-matched Pass-2 candidates held its own OCR
token, against 1773 of 1773 for Pass 1), so every caller that wants page
coordinates detects over TileSplitter's grid and merges once.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.tile_splitter import TileSplitter
from preprocessing_annotations.vlm.tiled_rooms import (
    MERGE_MAX_ROOMS,
    detect_rooms_in_tiles,
    splitter_for_image,
)


class RecordingBackend:
    """A backend that reports one room per tile, at a fixed tile-local box."""

    def __init__(self, local_bbox=(10, 20, 60, 50), fail_on: int = -1):
        self.local_bbox = local_bbox
        self.fail_on = fail_on
        self.tile_sizes: List[tuple] = []

    def detect_rooms_from_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        index = len(self.tile_sizes)
        self.tile_sizes.append(image.size)
        if index == self.fail_on:
            raise RuntimeError("tile detection failed")
        return [{"room_name": f"ROOM{index}", "bbox": list(self.local_bbox), "confidence": 0.9}]

    def detect_hallucinations(self, rooms, check_stripes: bool = True):
        return rooms


@pytest.fixture
def page() -> Image.Image:
    return Image.new("RGB", (4000, 3000), "white")


def test_every_tile_is_detected_and_boxes_land_in_page_space(page):
    splitter = TileSplitter(cols=2, rows=2, overlap_pct=0.1)
    backend = RecordingBackend()

    rooms = detect_rooms_in_tiles(backend, page, splitter)

    assert len(backend.tile_sizes) == 4
    assert all(size != page.size for size in backend.tile_sizes)
    # One tile sits at the origin, the others do not: boxes are offset, not tile-local.
    origins = {tuple(room["bbox"][:2]) for room in rooms}
    assert backend.local_bbox[:2] in origins
    assert len(origins) == len(rooms) > 1
    assert all(
        0 <= room["bbox"][0] <= page.width and 0 <= room["bbox"][1] <= page.height
        for room in rooms
    )


def test_a_failing_tile_does_not_lose_the_page(page):
    splitter = TileSplitter(cols=2, rows=2, overlap_pct=0.1)
    backend = RecordingBackend(fail_on=1)

    rooms = detect_rooms_in_tiles(backend, page, splitter)

    assert len(backend.tile_sizes) == 4
    assert len(rooms) == 3


def test_merge_caps_the_page(page):
    class ManyRooms(RecordingBackend):
        def detect_rooms_from_image(self, image):
            self.tile_sizes.append(image.size)
            return [
                {"room_name": f"R{i}", "bbox": [i * 7, i * 5, i * 7 + 4, i * 5 + 4], "confidence": 0.5}
                for i in range(MERGE_MAX_ROOMS)
            ]

    rooms = detect_rooms_in_tiles(ManyRooms(), page, TileSplitter(cols=2, rows=2))

    assert len(rooms) == MERGE_MAX_ROOMS


def test_splitter_for_image_declines_when_tiling_is_off(tmp_path, page):
    image_path = tmp_path / "page.png"
    page.save(image_path)
    config = PipelineConfig()
    config.use_tiling = False

    assert splitter_for_image(image_path, config) is None


def test_splitter_for_image_declines_a_page_below_the_trigger(tmp_path):
    image_path = tmp_path / "small.png"
    config = PipelineConfig()
    Image.new("RGB", (config.tile_trigger_px - 1, 500), "white").save(image_path)

    assert splitter_for_image(image_path, config) is None


def test_splitter_for_image_declines_an_unreadable_file(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_text("not an image")

    assert splitter_for_image(image_path, PipelineConfig()) is None


def test_splitter_for_image_returns_the_pipeline_grid(tmp_path, page):
    image_path = tmp_path / "page.png"
    page.save(image_path)
    config = PipelineConfig()

    tiling = splitter_for_image(image_path, config)

    assert tiling is not None
    image, splitter = tiling
    expected = TileSplitter.for_page(page.width, page.height, config)
    assert image.size == page.size
    assert (splitter.cols, splitter.rows, splitter.overlap_pct) == (
        expected.cols,
        expected.rows,
        expected.overlap_pct,
    )
