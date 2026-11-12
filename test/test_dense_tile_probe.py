"""dense_tile_probe's GPU-free half: which tiles it picks and how it scores.

The probe answers whether the Rockaway collapse is the base model or the
fine-tune, so its tile choice has to be the pipeline's grid and its GT counts
tile-local; an off-by-offset polygon would score every detection as a miss.
"""

import sys
from pathlib import Path

from PIL import Image
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from dense_tile_probe import densest_tiles, localization


def _page(tmp_path, name, size):
    Image.new("RGB", size, "white").save(tmp_path / f"{name}.png")


def test_picks_the_tile_holding_most_gt_rooms_in_tile_coordinates(tmp_path):
    config = PipelineConfig()
    side = config.tile_trigger_px * 2
    _page(tmp_path, "doc_a_page000", (side, side))
    # Three rooms around the page centre, one in the top-left corner.
    centre = side // 2
    gt = {"doc_a_page000": [
        box(centre - 50, centre - 50, centre + 50, centre + 50),
        box(centre - 250, centre - 50, centre - 150, centre + 50),
        box(centre + 150, centre - 50, centre + 250, centre + 50),
        box(100, 100, 200, 200),
    ]}

    [tile] = densest_tiles(tmp_path, gt, "doc_a", 1, config)

    assert len(tile.gt) == 3
    meta = tile.meta
    assert meta.x_offset > 0 and meta.y_offset > 0
    # Shifted into the tile by exactly its offset.
    assert max(p.bounds[2] for p in tile.gt) == centre + 250 - meta.x_offset
    assert min(p.bounds[1] for p in tile.gt) == centre - 50 - meta.y_offset


def test_only_the_requested_document_is_probed(tmp_path):
    config = PipelineConfig()
    size = (config.tile_trigger_px * 2, config.tile_trigger_px * 2)
    _page(tmp_path, "doc_a_page000", size)
    _page(tmp_path, "doc_b_page000", size)
    gt = {"doc_b_page000": [box(100, 100, 200, 200)]}

    tiles = densest_tiles(tmp_path, gt, "doc_a", 50, config)

    assert tiles and {tile.page for tile in tiles} == {"doc_a_page000"}
    assert all(not tile.gt for tile in tiles)


def test_a_page_below_the_tiling_trigger_yields_no_tiles(tmp_path):
    config = PipelineConfig()
    _page(tmp_path, "doc_a_page000", (config.tile_trigger_px, config.tile_trigger_px))

    assert densest_tiles(tmp_path, {}, "doc_a", 5, config) == []


def test_localization_counts_each_gt_room_once_by_detection_center():
    gt = [box(0, 0, 100, 100), box(200, 0, 300, 100), box(400, 0, 500, 100)]
    rooms = [
        {"bbox": [10, 10, 90, 90]},     # center in room 1
        {"bbox": [20, 20, 80, 80]},     # room 1 again: not a second hit
        {"bbox": [150, 0, 350, 100]},   # center (250, 50) in room 2
        {"bbox": [90, 0, 210, 100]},    # center (150, 50) between rooms
        {"bbox": None},                 # unparseable box: ignored, not fatal
    ]

    assert localization(rooms, gt) == {"located": 2, "center_in_gt": 3}


def test_a_detection_inside_nested_gt_rooms_locates_only_the_smallest():
    """The eval's rule: one GT room per detection, so a unit polygon wrapping
    its bedroom cannot turn one detection into two located rooms."""
    gt = [box(0, 0, 400, 400), box(100, 100, 200, 200)]

    assert localization([{"bbox": [120, 120, 180, 180]}], gt) == {"located": 1, "center_in_gt": 1}
