"""Phase 0 SFT decisions (config.SFTDatasetConfig) stay true, and the tile
grid the dataset exporter will share with inference (TileSplitter.for_page)
is the grid the pipeline has always used."""

import hashlib

import pytest

from preprocessing_annotations.automation.annotation_schema import LABEL_BOX
from preprocessing_annotations.config import (
    PipelineConfig,
    RoomGtEvalConfig,
    SFTDatasetConfig,
)
from preprocessing_annotations.ingestion.tile_splitter import TileSplitter


def test_box_convention_is_the_label_box_rooms_are_tagged_with():
    assert SFTDatasetConfig().box_convention == LABEL_BOX


def test_gt_category_map_is_the_frozen_one():
    """Fails when the map is edited: re-score the zero-shot baseline against
    the new map and update the hash in the same change."""
    path = RoomGtEvalConfig().gt_category_map_path
    if not path.exists():
        pytest.skip(f"GT category map not on this host: {path}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        SFTDatasetConfig().gt_category_map_sha256
    )


def _grid_before_extraction(width, height, config):
    """pipeline._detect_rooms_maybe_tiled's inline grid, as it stood before
    the grid moved to TileSplitter.for_page."""
    if config.tile_target_px > 0:
        cols = max(2, min(config.tile_cols, round(width / config.tile_target_px)))
        rows = max(2, min(config.tile_rows, round(height / config.tile_target_px)))
    else:
        cols, rows = config.tile_cols, config.tile_rows
    return cols, rows


@pytest.mark.parametrize("tile_target_px", [PipelineConfig().tile_target_px, 0])
@pytest.mark.parametrize("size", [(3001, 2200), (4500, 3000), (4500, 4500), (2200, 9000)])
def test_for_page_reproduces_the_pipeline_grid(size, tile_target_px):
    config = PipelineConfig()
    config.tile_target_px = tile_target_px

    splitter = TileSplitter.for_page(*size, config)

    assert (splitter.cols, splitter.rows) == _grid_before_extraction(*size, config)
    assert splitter.overlap_pct == config.tile_overlap_pct


def test_for_page_sends_small_pages_whole():
    config = PipelineConfig()
    edge = config.tile_trigger_px
    assert TileSplitter.for_page(edge, edge, config) is None


def test_for_page_respects_tiling_switch():
    config = PipelineConfig()
    config.use_tiling = False
    big = config.tile_trigger_px * 2
    assert TileSplitter.for_page(big, big, config) is None
