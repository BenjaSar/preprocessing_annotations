"""OCRConfig.extend_titleblock_zone_to_near_edge_only: a title-block zone is
extended along both ends of its long axis (default) or only toward the nearer
page edge."""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import (
    MEPTextExtractor,
    TextDetection,
    _extend_titleblock_zone,
)

# Bradley Fair p000's "KEY PLAN" notes block, 2200 x 1700 page.
BRADLEY_BLOCK = [(1341, 77, 2109, 816)]


@pytest.mark.parametrize("near_edge_only,expected", [
    (False, (0, 77, 2200, 816)),        # both ends: 43% of the page
    (True, (1341, 77, 2200, 816)),      # nearer (right) edge only
])
def test_horizontal_block_extension(near_edge_only, expected):
    assert _extend_titleblock_zone(BRADLEY_BLOCK, 2200, 1700, near_edge_only) == expected


def test_near_edge_picks_the_left_edge_for_a_left_title_block():
    assert _extend_titleblock_zone([(100, 50, 900, 300)], 3000, 4500, True) == (0, 50, 900, 300)


@pytest.mark.parametrize("zones,expected", [
    ([(2700, 1000, 2900, 4300)], (2700, 1000, 2900, 4500)),  # nearer bottom edge
    ([(2700, 300, 2900, 2000)], (2700, 0, 2900, 2000)),     # nearer top edge
])
def test_vertical_strip_extends_to_its_nearer_edge(zones, expected):
    assert _extend_titleblock_zone(zones, 3000, 4500, True) == expected


def test_extractor_reads_the_flag_from_config():
    detections = [
        TextDetection([[1900, 450], [2000, 450], [2000, 470], [1900, 470]], "BUILDING KEY PLAN - NT.S", 0.99),
    ]
    config = PipelineConfig().ocr
    assert config.extend_titleblock_zone_to_near_edge_only is False
    both = MEPTextExtractor(config).compute_exclusion_zones(detections, 2200, 1700)
    near = MEPTextExtractor(
        dataclasses.replace(config, extend_titleblock_zone_to_near_edge_only=True)
    ).compute_exclusion_zones(detections, 2200, 1700)
    assert both == [(0, 400, 2200, 520)]
    assert near == [(1850, 400, 2200, 520)]


@pytest.mark.parametrize("near_edge_only", [False, True])
def test_snapshot_records_the_flag(near_edge_only):
    from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline

    config = PipelineConfig()
    config.ocr = dataclasses.replace(config.ocr, extend_titleblock_zone_to_near_edge_only=near_edge_only)
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["ocr"]["extend_titleblock_zone_to_near_edge_only"] is near_edge_only
