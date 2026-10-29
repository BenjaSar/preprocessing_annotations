"""OCRConfig.room_name_placeholder_pattern: a drawing's own "???" marks around
a room name are removed for room-candidate matching, and nowhere else."""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor, TextDetection

# Real OCR from sprint1_verify56, BP-1 Lake Shore High School Electrical
# Pages p002 (viewed at 3x: the sheet prints the "???" itself).
LAKE_SHORE = TextDetection(
    bbox=[[2528, 1527], [2653, 1527], [2653, 1547], [2528, 1547]],
    text="??? CLASSROOM ???", confidence=0.9,
)


def _extractor(**ocr_overrides):
    config = PipelineConfig()
    return MEPTextExtractor(dataclasses.replace(config.ocr, **ocr_overrides))


@pytest.mark.parametrize("text,expected", [
    ("??? CLASSROOM ???", "CLASSROOM"),
    ("???STORAGE???", "STORAGE"),
    ("CLASSROOM", "CLASSROOM"),          # unchanged without the marks
    ("???", ""),
])
def test_strip_room_name_placeholders(text, expected):
    assert _extractor().strip_room_name_placeholders(text) == expected


def test_placeholder_wrapped_name_is_a_room_label():
    extractor = _extractor()
    assert extractor.is_room_name(LAKE_SHORE.text)
    assert extractor.matches_room_label(LAKE_SHORE.text)


def test_candidate_carries_the_name_without_the_marks():
    candidates = _extractor().find_room_candidates([LAKE_SHORE])
    assert [c.room_name for c in candidates] == ["CLASSROOM"]
    assert candidates[0].bbox == (2528, 1527, 125, 20)   # box unchanged


def test_pattern_comes_from_config_not_code():
    assert _extractor(room_name_placeholder_pattern=r"@+").strip_room_name_placeholders("@@@ CLASSROOM @@@") == "CLASSROOM"
    assert _extractor(room_name_placeholder_pattern=r"@+").find_room_candidates([LAKE_SHORE]) == []


def test_text_without_placeholders_is_untouched():
    plain = TextDetection(bbox=LAKE_SHORE.bbox, text="CLASSROOM", confidence=0.9)
    extractor = _extractor()
    assert [c.room_name for c in extractor.find_room_candidates([plain])] == ["CLASSROOM"]
    assert not extractor.is_room_name("TAIR")
