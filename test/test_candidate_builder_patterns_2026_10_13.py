"""Candidate-builder vocabulary added 2026-10-13: each real OCR form found in
GT rooms (eval/missed_room_ocr.py) now becomes a room candidate, and the
plural lounge form maps to the same class as the singular."""


import pytest

from preprocessing_annotations.automation.taxonomy import normalize_to_mandatory
from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor, TextDetection


@pytest.mark.parametrize("text", [
    "JANITORS CLOSET", "STAIR G", "STAIR A101", "COUNC. OFFICE", "COUNC.OFFICE",
    "MAIN OFFICE", "MAIL/IDF", "TEACHERS LOUNGE", "ADA RESTROOM",
])
def test_label_becomes_a_room_candidate(text):
    detection = TextDetection(
        bbox=[[100, 100], [300, 100], [300, 130], [100, 130]], text=text, confidence=0.99,
    )
    candidates = MEPTextExtractor(PipelineConfig().ocr).find_room_candidates([detection])
    assert [c.room_name for c in candidates] == [text]


def test_teachers_lounge_maps_like_teacher_lounge():
    assert normalize_to_mandatory("TEACHERS LOUNGE") == normalize_to_mandatory("TEACHER LOUNGE")
