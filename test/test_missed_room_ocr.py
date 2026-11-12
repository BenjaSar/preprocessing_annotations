"""eval/missed_room_ocr.py with a stub OCR and a generated PDF (no PaddleOCR)."""

import sys
from pathlib import Path

import fitz
import pytest
from PIL import Image
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from missed_room_ocr import (
    NO_TEXT,
    TEXT_AT_RUN_RESOLUTION,
    TEXT_ONLY_AT_HIGHER_RESOLUTION,
    PdfPageRenderer,
    candidate_boxes,
    classify_page,
    missed_gt_indices,
    texts_inside,
)
from preprocessing_annotations.ingestion.ocr_extractor import TextDetection

RUN_SIZE = (200, 100)      # run image px
PAGE_PT = (100, 50)        # PDF page pt -> run scale 2
FACTOR = 2.0               # render 400x200, ratio 2
GTS = [
    box(0, 0, 40, 40),     # A: has a candidate
    box(50, 0, 90, 40),    # B: text at run resolution
    box(100, 0, 160, 40),  # C: text only in the re-render crop (crop width 121)
    box(0, 50, 30, 90),    # D: no text anywhere
]


def _det(cx, cy, text, half=2):
    return TextDetection(
        bbox=[[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
        text=text, confidence=0.99,
    )


def _page(tmp_path, stem="docA_page000"):
    with fitz.open() as document:
        document.new_page(width=PAGE_PT[0], height=PAGE_PT[1])
        document.save(tmp_path / "docA.pdf")
    image = tmp_path / f"{stem}.png"
    Image.new("RGB", RUN_SIZE, "white").save(image)
    return image


def test_candidate_boxes_unions_vlm_and_ocr_rooms_as_xyxy():
    annotation = {"rooms": [{"bbox": [10, 10, 20, 20]}], "ocr_rooms": [{"bbox": [1, 2, 3, 4]}, {"bbox": []}]}
    assert candidate_boxes(annotation) == [(10, 10, 30, 30), (1, 2, 4, 6)]


def test_missed_gt_indices_excludes_rooms_with_a_candidate_centre():
    assert missed_gt_indices(GTS, [(10, 10, 30, 30)]) == [1, 2, 3]


def test_texts_inside_maps_crop_coordinates_back_to_run_pixels():
    crop_detection = _det(40, 40, "OFFICE")   # crop px, ratio 2, crop origin run (100, 0)
    assert texts_inside(GTS[2], [crop_detection], offset=(100, 0), scale=2.0) == ["OFFICE"]
    assert texts_inside(GTS[2], [crop_detection]) == []


def test_classify_page_sorts_missed_rooms_by_ocr_evidence(tmp_path):
    run_image = _page(tmp_path)

    def stub_ocr(path):
        if path == run_image:
            return [_det(70, 20, "RM 101")]
        with Image.open(path) as crop:
            return [_det(crop.width / 2, crop.height / 2, "OFFICE")] if crop.width == 121 else []

    rooms = classify_page(
        "docA_page000", run_image, GTS, ["a", "b", "c", "d"], [(10, 10, 30, 30)],
        stub_ocr, PdfPageRenderer(tmp_path, FACTOR),
    )

    assert [(r.gt_index, r.cause, r.texts) for r in rooms] == [
        (1, TEXT_AT_RUN_RESOLUTION, ["RM 101"]),
        (2, TEXT_ONLY_AT_HIGHER_RESOLUTION, ["OFFICE"]),
        (3, NO_TEXT, []),
    ]
    assert [r.gt_category for r in rooms] == ["b", "c", "d"]


def test_classify_page_skips_ocr_when_nothing_is_missed(tmp_path):
    run_image = _page(tmp_path)

    def failing_ocr(path):
        raise AssertionError("OCR must not run")

    assert classify_page(
        "docA_page000", run_image, GTS[:1], ["a"], [(10, 10, 30, 30)],
        failing_ocr, PdfPageRenderer(tmp_path, FACTOR),
    ) == []


def test_renderer_rejects_run_image_that_is_not_a_uniform_scale(tmp_path):
    _page(tmp_path)
    with pytest.raises(ValueError, match="uniform scale"):
        PdfPageRenderer(tmp_path, FACTOR).render("docA_page000", (200, 60))


def test_renderer_rejects_non_page_stem(tmp_path):
    with pytest.raises(ValueError, match="pageNNN"):
        PdfPageRenderer(tmp_path, FACTOR).render("not_a_page", RUN_SIZE)
