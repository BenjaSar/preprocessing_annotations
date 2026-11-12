#!/usr/bin/env python3
"""
Gate 6 (the reason this feature is Prototype, not Adopt), pinned as a
real regression test: merging the PDF text layer into raw_detections
is NOT just noisy, it is confirmed unsafe for compute_exclusion_zones
specifically. Measured on Lake Shore Electrical p002 (real PDF, real
PaddleOCR): raw_detections 133 -> 828 (+695), exclusion zones 3 -> 16.
Visually verified the 13 new zones cover CLASSROOM 142/143/144/146/147,
SCIENCE CLASSROOM 128/129/130, PREP ROOM/CLASSROOM 145/COMPUTER LAB, and
CLASSROOM 119 -- real rooms, not BOM tables. sft_validator.py drops any
room whose centroid falls in an exclusion zone, so flipping
use_pdf_text_layer on as currently wired would delete legitimate rooms.

This test is marked xfail: it documents a KNOWN, not-yet-fixed hazard
(use_pdf_text_layer is default-off specifically because of this), not
a bug to chase. It exists so that if someone later reroutes PDF-sourced
detections away from compute_exclusion_zones (the fix this file's
docstring calls for), the test flips to an unexpected PASS (xpass) --
a visible signal to update this file and reconsider the feature's
status, instead of the fix landing silently with no regression guard
noticing either way.

Uses the real project PDF + real PaddleOCR (CPU) -- no GPU/VLM needed. The
page image is RENDERED from that PDF by the pipeline's own PDFExtractor, so
the test carries its own fixture and runs wherever the PDF is checked out.

It did not always: the page used to be read from a pre-rendered PNG under
/tmp, and the 2026-11-05 host reboot deleted it. The test then self-skipped
for four days and nothing noticed, because a skip reads as green in a summary
line. A fixture outside the project is a test that silently stops running.
"""

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from preprocessing_annotations.config import OCRConfig
from preprocessing_annotations.ingestion.ocr_extractor import _centroid
from preprocessing_annotations.ingestion.pdf_extractor import PDFExtractor
from preprocessing_annotations.ingestion.two_pass_ocr_extractor import TwoPassOCRExtractor

pytestmark = pytest.mark.slow  # runs real PaddleOCR

_PDF = Path(
    "/home/ubuntu/floorplan_classifier/VLM/test_input/"
    "BP-1 Lake Shore High School Electrical Pages.pdf"
)
_PAGE_INDEX = 2
_MERGE_TOL_PX = 20  # matches AnnotationPipeline._PDF_TEXT_LAYER_MERGE_TOL_PX


@pytest.fixture(scope="module")
def page_image(tmp_path_factory):
    """The page under test, rendered from the project PDF.

    PDFExtractor is the pipeline's own rasterizer, so the image the test
    scores is the image production would score -- including the dpi-to-edge
    scaling in PDFExtractor._page_scale. Rendering beats storing a PNG: one
    fixture to keep in step, and nothing to lose to a reboot.
    """
    if not _PDF.exists():
        pytest.skip(f"project PDF not present ({_PDF})")

    for image, page_number in PDFExtractor().extract(_PDF):
        if page_number != _PAGE_INDEX:
            continue
        path = tmp_path_factory.mktemp("page") / f"{_PDF.stem}_page{_PAGE_INDEX:03d}.png"
        image.save(path)
        return path

    raise AssertionError(f"{_PDF.name} has no page at index {_PAGE_INDEX}")


@pytest.mark.xfail(
    reason=(
        "Known, documented hazard (PipelineConfig.use_pdf_text_layer): "
        "merging PDF-sourced detections into compute_exclusion_zones's "
        "input causes real classrooms to be misclassified as BOM/title-"
        "block zones. use_pdf_text_layer stays default-off because of "
        "this. An xpass here means someone fixed the routing -- update "
        "this test and PipelineConfig's docstring together."
    ),
    strict=True,
)
def test_pdf_merge_does_not_explode_exclusion_zone_count(page_image):
    from PIL import Image

    extractor = TwoPassOCRExtractor(
        OCRConfig(backend="paddleocr", confidence_threshold=0.5), vlm_backend=None
    )
    pass1_rooms, raw_detections = extractor.find_rooms_pass1(page_image)
    w, h = Image.open(page_image).size

    zones_before = extractor.ocr_extractor.compute_exclusion_zones(
        raw_detections, w, h
    )

    pdf_detections = PDFExtractor().extract_text_layer(_PDF, _PAGE_INDEX)
    assert pdf_detections, "fixture regression: PDF text layer should not be empty"

    existing_centroids = [_centroid(d.bbox) for d in raw_detections]
    merged = list(raw_detections)
    for pdf_det in pdf_detections:
        px, py = _centroid(pdf_det.bbox)
        if any(
            abs(px - ex) <= _MERGE_TOL_PX and abs(py - ey) <= _MERGE_TOL_PX
            for ex, ey in existing_centroids
        ):
            continue
        merged.append(pdf_det)

    zones_after = extractor.ocr_extractor.compute_exclusion_zones(merged, w, h)

    # Pinned to the measured baseline (not a tight equality -- OCR/PDF
    # extraction can vary slightly run to run) but the explosion this
    # feature must never cause is unmistakable at any reasonable margin.
    assert len(zones_before) <= 5, (
        f"baseline exclusion zone count drifted: {len(zones_before)} "
        f"(expected ~3) -- re-check this test's assumptions before trusting "
        f"the after-merge comparison below"
    )
    assert len(zones_after) <= len(zones_before) + 3, (
        f"PDF text layer merge caused exclusion zones to explode "
        f"{len(zones_before)} -> {len(zones_after)} -- this is the confirmed "
        f"hazard documented in PipelineConfig.use_pdf_text_layer: real rooms "
        f"get misclassified as BOM/title-block zones and dropped downstream. "
        f"Do not merge PDF-sourced detections into compute_exclusion_zones's "
        f"input without a design change (route them to abbreviation recovery "
        f"only)."
    )
