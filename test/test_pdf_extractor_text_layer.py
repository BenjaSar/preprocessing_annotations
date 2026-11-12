#!/usr/bin/env python3
"""
PDFExtractor.extract_text_layer against real project PDFs (real
PyMuPDF, no GPU/model). Reproduces the JANITOR recovery directly and
confirms the two boundary conditions the audit's scan found: a
substantial-text PDF and a zero-text ("...FLAT") PDF.

Skips gracefully if the project's test_input PDFs aren't present on
this machine (matches the skip pattern used by other real-fixture
tests in this suite, e.g. test_sam3_exemplar_integration.py).
"""

from pathlib import Path

import pytest

from preprocessing_annotations.ingestion.pdf_extractor import PDFExtractor

_TEST_INPUT = Path("/home/ubuntu/floorplan_classifier/VLM/test_input")
_LAKE_SHORE_PDF = _TEST_INPUT / "BP-1 Lake Shore High School Electrical Pages.pdf"
_FLAT_PDF = _TEST_INPUT / "BP-1 Kennedy Middle School  - AVI-ON LAYOUT FLAT.pdf"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"project fixture not present: {path}")


def test_janitor_recovered_at_correct_image_coordinates():
    """The exact case that motivated this prototype: 'JANITOR' exists in
    the PDF text layer, OCR found nothing at the same image location.
    Ground truth (fitz.open(...).load_page(2).get_text('words')):
    JANITOR at PDF coords (644.1,756.0) on a 2592x1728pt page; this
    method must scale that into the same image space the pipeline's
    rasterized page uses -- PDFExtractor._page_scale (dpi/72, capped)."""
    _skip_if_missing(_LAKE_SHORE_PDF)
    extractor = PDFExtractor()

    detections = extractor.extract_text_layer(_LAKE_SHORE_PDF, 2)

    janitor = [d for d in detections if d.text.upper() == "JANITOR"]
    assert len(janitor) == 1
    (x1, y1), _, (x2, y2), _ = janitor[0].bbox
    # Measured image location (audit crop confirmed "??? JANITOR ???"
    # visually here); allow a few px for the scale/rounding path.
    assert abs(x1 - 1118) < 5
    assert abs(y1 - 1312) < 5
    assert janitor[0].confidence == PDFExtractor.PDF_TEXT_LAYER_CONFIDENCE


def test_flat_pdf_returns_empty_not_an_error():
    """Text outlined to curves -- zero extractable words is the CORRECT
    answer for this PDF family, not a failure to be masked or logged
    as an error."""
    _skip_if_missing(_FLAT_PDF)
    extractor = PDFExtractor()

    detections = extractor.extract_text_layer(_FLAT_PDF, 0)

    assert detections == []


def test_nonexistent_pdf_returns_empty_list_no_raise():
    extractor = PDFExtractor()
    detections = extractor.extract_text_layer("/nonexistent/path.pdf", 0)
    assert detections == []


def test_out_of_range_page_index_returns_empty_list_no_raise():
    _skip_if_missing(_LAKE_SHORE_PDF)
    extractor = PDFExtractor()
    detections = extractor.extract_text_layer(_LAKE_SHORE_PDF, 9999)
    assert detections == []


def test_bbox_is_four_corner_points_not_a_tuple():
    """The exact contract mismatch that caused the R1/S1 bug elsewhere
    in this codebase: ocr_extractor.TextDetection.bbox must be
    [[x,y],[x,y],[x,y],[x,y]], not a 4-tuple (that's ocr_adapter's
    different TextDetection). Guards this at the source."""
    _skip_if_missing(_LAKE_SHORE_PDF)
    extractor = PDFExtractor()

    detections = extractor.extract_text_layer(_LAKE_SHORE_PDF, 2)

    assert len(detections) > 0
    sample = detections[0]
    assert isinstance(sample.bbox, list)
    assert len(sample.bbox) == 4
    assert all(len(point) == 2 for point in sample.bbox)
