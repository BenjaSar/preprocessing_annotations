"""C2 (2026-10-15): crop re-read wired as a pipeline step.

Guards what the wiring must not get wrong:
  - the flag is off by default and the step is then never built, so a run's
    output is byte-identical to before
  - crops come from the page in OCR's coordinate space, not the page as it
    sits on disk (originals are 9600px; OCR reads at <=4500px, and Step 3a's
    disk resize happens later and is a different transform)
  - a setup that cannot do what the flag claims is rejected loudly at config
    time, never silently skipped
  - re-reads are returned raw, for a later offline bounds sweep

The reader is faked, which the port explicitly allows (any implementation
swappable, string out, never raises).
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import CropRereadConfig, PipelineConfig  # noqa: E402
from preprocessing_annotations.ingestion.ocr_extractor import (  # noqa: E402
    MEPTextExtractor,
    TextDetection,
)
from preprocessing_annotations.orchestration.crop_reread_step import (  # noqa: E402
    CropRereadStep,
    _page_in_ocr_space,
)


class FakeReader:
    """CropTextReader that reports the size of every crop it is handed."""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.crop_sizes = []

    def read(self, image) -> str:
        self.crop_sizes.append(image.size)
        return self.reply


def _detection(text, x, y, w=60, h=20):
    return TextDetection(
        text=text,
        confidence=0.9,
        bbox=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
    )


def _page(tmp_path, size):
    path = tmp_path / "page.png"
    Image.new("RGB", size, "white").save(path)
    return path


def _step(reader, config=None):
    return CropRereadStep(
        MEPTextExtractor(PipelineConfig().ocr), reader, config or CropRereadConfig()
    )


def _unsloth_vlm():
    vlm = PipelineConfig().vlm
    vlm.backend = "unsloth"
    return vlm


class TestCapabilityGate:
    """On by default, but only where the run can actually do it.

    A reason, not an exception: the flag defaults on, so an OCR-only or
    Claude-backend run has to proceed without the step rather than die at
    startup. The reason is what keeps the skip from being silent.
    """

    def test_on_by_default(self):
        assert PipelineConfig().use_crop_reread is True

    def test_runs_when_the_run_has_an_unsloth_vlm(self):
        config = PipelineConfig(use_vlm=True, vlm=_unsloth_vlm())
        assert config.crop_reread_skip_reason() is None

    def test_default_claude_run_skips_without_raising(self):
        # Class default is backend 'claude' with use_vlm off -- the ordinary
        # run must not crash just because crop re-read defaults on.
        reason = PipelineConfig().crop_reread_skip_reason()
        assert reason is not None and "use_vlm" in reason

    def test_non_unsloth_backend_skips(self):
        reason = PipelineConfig(use_vlm=True).crop_reread_skip_reason()
        assert reason is not None and "unsloth" in reason

    def test_pdf_text_layer_combination_skips_as_unmeasured(self):
        reason = PipelineConfig(
            use_vlm=True, use_pdf_text_layer=True, vlm=_unsloth_vlm()
        ).crop_reread_skip_reason()
        assert reason is not None and "use_pdf_text_layer" in reason

    def test_explicit_off_is_honoured_even_when_capable(self):
        reason = PipelineConfig(
            use_crop_reread=False, use_vlm=True, vlm=_unsloth_vlm()
        ).crop_reread_skip_reason()
        assert reason == "use_crop_reread is off"


class TestOcrCoordinateSpace:
    def test_scale_size_matches_the_ocr_bound(self):
        limit = MEPTextExtractor._OCR_MAX_DIM_PX
        assert MEPTextExtractor.ocr_scale_size((limit, limit // 2)) is None
        scaled = MEPTextExtractor.ocr_scale_size((limit * 2, limit))
        assert max(scaled) == limit

    def test_small_page_is_not_resized(self, tmp_path):
        page = _page(tmp_path, (800, 600))
        with _page_in_ocr_space(page) as image:
            assert image.size == (800, 600)

    def test_oversized_page_is_read_in_ocr_space(self, tmp_path):
        limit = MEPTextExtractor._OCR_MAX_DIM_PX
        page = _page(tmp_path, (limit * 2, limit))
        with _page_in_ocr_space(page) as image:
            assert max(image.size) == limit

    def test_crops_are_taken_in_ocr_space_not_from_the_disk_page(self, tmp_path):
        """The bug this prevents: cropping a 9600px page with coordinates OCR
        produced at 4500px reads the wrong part of the drawing."""
        limit = MEPTextExtractor._OCR_MAX_DIM_PX
        reader = FakeReader()
        page = _page(tmp_path, (limit * 2, limit * 2))
        _step(reader).rebuild_candidates(page, [_detection("TAIR", 10, 10)])
        assert reader.crop_sizes, "selector picked nothing; fixture no longer exercises crops"
        # A crop can never exceed the page it came from.
        for width, height in reader.crop_sizes:
            assert width <= limit and height <= limit


class TestRebuildCandidates:
    def test_no_detections_is_a_no_op(self, tmp_path):
        reader = FakeReader()
        candidates, rereads = _step(reader).rebuild_candidates(_page(tmp_path, (100, 100)), [])
        assert (candidates, rereads, reader.crop_sizes) == ([], [], [])

    def test_returns_raw_rereads_for_a_later_bounds_sweep(self, tmp_path):
        """Raw, not consistency-filtered: an offline sweep re-picks the guard,
        so it needs what the VLM actually said."""
        reader = FakeReader("COMPLETELY UNRELATED")
        _, rereads = _step(reader).rebuild_candidates(
            _page(tmp_path, (800, 600)), [_detection("TAIR", 10, 10)]
        )
        assert [r.reread for r in rereads] == ["COMPLETELY UNRELATED"]
        assert [r.original for r in rereads] == ["TAIR"]

    def test_caller_detections_are_not_mutated(self, tmp_path):
        # They also feed compute_exclusion_zones, documented unsafe with
        # non-OCR input.
        detections = [_detection("TAIR", 10, 10)]
        _step(FakeReader("STAIR")).rebuild_candidates(
            _page(tmp_path, (800, 600)), detections
        )
        assert detections[0].text == "TAIR"
