"""use_ocr_vlm_fallback: Pass 2 is off by default (2026-10-20) and can be
turned back on without touching OCR device selection (which depends on the
VLM backend being present)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import RoomCandidate
from preprocessing_annotations.ingestion.two_pass_ocr_extractor import TwoPassOCRExtractor
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline


class UnslothQwenBackend:                       # name matters: triggers CPU forcing
    def __init__(self):
        self.calls = 0

    def detect_rooms(self, image_path):
        self.calls += 1
        return [{"bbox": [10, 10, 60, 50], "room_name": "OFFICE", "confidence": 0.9}]


def _low_confidence_room():
    return RoomCandidate(bbox=(10, 10, 50, 40), room_number="", room_name="OFFICE", confidence=0.5)


def _extractor(enabled):
    config = PipelineConfig()
    backend = UnslothQwenBackend()
    extractor = TwoPassOCRExtractor(
        config.ocr, vlm_backend=backend, vlm_fallback_enabled=enabled,
    )
    return extractor, backend, config.ocr


def test_enabled_fallback_runs_when_a_low_confidence_room_exists():
    extractor, backend, _ = _extractor(True)
    extractor.refine_with_vlm("p.png", [_low_confidence_room()])
    assert backend.calls == 1


def test_disabled_fallback_skips_pass2_and_returns_pass1_unchanged():
    extractor, backend, _ = _extractor(False)
    rooms = [_low_confidence_room()]
    assert extractor.refine_with_vlm("p.png", rooms) is rooms
    assert backend.calls == 0


def test_disabled_fallback_still_forces_ocr_to_cpu_for_a_local_vlm():
    _, _, ocr_config = _extractor(False)
    assert ocr_config.device == "cpu"


def test_config_default_keeps_fallback_off_and_snapshot_records_it():
    config = PipelineConfig()
    assert config.use_ocr_vlm_fallback is False
    assert AnnotationPipeline(config)._build_config_dict()["pipeline"]["use_ocr_vlm_fallback"] is False
    config.use_ocr_vlm_fallback = True
    assert AnnotationPipeline(config)._build_config_dict()["pipeline"]["use_ocr_vlm_fallback"] is True


def test_pipeline_passes_flag_to_extractor():
    assert AnnotationPipeline(PipelineConfig()).ocr_extractor.vlm_fallback_enabled is False
    config = PipelineConfig()
    config.use_ocr_vlm_fallback = True
    assert AnnotationPipeline(config).ocr_extractor.vlm_fallback_enabled is True


def test_cli_flag_registered():
    result = subprocess.run(
        [sys.executable, "-m", "preprocessing_annotations.orchestration.pipeline", "--help"],
        capture_output=True, text=True, timeout=60,
        cwd=str(Path(__file__).parent.parent / "src"),
    )
    assert "--ocr-vlm-fallback" in result.stdout
    assert "--no-ocr-vlm-fallback" not in result.stdout
