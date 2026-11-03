"""Window-detection tiers: CubiCasa5K, the VLM backends, and the SFT output.

Rewritten 2026-10-29. Every test here used to end in `return False` inside a
bare `except Exception`, so pytest recorded a pass whatever happened: seven
green results that asserted nothing. Two causes were hiding behind that:
the imports named top-level modules (`vlm_backend`, `config`, `window_detector`)
that do not exist outside the package, and the CubiCasa model was looked for in
test/models/ while the detector's own default is models/ beside the package.
Both are fixed here; the model loads on CPU so a test run cannot take VRAM from
a live pipeline or training job.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.annotation_schema import SFTAnnotationBuilder
from preprocessing_annotations.automation.taxonomy import add_window_suffix
from preprocessing_annotations.config.config import VLMConfig
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
    WindowDetector,
    apply_window_suffixes,
)
from preprocessing_annotations.vlm.vlm_backend import (
    ClaudeBackend,
    Qwen2_5VLBackend,
    UnslothQwenBackend,
    VLMFactory,
)

CPU_DEVICE = "cpu"
UNSLOTH_BACKEND = "unsloth"


def test_cubicasa5k_model_loads():
    """The checkpoint the detector defaults to is loadable."""
    detector = CubiCasa5KDetector(device=CPU_DEVICE)
    if not detector.is_available:
        pytest.skip(f"CubiCasa5K checkpoint not present: {detector.model_path}")
    assert detector.load_model(), f"checkpoint present but unloadable: {detector.model_path}"


def test_unsloth_backend_is_reachable_through_the_factory():
    """The configured unsloth model key resolves, and the factory builds that backend."""
    if importlib.util.find_spec(UNSLOTH_BACKEND) is None:
        pytest.skip("unsloth is not installed")

    config = VLMConfig()
    config.backend = UNSLOTH_BACKEND
    backend = VLMFactory.create(config)

    assert isinstance(backend, UnslothQwenBackend)
    assert config.unsloth_model in UnslothQwenBackend.MODEL_REGISTRY


def test_window_detector_maps_every_room_without_models():
    """With no tier available, each room still gets exactly one mapping."""
    rooms = [
        {"id": 0, "type": "CONFERENCE", "bbox": [50, 50, 200, 200]},
        {"id": 1, "type": "PRIVATE OFFICE", "bbox": [250, 250, 400, 400]},
    ]
    mappings = WindowDetector().detect_windows(
        image_array=np.zeros((500, 500, 3), dtype=np.uint8),
        rooms=rooms,
        pdf_path=None,
        vlm_backend=None,
    )

    assert len(mappings) == len(rooms)
    assert [mapping.room_id for mapping in mappings] == [room["id"] for room in rooms]


@pytest.mark.parametrize("backend_class", [Qwen2_5VLBackend, ClaudeBackend])
def test_window_detection_prompt_asks_for_json_windows(backend_class):
    prompt = backend_class(VLMConfig())._build_window_detection_prompt()

    assert "window" in prompt.lower()
    assert "JSON" in prompt.upper()


def test_window_response_parsing_keeps_order_and_types():
    response = """[
        {"bbox": [10, 20, 30, 40], "type": "window", "confidence": 0.95},
        {"bbox": [50, 50, 80, 90], "type": "window", "confidence": 0.87},
        {"bbox": [100, 10, 150, 50], "type": "skylight", "confidence": 0.92}
    ]"""

    windows = Qwen2_5VLBackend(VLMConfig())._parse_window_response(response)

    assert [window["type"] for window in windows] == ["window", "window", "skylight"]


def test_window_suffix_is_applied_only_to_rooms_that_contain_a_window():
    windows = [
        WindowDetection(
            bbox=(100, 100, 150, 150),
            confidence=0.9,
            source_tier=WindowDetectionTier.CUBICASA5K,
        ),
        WindowDetection(
            bbox=(200, 200, 250, 250),
            confidence=0.85,
            source_tier=WindowDetectionTier.CUBICASA5K,
        ),
    ]
    rooms = [
        {"id": 0, "type": "CONFERENCE", "bbox": [50, 50, 300, 300]},
        {"id": 1, "type": "CORRIDOR", "bbox": [310, 50, 400, 300]},
    ]

    apply_window_suffixes(rooms, WindowDetector().map_windows_to_rooms(windows, rooms))

    assert rooms[0]["type"] == add_window_suffix("CONFERENCE", has_windows=True)
    assert rooms[1]["type"] == "CORRIDOR"
    assert "window_detection" in rooms[0]


def test_sft_room_carries_the_window_suffix_and_the_schema_fields():
    builder = SFTAnnotationBuilder()
    suffixed_type = add_window_suffix("CONFERENCE", has_windows=True)

    room = builder.build_room(
        room_id=1,
        mandatory_type=suffixed_type,
        original_name="CONF RM",
        room_number="101",
        bbox=[100, 100, 300, 300],
        detection_score=0.9,
        classification_match_type="vlm",
        ocr_confidence=1.0,
        source="vlm_only",
        detection_method="vlm",
        detection_model="Qwen",
    )
    room_dict = builder.to_dict(room)

    assert room_dict["type"] == suffixed_type
    for field in ("nameUnique", "coverage", "provenance"):
        assert field in room_dict
