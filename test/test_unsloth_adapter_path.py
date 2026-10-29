"""A5: a LoRA adapter can be loaded for inference, so a fine-tune is scored
by the same harness as its baseline.

No model is loaded here. The contract under test is the wiring: the default
is the base model, the CLI reaches the config, the run records which adapter
produced it, and a missing adapter fails loudly instead of silently scoring
the base model.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig, VLMConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend


def test_default_is_the_base_model():
    """Every run before A5 measured the base model; the default must keep that."""
    assert VLMConfig().unsloth_adapter_path is None


def test_snapshot_records_the_adapter(tmp_path):
    """A run's own artifacts must say whether an adapter produced its output."""
    config = PipelineConfig()
    config.vlm.unsloth_adapter_path = tmp_path / "adapter"

    snapshot = AnnotationPipeline(config)._build_config_dict()

    assert snapshot["vlm"]["unsloth_adapter_path"] == str(tmp_path / "adapter")


def test_snapshot_records_none_without_an_adapter():
    snapshot = AnnotationPipeline(PipelineConfig())._build_config_dict()
    assert snapshot["vlm"]["unsloth_adapter_path"] is None


def test_missing_adapter_raises_instead_of_scoring_the_base_model(tmp_path, monkeypatch):
    """Silently falling back would report a fine-tune's score for the base
    model -- worse than no run at all."""
    config = VLMConfig()
    config.unsloth_adapter_path = tmp_path / "not_there"
    backend = UnslothQwenBackend(config)

    loaded = {}

    class _StubModel:
        def load_adapter(self, path):  # pragma: no cover - must not be reached
            loaded["path"] = path

    def _from_pretrained(model_id, **kwargs):
        return _StubModel(), object()

    stub = type(sys)("unsloth")
    stub.FastVisionModel = type(
        "FastVisionModel", (),
        {"from_pretrained": staticmethod(_from_pretrained),
         "for_inference": staticmethod(lambda model: None)},
    )
    monkeypatch.setitem(sys.modules, "unsloth", stub)

    with pytest.raises(FileNotFoundError):
        backend.initialize()

    assert loaded == {}
    assert backend.initialized is False


def test_room_prompt_defaults_to_production():
    """Every run before the fine-tune used the production prompt."""
    assert VLMConfig().unsloth_room_prompt == "production"


def test_room_prompt_selects_the_builder_an_adapter_was_trained_with():
    """An adapter must be asked the way it was trained: the SFT dataset uses
    the label-box prompt, not the production wall-to-wall one."""
    from preprocessing_annotations.vlm import prompt_templates

    config = VLMConfig()
    config.unsloth_room_prompt = "sft_label"
    prompt = UnslothQwenBackend(config)._build_room_detection_prompt(1237, 1178)

    assert prompt == prompt_templates.build_room_prompt_sft_label(1237, 1178, config.max_rooms)
    assert "LABEL TEXT ONLY" in prompt


def test_production_prompt_is_unchanged_by_default():
    from preprocessing_annotations.vlm import prompt_templates

    config = VLMConfig()
    prompt = UnslothQwenBackend(config)._build_room_detection_prompt(1237, 1178)

    assert prompt == prompt_templates.build_room_prompt_unsloth(1237, 1178, config.max_rooms)


def test_unknown_room_prompt_raises():
    config = VLMConfig()
    config.unsloth_room_prompt = "nope"
    with pytest.raises(ValueError, match="unknown room prompt"):
        UnslothQwenBackend(config)._build_room_detection_prompt(100, 100)


def test_snapshot_records_the_room_prompt():
    config = PipelineConfig()
    config.vlm.unsloth_room_prompt = "sft_label"
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["vlm"]["unsloth_room_prompt"] == "sft_label"


def test_existing_adapter_is_loaded_onto_the_model(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = VLMConfig()
    config.unsloth_adapter_path = adapter
    backend = UnslothQwenBackend(config)

    loaded = {}
    order = []

    class _StubModel:
        def load_adapter(self, path):
            loaded["path"] = path
            order.append("load_adapter")

    def _from_pretrained(model_id, **kwargs):
        return _StubModel(), object()

    stub = type(sys)("unsloth")
    stub.FastVisionModel = type(
        "FastVisionModel", (),
        {"from_pretrained": staticmethod(_from_pretrained),
         "for_inference": staticmethod(lambda model: order.append("for_inference"))},
    )
    monkeypatch.setitem(sys.modules, "unsloth", stub)

    backend.initialize()

    assert loaded["path"] == str(adapter)
    assert order == ["load_adapter", "for_inference"], "adapter must be on before optimizing"
    assert backend.initialized is True
