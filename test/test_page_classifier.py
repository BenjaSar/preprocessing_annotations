#!/usr/bin/env python3
"""
AV-6 Page-Type Classifier Tests — validate response parsing and the
PageClassifier adapter without loading any model (UnslothQwenBackend
is faked, same pattern as test_door_detector.py's _FakeCubiCasa).

Covers:
  - _parse_page_type_response: valid JSON, JSON wrapped in prose/
    markdown, invalid JSON, unrecognized page_type, missing rationale
  - UnslothPageClassifier.classify translates the backend's dict into
    a PageTypeResult without touching model internals
  - AnnotationPipeline.page_classifier is None when use_page_type is
    off (no VLM constructed) and resolves to an adapter when on with
    an Unsloth backend configured
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend
from preprocessing_annotations.vlm.classify import PageTypeResult, UnslothPageClassifier


def test_parse_valid_json():
    result = UnslothQwenBackend._parse_page_type_response(
        '{"page_type": "floor_plan", "rationale": "shows labeled rooms"}'
    )
    assert result == {"page_type": "floor_plan", "rationale": "shows labeled rooms"}


def test_parse_json_wrapped_in_prose():
    text = 'Sure, here is the answer:\n{"page_type": "other", "rationale": "unclear"}\nDone.'
    result = UnslothQwenBackend._parse_page_type_response(text)
    assert result["page_type"] == "other"
    assert result["rationale"] == "unclear"


def test_parse_invalid_json_falls_back_to_other():
    result = UnslothQwenBackend._parse_page_type_response("not json at all")
    assert result["page_type"] == "other"


def test_parse_unrecognized_page_type_falls_back_to_other():
    result = UnslothQwenBackend._parse_page_type_response(
        '{"page_type": "site_plan", "rationale": "x"}'
    )
    assert result["page_type"] == "other"
    assert "site_plan" in result["rationale"]


def test_parse_missing_rationale_defaults_to_empty_string():
    result = UnslothQwenBackend._parse_page_type_response('{"page_type": "floor_plan"}')
    assert result == {"page_type": "floor_plan", "rationale": ""}


class _FakeBackend:
    """Mimics UnslothQwenBackend's classify_page_type return shape,
    without model internals."""

    def __init__(self, response: dict):
        self._response = response
        self.calls = []

    def classify_page_type(self, image_path):
        self.calls.append(image_path)
        return self._response


def test_adapter_translates_backend_dict_to_result():
    fake = _FakeBackend({"page_type": "non_architectural_document", "rationale": "a schedule"})
    adapter = UnslothPageClassifier(fake)

    result = adapter.classify("page000.png")

    assert result == PageTypeResult(
        page_type="non_architectural_document", rationale="a schedule"
    )
    assert fake.calls == ["page000.png"]


def test_pipeline_page_classifier_none_when_disabled():
    from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
    from preprocessing_annotations.config import PipelineConfig

    pipeline = AnnotationPipeline(PipelineConfig())  # use_page_type defaults False

    assert pipeline.page_classifier is None
    # Confirms no VLM was constructed as a side effect of the check.
    assert pipeline._vlm_annotator is None


def test_pipeline_page_classifier_resolves_for_unsloth_backend():
    from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
    from preprocessing_annotations.config import PipelineConfig

    config = PipelineConfig()
    config.use_page_type = True
    config.vlm.backend = "unsloth"
    pipeline = AnnotationPipeline(config)
    # Avoid a real model load: inject a fake already-constructed backend
    # the same way pipeline.py's lazy `vlm_annotator` property would.
    pipeline._vlm_annotator = UnslothQwenBackend(config.vlm)

    classifier = pipeline.page_classifier

    assert isinstance(classifier, UnslothPageClassifier)
    # Resolved once, cached -- second access returns the same instance.
    assert pipeline.page_classifier is classifier
