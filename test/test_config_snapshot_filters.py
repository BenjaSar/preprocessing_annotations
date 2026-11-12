"""The run's config snapshot (pipeline_config.json / run manifest) records the
room-filter and attachment settings and the filter flags, so a run states
which filter behavior produced its output."""

import json

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline


def test_snapshot_records_room_filter_attachment_and_filter_flags():
    config = PipelineConfig()
    config.room_filter.skip_vlm_echo_candidates = False
    config.room_filter.merge_proximity_px = 50

    snapshot = AnnotationPipeline(config)._build_config_dict()

    assert snapshot["room_filter"]["skip_vlm_echo_candidates"] is False
    assert snapshot["room_filter"]["merge_proximity_px"] == 50
    assert snapshot["attachment"]["policy"] == config.attachment.policy
    for flag in ("use_window_elongation_filter", "exempt_sam_expanded_from_ink_filter",
                 "block_single_letter_abbreviations", "use_page_type", "use_pdf_text_layer"):
        assert snapshot["pipeline"][flag] is getattr(config, flag)


def test_snapshot_stays_json_serializable():
    json.dumps(AnnotationPipeline(PipelineConfig())._build_config_dict())


def test_snapshot_records_vlm_generation_settings():
    """A run's decoding must be reconstructable from its own artifacts: the
    generation settings default from VLM_* env vars, so the snapshot is the
    only record of what this run actually used."""
    config = PipelineConfig()
    snapshot = AnnotationPipeline(config)._build_config_dict()

    for field in ("do_sample", "seed", "temperature", "top_p", "top_k",
                  "presence_penalty", "repetition_penalty"):
        assert snapshot["vlm"][field] == getattr(config.vlm, field)


def test_snapshot_records_the_room_token_budget_inputs():
    """The budget changed between runs (G-EVAL-3: 1024 -> 1275), and a gated
    comparison across that change is invalid -- so each run has to say which
    budget it generated under."""
    config = PipelineConfig()
    config.vlm.max_rooms = 30
    snapshot = AnnotationPipeline(config)._build_config_dict()["vlm"]

    for field in ("room_max_new_tokens", "room_tokens_per_room", "max_rooms"):
        assert snapshot[field] == getattr(config.vlm, field)
    assert max(snapshot["room_max_new_tokens"], snapshot["max_rooms"] * snapshot["room_tokens_per_room"]) \
        == config.vlm.room_token_budget()


def test_snapshot_records_ocr_candidate_patterns():
    config = PipelineConfig()
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["ocr"]["room_name_patterns"] == list(config.ocr.room_name_patterns)
    assert snapshot["ocr"]["room_number_pattern"] == config.ocr.room_number_pattern
