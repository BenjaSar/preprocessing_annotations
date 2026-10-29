"""The run's config snapshot (pipeline_config.json / run manifest) records the
room-filter and attachment settings and the filter flags, so a run states
which filter behavior produced its output."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


def test_snapshot_records_ocr_candidate_patterns():
    config = PipelineConfig()
    snapshot = AnnotationPipeline(config)._build_config_dict()
    assert snapshot["ocr"]["room_name_patterns"] == list(config.ocr.room_name_patterns)
    assert snapshot["ocr"]["room_number_pattern"] == config.ocr.room_number_pattern
