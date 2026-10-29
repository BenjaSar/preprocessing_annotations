#!/usr/bin/env python3
"""
T3: config/flag contract tests for the SAM3 exemplar feature.

Covers:
  - use_sam3_exemplar defaults False (regression guard: default-off ship)
  - AnnotationPipeline.to_dict() reports the flag (no cheap way to verify
    a run's actual config from its output otherwise)
  - --use-sam3-exemplar is registered on the CLI parser
  - a warning fires when use_sam3_exemplar is set without use_yolo_objects
    (A4: that combination compiles, runs, and silently no-ops every page)
"""

import subprocess
import sys
from pathlib import Path

# Add the package dir (parent of test/) so `preprocessing_annotations` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import (
    AnnotationPipeline,
    _warn_if_sam3_exemplar_orphaned,
)

_VLM_PYTHON = sys.executable


def test_use_sam3_exemplar_defaults_false():
    config = PipelineConfig()
    assert config.use_sam3_exemplar is False


def test_config_dict_reports_the_flag():
    config = PipelineConfig()
    config.use_sam3_exemplar = True
    pipeline = AnnotationPipeline(config)

    d = pipeline._build_config_dict()

    assert d["pipeline"]["use_sam3_exemplar"] is True


def test_cli_flag_registered():
    """--help exits before any pipeline work runs (argparse's own
    behavior) -- safe to invoke as a subprocess smoke test."""
    result = subprocess.run(
        [_VLM_PYTHON, "-m", "preprocessing_annotations.orchestration.pipeline", "--help"],
        capture_output=True, text=True, timeout=60,
        cwd=str(Path(__file__).parent.parent / "src"),
    )
    assert "--use-sam3-exemplar" in result.stdout


def test_warns_when_orphaned(caplog):
    config = PipelineConfig()
    config.use_sam3_exemplar = True
    config.use_yolo_objects = False

    with caplog.at_level("WARNING"):
        _warn_if_sam3_exemplar_orphaned(config)

    assert any("no effect without --use-yolo-objects" in r.message for r in caplog.records)


def test_no_warning_when_yolo_objects_enabled(caplog):
    config = PipelineConfig()
    config.use_sam3_exemplar = True
    config.use_yolo_objects = True

    with caplog.at_level("WARNING"):
        _warn_if_sam3_exemplar_orphaned(config)

    assert caplog.records == []


def test_no_warning_when_sam3_exemplar_disabled(caplog):
    config = PipelineConfig()  # both default False
    with caplog.at_level("WARNING"):
        _warn_if_sam3_exemplar_orphaned(config)

    assert caplog.records == []
