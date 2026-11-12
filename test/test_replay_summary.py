"""replay_summary feeds two decisions a chain cannot get wrong: the noise floor
a gate compares against, and whether a set of runs came from one pipeline."""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from replay_summary import UNKNOWN, load, one_fingerprint, recall_spread, table


def _replay(recall, fingerprint=None):
    """A room_replay.py report: scores nested under a section, run metadata
    at the top level."""
    report = {
        "run_dir": "run",
        "baseline": {
            "localization": {"label_recall": recall, "label_precision": 0.935},
            "category": {"accuracy": 0.945},
            "per_document": {},
        },
    }
    if fingerprint is not None:
        report["run_code_fingerprint"] = fingerprint
    return report


def _written(tmp_path, name, recall, fingerprint=None):
    path = tmp_path / f"replay_{name}.json"
    path.write_text(json.dumps(_replay(recall, fingerprint)))
    return path


def test_the_spread_is_the_range_of_recall(tmp_path):
    paths = [_written(tmp_path, "a", 0.5872), _written(tmp_path, "b", 0.6012),
             _written(tmp_path, "c", 0.5972)]

    spread = recall_spread(load(paths, "baseline"))

    assert spread["min"] == 0.5872
    assert spread["max"] == 0.6012
    assert spread["range"] == pytest.approx(0.0140)


def test_one_report_has_a_zero_spread(tmp_path):
    """Reported truthfully rather than refused -- the caller decides what a
    zero-width spread is worth. This is the shape that let a .0020 'win' pass."""
    spread = recall_spread(load([_written(tmp_path, "a", 0.6112)], "baseline"))

    assert spread["range"] == 0


def test_the_spread_needs_a_report():
    with pytest.raises(ValueError):
        recall_spread([])


def test_agreeing_fingerprints_are_reported(tmp_path):
    paths = [_written(tmp_path, "a", 0.59, "92ac4c7a44d3"),
             _written(tmp_path, "b", 0.60, "92ac4c7a44d3")]

    assert one_fingerprint(load(paths, "baseline")) == "92ac4c7a44d3"


def test_disagreeing_fingerprints_report_unknown(tmp_path):
    """So a caller comparing against its own fingerprint refuses the set
    instead of picking a winner: the 2026-10-28 merge fix moved every baseline
    ~.025, and mixing code versions reads that as a model win."""
    paths = [_written(tmp_path, "a", 0.59, "92ac4c7a44d3"),
             _written(tmp_path, "b", 0.60, "2548b4634e79")]

    assert one_fingerprint(load(paths, "baseline")) == UNKNOWN


def test_reports_predating_the_field_report_unknown(tmp_path):
    paths = [_written(tmp_path, "a", 0.59), _written(tmp_path, "b", 0.60)]

    assert one_fingerprint(load(paths, "baseline")) == UNKNOWN


def test_the_table_names_every_run_and_ends_with_the_spread(tmp_path):
    paths = [_written(tmp_path, "adapter", 0.6112), _written(tmp_path, "cap200", 0.6132)]

    rendered = table(paths, load(paths, "baseline"))

    assert "replay_adapter" in rendered
    assert "replay_cap200" in rendered
    assert "0.6112" in rendered and "0.6132" in rendered
    assert rendered.splitlines()[-1].startswith("spread")


def test_a_section_that_is_not_there_is_refused(tmp_path):
    """Scoring the wrong section silently would compare as-run filters against
    replayed ones."""
    with pytest.raises(SystemExit):
        load([_written(tmp_path, "a", 0.59)], "variant")
