#!/usr/bin/env python3
"""
AV-4 Vocabulary Gap Proposer Tests.

Covers:
  - a synthetic log with SemanticRoomValidator's "Filtered (no valid
    keywords)" lines is parsed into ranked candidates
  - candidates are ranked by frequency, most-dropped first
  - min_count filters out infrequent candidates
  - unrelated log lines (different level/message) are ignored
  - a directory of multiple log files is aggregated across all of them
  - replay against the real pre-fix log (2026-09-14, predates the
    2026-09-15 BLOOM/VEG shipped fix) surfaces the exact terms that
    fix later added -- the validation this tool's own docstring claims
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.tools.vocab_tool import propose_vocabulary_gaps

_SAMPLE_LOG = """\
19:51:29 | DEBUG    | preprocessing_annotations.automation.sft_validator | Filtered (no valid keywords): P2 (expanded: P2)
19:51:29 | DEBUG    | preprocessing_annotations.automation.sft_validator | Filtered (no valid keywords): BLOOM RM 1 (expanded: BLOOM RM 1)
19:51:29 | INFO     | preprocessing_annotations.automation.sft_validator | Step 1: semantic filter removed 6 room(s) (34 remaining)
19:51:30 | DEBUG    | preprocessing_annotations.automation.sft_validator | Filtered (no valid keywords): BLOOM RM 1 (expanded: BLOOM RM 1)
19:51:30 | DEBUG    | preprocessing_annotations.automation.sft_validator | Filtered (too small): FOO (bbox=[0,0,1,1])
"""


def test_parses_and_ranks_by_frequency(tmp_path):
    log_file = tmp_path / "pipeline_2026-09-14_185351.log"
    log_file.write_text(_SAMPLE_LOG)

    candidates = propose_vocabulary_gaps(log_file)

    by_name = {c.name: c.count for c in candidates}
    assert by_name["BLOOM RM 1"] == 2
    assert by_name["P2"] == 1
    # "Filtered (too small)" is a different sft_validator drop reason
    # (bbox-size gate, not keyword gate) -- must not be picked up here.
    assert "FOO" not in by_name
    # ranked, most-frequent first
    assert candidates[0].name == "BLOOM RM 1"


def test_min_count_filters_infrequent_candidates(tmp_path):
    log_file = tmp_path / "pipeline_2026-09-14_185351.log"
    log_file.write_text(_SAMPLE_LOG)

    candidates = propose_vocabulary_gaps(log_file, min_count=2)

    assert len(candidates) == 1
    assert candidates[0].name == "BLOOM RM 1"


def test_empty_log_yields_no_candidates(tmp_path):
    log_file = tmp_path / "pipeline_empty.log"
    log_file.write_text("19:00:00 | INFO | x | nothing relevant here\n")

    assert propose_vocabulary_gaps(log_file) == []


def test_directory_of_logs_aggregated(tmp_path):
    (tmp_path / "pipeline_a.log").write_text(
        "19:51:29 | DEBUG | x | Filtered (no valid keywords): STAIR #3 (expanded: STAIR #3)\n"
    )
    (tmp_path / "pipeline_b.log").write_text(
        "19:51:29 | DEBUG | x | Filtered (no valid keywords): STAIR #3 (expanded: STAIR #3)\n"
    )

    candidates = propose_vocabulary_gaps(tmp_path)

    assert len(candidates) == 1
    assert candidates[0].name == "STAIR #3"
    assert candidates[0].count == 2


def test_replay_against_real_pre_fix_log_surfaces_bloom_veg():
    """The exact validation this tool's docstring claims: replayed
    against a real run that predates the 2026-09-15 BLOOM/VEG
    vocabulary fix, the proposer must surface the terms that fix later
    added. Skips (not fails) if the artifact isn't present in this
    environment -- it's a real run output on the project filesystem,
    not a repo-tracked fixture."""
    log_path = Path(
        "/home/ubuntu/floorplan_classifier/VLM/track2_pilot_output/"
        "pipeline_2026-09-14_185351.log"
    )
    if not log_path.exists():
        import pytest
        pytest.skip("real pre-fix run artifact not present in this environment")

    candidates = propose_vocabulary_gaps(log_path)
    names = {c.name for c in candidates}

    assert "BLOOM RM 1" in names
    assert "BLOOM ROOM 3" in names
    assert "VEG ROOM #1" in names
