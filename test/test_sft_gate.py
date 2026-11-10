"""The D8 gate: a fine-tune passes only by beating the baseline SPREAD, not
one lucky baseline run, and never by trading a document away."""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from sft_gate import apply_gate, load_reports, scored_report

TOLERANCE = 0.01


def _report(recall, precision=0.935, category=0.945, per_document=None, gt_rooms=100):
    """per_document values are recall rates; located counts follow from gt_rooms."""
    return {
        "run_dir": "run",
        "localization": {"label_recall": recall, "label_precision": precision},
        "category": {"accuracy": category},
        "per_document": {
            document: {
                "label_recall": value,
                "gt_rooms": gt_rooms,
                "gt_rooms_located": round(value * gt_rooms),
            }
            for document, value in (per_document or {"doc_a": recall}).items()
        },
    }


def test_clear_win_passes():
    baselines = [_report(0.570), _report(0.575)]
    candidate = _report(0.640)

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert verdict["passed"]
    assert verdict["checks"]["recall_above_baseline_spread"]


def test_a_win_inside_the_seed_spread_fails():
    """0.576 beats the median but sits inside the run-to-run noise."""
    baselines = [_report(0.570), _report(0.580)]

    verdict = apply_gate(baselines, _report(0.576), TOLERANCE)

    assert not verdict["passed"]
    assert not verdict["checks"]["recall_above_baseline_spread"]


def test_recall_bought_with_precision_fails():
    baselines = [_report(0.570), _report(0.575)]
    candidate = _report(0.700, precision=0.900)

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert not verdict["passed"]
    assert not verdict["checks"]["precision_within_tolerance"]


def test_recall_bought_with_category_accuracy_fails():
    baselines = [_report(0.570), _report(0.575)]
    candidate = _report(0.700, category=0.900)

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert not verdict["passed"]
    assert not verdict["checks"]["category_within_tolerance"]


def test_a_drop_within_tolerance_still_passes():
    baselines = [_report(0.570), _report(0.575)]
    candidate = _report(0.700, precision=0.935 - TOLERANCE, category=0.945 - TOLERANCE)

    assert apply_gate(baselines, candidate, TOLERANCE)["passed"]


def test_a_document_losing_recall_fails_even_when_the_total_rises():
    baselines = [_report(0.570, per_document={"doc_a": 0.50, "doc_b": 0.70})]
    candidate = _report(0.640, per_document={"doc_a": 0.80, "doc_b": 0.40})

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert not verdict["passed"]
    assert list(verdict["documents_losing_recall"]) == ["doc_b"]


def test_losing_a_single_room_is_treated_as_noise():
    """Measured: the three baseline runs differ only by seed, and
    masonic_heights_electrical swings one located room between them
    (100, 100, 99). A one-room difference cannot be called a regression."""
    baselines = [_report(0.570, per_document={"small_doc": 3 / 28}, gt_rooms=28)]
    candidate = _report(0.640, per_document={"small_doc": 2 / 28}, gt_rooms=28)

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert verdict["documents_losing_recall"] == {}
    assert verdict["passed"]


def test_losing_two_rooms_still_fails():
    baselines = [_report(0.570, per_document={"small_doc": 4 / 28}, gt_rooms=28)]
    candidate = _report(0.640, per_document={"small_doc": 2 / 28}, gt_rooms=28)

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert verdict["documents_losing_recall"]["small_doc"]["rooms_lost"] == 2
    assert not verdict["passed"]


def test_min_rooms_lost_one_restores_the_strict_rule():
    baselines = [_report(0.570, per_document={"small_doc": 3 / 28}, gt_rooms=28)]
    candidate = _report(0.640, per_document={"small_doc": 2 / 28}, gt_rooms=28)

    verdict = apply_gate(baselines, candidate, TOLERANCE, min_rooms_lost=1)

    assert verdict["documents_losing_recall"]["small_doc"]["rooms_lost"] == 1
    assert not verdict["passed"]


def test_a_report_without_room_counts_falls_back_to_the_strict_rule():
    """No counts means the floor cannot be applied; the document must not be
    waved through on that account."""
    def _rates_only(recall, per_document):
        report = _report(recall, per_document=per_document)
        for score in report["per_document"].values():
            score.pop("gt_rooms_located")
        return report

    verdict = apply_gate([_rates_only(0.570, {"doc": 0.50})],
                         _rates_only(0.640, {"doc": 0.49}), TOLERANCE)

    assert verdict["documents_losing_recall"]["doc"]["rooms_lost"] is None
    assert not verdict["passed"]


def test_rooms_lost_is_measured_against_the_worst_baseline():
    """Same rule as the rate: below the luckiest baseline alone is noise."""
    baselines = [_report(0.570, per_document={"doc": 10 / 100}),
                 _report(0.575, per_document={"doc": 14 / 100})]
    candidate = _report(0.640, per_document={"doc": 9 / 100})

    verdict = apply_gate(baselines, candidate, TOLERANCE)

    assert verdict["documents_losing_recall"] == {}, "1 room below the worst baseline is noise"


def test_a_document_is_judged_against_the_worst_baseline():
    """Below the luckiest baseline is noise; below all of them is a loss."""
    baselines = [
        _report(0.570, per_document={"doc_a": 0.50}),
        _report(0.575, per_document={"doc_a": 0.60}),
    ]

    verdict = apply_gate(baselines, _report(0.640, per_document={"doc_a": 0.55}), TOLERANCE)

    assert verdict["documents_losing_recall"] == {}
    assert verdict["passed"]


def test_a_document_missing_from_the_candidate_fails():
    """A document the candidate never scored cannot be assumed unharmed."""
    baselines = [_report(0.570, per_document={"doc_a": 0.5, "doc_b": 0.5})]

    verdict = apply_gate(baselines, _report(0.640, per_document={"doc_a": 0.8}), TOLERANCE)

    assert not verdict["passed"]
    assert verdict["documents_missing_from_candidate"] == ["doc_b"]


def test_spread_is_reported_for_the_record():
    verdict = apply_gate([_report(0.570), _report(0.580)], _report(0.640), TOLERANCE)

    spread = verdict["baseline_spread"]["label_recall"]
    assert (spread["min"], spread["max"]) == (0.570, 0.580)
    assert spread["range"] == pytest.approx(0.010)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_a_room_gt_eval_report_is_scored_at_the_top_level():
    scores, kind = scored_report(_report(0.57), "baseline")
    assert kind == "room_gt_eval"
    assert scores["localization"]["label_recall"] == 0.57


def _section(recall):
    """A replay report's scored section: no run_dir, which sits at its top level."""
    return {k: v for k, v in _report(recall).items() if k != "run_dir"}


def test_a_replay_report_is_scored_from_the_named_section():
    """as_run is the run's own numbers; baseline is replayed under current
    code. Picking the wrong one is how a pipeline fix reads as a model win."""
    replay = {"run_dir": "runs/x", "as_run": _section(0.57), "baseline": _section(0.60)}

    scores, kind = scored_report(replay, "baseline")

    assert kind == "baseline"
    assert scores["localization"]["label_recall"] == 0.60
    assert scores["run_dir"] == "runs/x", "the verdict identifies the candidate by run_dir"


def test_mixing_report_kinds_is_refused(tmp_path):
    """The stale-baseline trap: pre-fix baselines against a replayed candidate."""
    direct = _write(tmp_path, "baseline.json", _report(0.57))
    replay = _write(tmp_path, "candidate.json", {"as_run": _report(0.57), "baseline": _report(0.60)})

    with pytest.raises(SystemExit, match="different kinds"):
        load_reports([direct, replay], "baseline")


def test_one_kind_loads_fine(tmp_path):
    paths = [_write(tmp_path, f"{i}.json", {"baseline": _report(0.57 + i / 100)}) for i in range(2)]
    assert len(load_reports(paths, "baseline")) == 2


def test_a_report_without_the_section_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="neither top-level scores"):
        scored_report({"as_run": _report(0.57)}, "baseline")


def _coded(recall, fingerprint, **kwargs):
    return {**_report(recall, **kwargs), "run_code_fingerprint": fingerprint}


def test_runs_from_one_pipeline_version_pass_the_comparability_check():
    verdict = apply_gate([_coded(0.570, "abc"), _coded(0.575, "abc")],
                         _coded(0.640, "abc"), TOLERANCE)
    assert verdict["checks"]["one_pipeline_version"]
    assert verdict["passed"]


def test_a_candidate_built_by_different_code_fails():
    """Measured 2026-10-29/30: baselines, control and adapter ran hours apart
    while other work edited the pipeline, and nothing flagged it."""
    verdict = apply_gate([_coded(0.570, "abc"), _coded(0.575, "abc")],
                         _coded(0.640, "def"), TOLERANCE)

    assert not verdict["checks"]["one_pipeline_version"]
    assert verdict["code_fingerprints"] == ["abc", "def"]
    assert not verdict["passed"]


def test_baselines_disagreeing_with_each_other_also_fail():
    verdict = apply_gate([_coded(0.570, "abc"), _coded(0.575, "def")],
                         _coded(0.640, "abc"), TOLERANCE)
    assert not verdict["checks"]["one_pipeline_version"]


def test_unknown_fingerprints_do_not_fail_the_check():
    """Runs predating the field cannot be checked; reported, not punished."""
    verdict = apply_gate([_report(0.570), _report(0.575)], _report(0.640), TOLERANCE)

    assert verdict["checks"]["one_pipeline_version"]
    assert verdict["code_fingerprints"] == ["unknown"]


def test_one_known_version_mixed_with_unknown_still_passes():
    verdict = apply_gate([_coded(0.570, "abc"), _report(0.575)],
                         _coded(0.640, "abc"), TOLERANCE)
    assert verdict["checks"]["one_pipeline_version"]


def test_the_gate_needs_a_baseline():
    with pytest.raises(ValueError):
        apply_gate([], _report(0.640), TOLERANCE)


# --- a single baseline cannot measure its own noise -------------------------
#
# 2026-11-07: gate_sft_cap200_v1_vs_adapter_real_v1.json compared one candidate
# against ONE baseline run, so baseline_spread.range was 0 and a +.0020 recall
# delta passed "recall_above_baseline_spread". The five-seed spread measured in
# the same chain was .0140 -- the delta was a seventh of the noise the gate
# exists to reject. These tests pin the reporting that makes that visible and
# the floor that lets a caller enforce it.


def test_a_single_baseline_reports_that_the_spread_is_not_measurable():
    verdict = apply_gate([_report(0.6112)], _report(0.6132), TOLERANCE)

    assert verdict["baselines"] == 1
    assert not verdict["spread_measurable"]
    assert verdict["baseline_spread"]["label_recall"]["range"] == 0


def test_two_baselines_can_measure_their_own_spread():
    verdict = apply_gate([_report(0.570), _report(0.580)], _report(0.640), TOLERANCE)

    assert verdict["spread_measurable"]


def test_the_margin_over_the_best_baseline_is_reported():
    """The number a reader needs to judge a win, next to the pass/fail."""
    verdict = apply_gate([_report(0.6112)], _report(0.6132), TOLERANCE)

    assert verdict["recall_margin_over_best_baseline"] == 0.002


def test_a_noise_floor_rejects_a_win_smaller_than_the_measured_spread():
    """The cap200-vs-adapter case, re-judged against the 5-seed spread."""
    verdict = apply_gate([_report(0.6112)], _report(0.6132), TOLERANCE,
                         noise_floor=0.0140)

    assert not verdict["checks"]["recall_above_baseline_spread"]
    assert not verdict["passed"]


def test_a_win_larger_than_the_noise_floor_still_passes():
    verdict = apply_gate([_report(0.6112)], _report(0.6500), TOLERANCE,
                         noise_floor=0.0140)

    assert verdict["checks"]["recall_above_baseline_spread"]
    assert verdict["passed"]


def test_the_default_floor_leaves_every_published_verdict_unchanged():
    """Regression guard: the six gates already written to rebaseline_1275/
    were produced with no floor, and must still read the same way."""
    baselines = [_report(0.570), _report(0.580)]

    assert apply_gate(baselines, _report(0.640), TOLERANCE)["passed"]
    assert apply_gate(baselines, _report(0.640), TOLERANCE, noise_floor=0.0)["passed"]
    assert not apply_gate(baselines, _report(0.576), TOLERANCE)["passed"]


def test_the_floor_is_reported_in_the_verdict():
    verdict = apply_gate([_report(0.570)], _report(0.640), TOLERANCE, noise_floor=0.0140)

    assert verdict["noise_floor"] == 0.0140
