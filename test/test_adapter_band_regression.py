"""The shipped adapter's recall band, pinned against silent pipeline drift.

Nothing else in the suite fails if `adapter_real_v1` stops scoring ~.610. One
change has already moved this number across the board: the 2026-10-28 merge
fix lifted every baseline by ~.025, and it was caught only because someone
re-ran the baselines by hand and noticed.

What this pins, and what it does not. Re-scoring a stored run with
`eval/room_replay.py` does NOT run the model -- it re-reads predictions
already on disk and re-applies the current filters and scorer. So the result
is deterministic, the expected drift is exactly zero, and any movement is a
change in scoring or filtering code. The model side of the same band is
checked by `VLM/adapter_pin_check.sh`, which re-runs the adapter and compares
through the same `eval/adapter_pin.py` functions -- hours on a GPU, so a
chain job rather than a test.

Marked `slow`: three replays at ~68s each. It needs no GPU, but it is far too
long for the default suite.

The band lives in `eval/pins/`, beside the code that reads it and inside the
project -- not in `/tmp`, where a page fixture was parked until the
2026-11-05 reboot deleted it and its test self-skipped for four days.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from adapter_pin import PIN_PATH, check, drift, load_pin, outside_band

# Runs live beside the pipeline, not under test/: they are gigabytes of run
# artifacts, written by the chains and overwritten by later ones.
RUN_ROOT = PROJECT_ROOT.parent
REPLAY = PROJECT_ROOT / "eval" / "room_replay.py"


# --- the pin itself, always checked, even where the runs are long gone ------


def test_the_pin_is_internally_consistent():
    """A band that does not match its own measurements would pass every run
    and mean nothing."""
    pin = load_pin()

    assert pin["band"]["min"] == min(pin["runs"].values())
    assert pin["band"]["max"] == max(pin["runs"].values())


def test_the_pin_names_several_seeds():
    """One seed is a point, not a band; the 2026-11-07 cap200 verdict is what
    reading one seed as a result looks like."""
    assert len(load_pin()["runs"]) >= 3


def test_the_tolerance_is_far_inside_the_band_width():
    """A tolerance as wide as the band would admit any member as any other."""
    pin = load_pin()
    width = pin["band"]["max"] - pin["band"]["min"]

    assert 0 < pin["tolerance"] < width / 4


def test_drift_inside_tolerance_is_not_drift():
    assert drift(0.6112, 0.6112, 0.0005) is None
    assert drift(0.6113, 0.6112, 0.0005) is None


def test_drift_outside_tolerance_is_reported_with_its_distance():
    assert drift(0.6362, 0.6112, 0.0005) == pytest.approx(0.0250)


def test_the_2026_10_28_merge_fix_would_be_caught():
    """That change lifted recall ~.025 everywhere. The guard exists for it."""
    pin = load_pin()

    assert outside_band(pin["band"]["max"] + 0.025, pin["band"], pin["tolerance"])


def test_a_value_inside_the_band_passes():
    pin = load_pin()
    middle = (pin["band"]["min"] + pin["band"]["max"]) / 2

    assert not outside_band(middle, pin["band"], pin["tolerance"])


# --- check(), the rule both the test and the chain apply -------------------


def _report(recall, section="baseline"):
    return {section: {"localization": {"label_recall": recall}}}


def test_check_passes_a_run_that_still_scores_its_pinned_value():
    pin = load_pin()
    run = "adapter_real_v1_b1275"

    verdict = check(_report(pin["runs"][run]), run, pin)

    assert verdict["matches_pin"] and verdict["inside_band"]
    assert verdict["drift"] is None


def test_check_fails_a_run_that_moved():
    pin = load_pin()
    run = "adapter_real_v1_b1275"

    verdict = check(_report(pin["runs"][run] + 0.025), run, pin)

    assert not verdict["matches_pin"]
    assert verdict["drift"] == pytest.approx(0.025)


def test_check_refuses_a_run_the_pin_does_not_name():
    """Comparing an unpinned run against some other run's number would be
    meaningless whichever way it came out."""
    with pytest.raises(KeyError):
        check(_report(0.6112), "some_other_run", load_pin())


def test_check_reads_a_top_level_report_too():
    """room_gt_eval scores at the top level, room_replay under a section."""
    pin = load_pin()
    run = "adapter_real_v1_b1275"
    flat = {"localization": {"label_recall": pin["runs"][run]}}

    assert check(flat, run, pin)["matches_pin"]


# --- the live check --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("run_dir", sorted(load_pin()["runs"]))
def test_a_stored_adapter_run_still_scores_what_it_scored(run_dir, tmp_path):
    """Re-score the run on disk under today's code and compare to the pin."""
    pin = load_pin()
    run = RUN_ROOT / run_dir
    if not (run / "pipeline_config.json").exists():
        pytest.skip(f"run artifacts not present ({run}); the chains overwrite these")

    recorded = json.loads((run / "pipeline_config.json").read_text())["code_fingerprint"]
    assert recorded == pin["code_fingerprint"], (
        f"{run_dir} was produced by {recorded}, but the band was measured on "
        f"{pin['code_fingerprint']} -- these are different runs, so the "
        f"comparison below would be meaningless. Re-measure the pin."
    )

    report = tmp_path / "replay.json"
    result = subprocess.run(
        [sys.executable, str(REPLAY), "--run", str(run), "--out", str(report)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    verdict = check(json.loads(report.read_text()), run_dir, pin)

    assert verdict["matches_pin"], (
        f"{run_dir} re-scores to {verdict['recall']:.6f}, pinned at "
        f"{verdict['pinned']:.6f} ({verdict['drift']:+.4f}). Replay does not "
        f"re-run the model, so this is a scoring or filtering change. If it is "
        f"intended, re-measure every run in {PIN_PATH.name} together -- a band "
        f"half-updated is worse than none."
    )
