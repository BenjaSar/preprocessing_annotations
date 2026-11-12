"""The shipped adapter's pinned recall, and the check against it.

Two callers, one rule. `test/test_adapter_band_regression.py` re-scores runs
already on disk with this -- minutes, no GPU, and it pins the SCORER, since
replay never invokes the model. `VLM/adapter_pin_check.sh` re-runs the model
and scores the fresh output with the same functions -- hours on a GPU, and it
pins the MODEL. One band file behind both, so the cheap check and the
expensive one cannot drift into disagreeing definitions of "still correct".

Why a pinned seed rather than a fresh one. A fresh seed would need an invented
allowance: three runs give a .0040 band, and nothing measured says how far a
fourth may legitimately land outside it. A pinned seed needs no such number --
seeded sampling in this pipeline is deterministic (measured 2026-11-02:
control_v2 reproduced the Phase B control exactly), so re-running seed 1234
must return seed 1234's recall. A difference is then a real change in the
model, the adapter or the pipeline, not sampling noise. If that determinism
ever stops holding, this check turns noisy -- which is itself worth learning,
and `band` is reported alongside so a near-miss can be read in context.

Usage:
    python eval/adapter_pin.py --report replay.json --run-dir adapter_real_v1_b1275
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from report_io import emit_report

logger = logging.getLogger(__name__)

PIN_PATH = Path(__file__).parent / "pins" / "adapter_real_v1_band.json"


def load_pin(path: Path = PIN_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def drift(recall: float, expected: float, tolerance: float) -> Optional[float]:
    """Distance from the pinned value, or None when within tolerance.

    Returns the distance rather than a bool: ".0003 over" and ".025 over" are
    the same boolean and call for very different reactions.
    """
    return None if abs(recall - expected) <= tolerance else recall - expected


def outside_band(recall: float, band: Dict[str, float], tolerance: float) -> bool:
    return recall < band["min"] - tolerance or recall > band["max"] + tolerance


def recall_of(report: Dict[str, Any], section: str) -> float:
    """Label recall from a room_replay.py report, or a room_gt_eval.py one.

    room_replay nests its scorings under sections and only the replayed ones
    reflect code changes since the run; room_gt_eval scores at the top level.
    """
    scored = report[section] if section in report else report
    return float(scored["localization"]["label_recall"])


def check(report: Dict[str, Any], run_dir: str, pin: Dict[str, Any]) -> Dict[str, Any]:
    """Compare one scored run against its pinned value. Pure; returns a verdict.

    A run the pin does not name is refused rather than waved through: the pin
    is per-run, so comparing an unpinned run against another run's number
    would be meaningless whichever way it came out.
    """
    if run_dir not in pin["runs"]:
        raise KeyError(
            f"{run_dir!r} is not pinned; the band names {sorted(pin['runs'])}"
        )

    recall = recall_of(report, pin["replay_section"])
    expected = pin["runs"][run_dir]
    moved = drift(recall, expected, pin["tolerance"])
    return {
        "run_dir": run_dir,
        "adapter": pin["adapter"],
        "recall": recall,
        "pinned": expected,
        "drift": moved,
        "tolerance": pin["tolerance"],
        "band": pin["band"],
        "inside_band": not outside_band(recall, pin["band"], pin["tolerance"]),
        "matches_pin": moved is None,
    }


def explain(verdict: Dict[str, Any]) -> str:
    if verdict["matches_pin"]:
        return (f"{verdict['run_dir']} scores {verdict['recall']:.6f}, pinned at "
                f"{verdict['pinned']:.6f} -- unchanged")
    inside = "inside" if verdict["inside_band"] else "outside"
    return (
        f"{verdict['run_dir']} scores {verdict['recall']:.6f}, pinned at "
        f"{verdict['pinned']:.6f} ({verdict['drift']:+.4f}), {inside} the "
        f"[{verdict['band']['min']:.4f}, {verdict['band']['max']:.4f}] band. "
        f"Seeded sampling is deterministic here, so re-running one seed must "
        f"return that seed's number: this is a change in the model, the "
        f"adapter or the pipeline, not noise. If it is intended, re-measure "
        f"every run in the band file together -- a band half-updated is worse "
        f"than none."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True,
                        help="room_replay.py report for the run to check")
    parser.add_argument("--run-dir", required=True,
                        help="Which pinned run this report scores")
    parser.add_argument("--pin", type=Path, default=PIN_PATH,
                        help="Band file (default: eval/pins/adapter_real_v1_band.json)")
    parser.add_argument("--out", type=Path, default=None, help="Also write the verdict here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    verdict = check(json.loads(args.report.read_text()), args.run_dir, load_pin(args.pin))
    logger.info(explain(verdict))
    emit_report(verdict, args.out)
    sys.exit(0 if verdict["matches_pin"] else 1)
