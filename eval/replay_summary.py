"""Summarize a set of replay reports: their recall spread, their fingerprint,
or a table of them.

A chain needs three things from a pile of `eval/room_replay.py` reports before
it can gate anything, and all three were being done by hand or by inline
`python -c` in shell scripts:

  --spread        the seed spread, to pass to `sft_gate --noise-floor` when a
                  comparison has too few baselines to measure its own noise
  --fingerprint   the one source fingerprint behind them, to refuse mixing runs
                  from different code
  --table         recall per run, so a reader sees the overlap between two
                  bands rather than only a verdict

The spread is computed by sft_gate's own `_spread` over sft_gate's own
`scored_report`, so the floor handed to the gate is measured exactly the way
the gate measures the spread it compares against. Reimplementing it here is
how the two would drift apart.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).parent))

from sft_gate import RECALL, _overall, _spread, scored_report

UNKNOWN = "unknown"


def load(paths: Sequence[Path], section: str) -> List[Dict[str, Any]]:
    return [scored_report(json.loads(path.read_text()), section)[0] for path in paths]


def recall_spread(reports: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """The run-to-run range of label recall. One report has a range of 0 --
    true, and the caller must decide what a zero-width spread is worth."""
    if not reports:
        raise ValueError("need at least one report")
    return _spread([_overall(r, RECALL) for r in reports])


def one_fingerprint(reports: Sequence[Dict[str, Any]]) -> str:
    """The single source fingerprint behind every report, or UNKNOWN.

    Returns UNKNOWN when the reports disagree, so a caller comparing it to its
    own fingerprint refuses the set rather than picking a winner. Reports
    predating the field also read UNKNOWN and are refused the same way --
    'I cannot tell' and 'they differ' both mean do not mix these.
    """
    fingerprints = {r.get("run_code_fingerprint", UNKNOWN) for r in reports}
    return fingerprints.pop() if len(fingerprints) == 1 else UNKNOWN


def table(paths: Sequence[Path], reports: Sequence[Dict[str, Any]]) -> str:
    width = max(len(p.stem) for p in paths)
    lines = [
        f"{path.stem:{width}s}  recall {_overall(report, RECALL):.4f}"
        for path, report in zip(paths, reports)
    ]
    spread = recall_spread(reports)
    lines.append(
        f"{'spread':{width}s}  min {spread['min']:.4f}  median {spread['median']:.4f}  "
        f"max {spread['max']:.4f}  range {spread['range']:.4f}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, nargs="+", help="room_replay.py reports")
    parser.add_argument("--replay-section", default="baseline",
                        help="Which section of a replay report to read (default: baseline, "
                             "i.e. replayed under current code)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--spread", action="store_true",
                      help="Print the recall range alone, for --noise-floor (the default)")
    mode.add_argument("--fingerprint", action="store_true",
                      help="Print the single source fingerprint, or 'unknown' if they differ")
    mode.add_argument("--table", action="store_true", help="Print recall per run, then the spread")
    args = parser.parse_args()

    reports = load(args.reports, args.replay_section)
    if args.fingerprint:
        print(one_fingerprint(reports))
    elif args.table:
        print(table(args.reports, reports))
    else:
        print(f"{recall_spread(reports)['range']:.4f}")
