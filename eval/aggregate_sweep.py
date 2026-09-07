"""I-2: print P/R/F1 per (dataset, category, conf) from sweep/*_conf*.json.

Read-only aggregator over eval/kaggle_door_window_eval.py --output JSON.
Not a CLI flag on that script (YAGNI) -- the sweep is a shell loop, this
is a separate small reader over its output directory.
"""

import json
import re
import sys
from pathlib import Path

_PAT = re.compile(r"^(?P<dataset>.+)_conf(?P<conf>[0-9.]+)\.json$")


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main() -> None:
    sweep_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep")
    rows = []
    for path in sorted(sweep_dir.glob("*_conf*.json")):
        match = _PAT.match(path.name)
        if not match:
            continue
        report = json.loads(path.read_text())[0]
        for category, score in report["vs_gt"].items():
            p, r = score["precision"], score["recall"]
            rows.append((
                match["dataset"], category, float(match["conf"]),
                p, r, f1(p, r), score["true_positives"],
                score["false_positives"], score["false_negatives"],
            ))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    header = f"{'dataset':<22} {'cat':<8} {'conf':>5} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    for dataset, category, conf, p, r, score_f1, tp, fp, fn in rows:
        print(
            f"{dataset:<22} {category:<8} {conf:>5.2f} {p:>7.4f} {r:>7.4f} "
            f"{score_f1:>7.4f} {tp:>5} {fp:>5} {fn:>5}"
        )


if __name__ == "__main__":
    main()
