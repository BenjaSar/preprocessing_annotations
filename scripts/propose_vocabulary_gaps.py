#!/usr/bin/env python3
"""AV-4 (2026-09-23): rank room-name vocabulary gaps from a pipeline run's log.

Reads pipeline_<timestamp>.log (or a directory of them) for
SemanticRoomValidator's "Filtered (no valid keywords)" lines and prints
a ranked candidate list. Proposal only -- edits nothing.

Usage:
    python scripts/propose_vocabulary_gaps.py --log /path/to/output_dir
    python scripts/propose_vocabulary_gaps.py --log /path/to/pipeline_2026-09-14_185351.log --min-count 2
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing_annotations.tools.vocab_tool import TARGET_FILES, propose_vocabulary_gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", "-l", required=True,
        help="pipeline_<timestamp>.log file, or a directory containing one or more",
    )
    parser.add_argument(
        "--min-count", type=int, default=1,
        help="Only show names dropped at least this many times (default: 1)",
    )
    args = parser.parse_args()

    candidates = propose_vocabulary_gaps(args.log, min_count=args.min_count)

    if not candidates:
        print("No vocabulary-gap candidates found (no 'Filtered (no valid "
              "keywords)' lines matched min_count).")
        return

    print(f"{len(candidates)} vocabulary-gap candidate(s), most-dropped first:\n")
    for candidate in candidates:
        print(f"  {candidate.count:>4}x  {candidate.name}")

    print("\nA confirmed term needs updating in BOTH:")
    for target in TARGET_FILES:
        print(f"  - {target}")
    print(
        "\nThis is a proposal list only -- verify each name against the "
        "real page (visual confirmation) before adding it anywhere, same "
        "bar every shipped vocabulary fix in this project used."
    )


if __name__ == "__main__":
    main()
