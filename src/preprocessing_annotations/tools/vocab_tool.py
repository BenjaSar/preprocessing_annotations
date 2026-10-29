"""AV-4 (2026-09-23): vocabulary-gap proposer.

Confirmed gap: every shipped room-vocabulary fix in this project's
history (STAIR, R.R., unit codes, ELEV/UTIL/REST, BLOOM/VEG,
SHOWROOM/TRIAGE) was found by a human reading pipeline output by hand.
Nothing in the pipeline surfaces a ranked candidate list on its own.

Scope, and its limit: SemanticRoomValidator.filter_rooms
(sft_validator.py) already logs every room name it drops for matching
no recognized keyword ("Filtered (no valid keywords): ..."), at DEBUG
level -- and setup_logging's file handler (pipeline.py) is always
DEBUG regardless of --verbose, so pipeline_<timestamp>.log already
carries this signal for every run on disk. This tool parses that log
and ranks the dropped names by frequency. Proposal only: it never
writes to config.py / sft_validator.py / taxonomy.py.

Blind spot, stated not hidden: OCRConfig.room_name_patterns
(ocr_extractor.py's earlier gate, before a name candidate is even
built) has no log line on rejection at all -- a name filtered there
never reaches the log line this tool reads, and is invisible to it.
Closing that would mean adding a new log statement to
ocr_extractor.py's hot loop -- a separate, larger-blast-radius change,
not made here.

Validated (2026-09-23): replayed against
track2_pilot_output/pipeline_2026-09-14_185351.log, a run that predates
the BLOOM/VEG vocabulary fix shipped 2026-09-15 -- confirmed it
surfaces "BLOOM RM 1", "BLOOM ROOM 3", "VEG ROOM #1", the exact terms
that fix later added.
"""

import re
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple, Union

_FILTERED_LINE = re.compile(
    r"Filtered \(no valid keywords\): (?P<name>.+?) \(expanded: (?P<expanded>.+)\)\s*$"
)

# Where a proposed term must land -- static guidance, not computed per
# term. Every shipped vocabulary fix in this project's history touched
# both files (see this module's docstring precedent list); a term
# landing in only one has historically shipped incomplete (e.g. R3's
# "R.R." side-finding: matched sft_validator.py's gate but fell through
# taxonomy.py's default to the wrong type until both were updated).
TARGET_FILES = (
    "automation/sft_validator.py (VALID_ROOM_KEYWORDS or VALID_ROOM_REGEXES)",
    "automation/taxonomy.py (synonym mapping to a canonical type)",
)


class VocabularyGapCandidate(NamedTuple):
    name: str
    count: int


def _iter_log_files(log_path: Union[str, Path]) -> List[Path]:
    path = Path(log_path)
    if path.is_dir():
        return sorted(path.glob("pipeline_*.log"))
    return [path]


def propose_vocabulary_gaps(
    log_path: Union[str, Path], min_count: int = 1,
) -> List[VocabularyGapCandidate]:
    """Parse pipeline_*.log for SemanticRoomValidator's "Filtered (no
    valid keywords)" lines and rank dropped names by frequency.

    Args:
        log_path: A single pipeline_<timestamp>.log file, or a
            directory containing one or more of them (all are read).
        min_count: Only return names dropped at least this many times.
            Default 1 keeps everything.

    Returns:
        VocabularyGapCandidate list, most-frequent first.
    """
    counts: Counter = Counter()
    for log_file in _iter_log_files(log_path):
        try:
            text = log_file.read_text(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            match = _FILTERED_LINE.search(line)
            if match:
                counts[match.group("name").strip()] += 1

    return [
        VocabularyGapCandidate(name=name, count=count)
        for name, count in counts.most_common()
        if count >= min_count
    ]
