"""A content hash of the pipeline source, recorded in every run.

Two runs compared against each other must have been produced by the same
code. On 2026-10-29/30 they were not: parallel work edited pipeline.py,
two_pass_ocr_extractor.py, taxonomy.py and others while a chain was running
baseline, control and adapter runs hours apart, and nothing in any artifact
said so. eval/room_replay.py re-runs the FILTER stage under current code, so
it neutralises filter-stage edits -- but OCR and VLM stage edits are baked
into each run's recorded annotations and survive the replay.

Git is not enough: most of this project's runs are made from a working tree
with uncommitted edits, so a commit hash would call two different trees the
same. This hashes the files themselves.
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# The installed package root: everything whose behaviour a run depends on.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DIGEST_CHARS = 12


def _source_files(root: Path) -> Iterable[Path]:
    """Every .py under `root`, excluding caches, in a stable order."""
    return sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def source_fingerprint(root: Path = _PACKAGE_ROOT) -> str:
    """Short hex digest of the package's source, or "unknown" if unreadable.

    Covers each file's path (relative to `root`) and bytes, so a move counts
    as a change. Never raises: a run must not fail because its provenance
    could not be computed, and "unknown" is honest about the gap.
    """
    digest = hashlib.sha256()
    try:
        for path in _source_files(root):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    except OSError as error:
        logger.warning("could not fingerprint source at %s: %s", root, error)
        return "unknown"
    return digest.hexdigest()[:_DIGEST_CHARS]
