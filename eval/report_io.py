"""Shared report-emission for the eval CLIs.

AF3 built this in kaggle_door_window_eval.py to stop measurements from
living only in terminal scrollback -- two of this project's recorded
baselines had to be recovered from a session transcript, and one config
docstring drifted stale because the newer numbers were never written
anywhere. door_localization_diagnostic.py had the same stdout-only gap.
Extracted here once a second real caller existed, rather than duplicating
the function -- same reasoning as any other module-level extraction in
this codebase (e.g. window_detector.py's map_detections_to_rooms).
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def emit_report(payload: Any, output_path: Optional[Path]) -> None:
    """Print the report, and additionally persist it when a path is given.

    One serialization feeds both sinks, so a persisted file can never
    disagree with what was printed -- including the trailing newline
    print() adds, so the file diffs clean against captured stdout.

    Default (``output_path=None``) is stdout-only, byte-identical to
    every prior behavior of both callers.
    """
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if output_path is not None:
        output_path.write_text(text)
        logger.info("wrote report to %s", output_path)
