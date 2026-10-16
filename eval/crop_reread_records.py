"""Shared by crop_reread_probe.py (writes) and candidate_replay.py (reads):
the re-read file format.

`reread_selector` is re-exported from the pipeline (C2, 2026-10-15), not
defined here: the live crop-reread step builds the same selector, and two
copies would let the probe and the pipeline drift apart silently.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from preprocessing_annotations.ingestion.crop_reread import Reread
from preprocessing_annotations.orchestration.crop_reread_step import reread_selector

__all__ = ["reread_selector", "rereads_payload", "load_rereads"]


def rereads_payload(pages: Mapping[str, List[Reread]]) -> Dict[str, List[Dict[str, Any]]]:
    return {page: [dataclasses.asdict(r) for r in rereads] for page, rereads in pages.items()}


def load_rereads(path: Path) -> Dict[str, List[Reread]]:
    """Per annotation file stem, as written by crop_reread_probe.py."""
    pages = json.loads(path.read_text())["pages"]
    return {page: [Reread(**r) for r in rereads] for page, rereads in pages.items()}
