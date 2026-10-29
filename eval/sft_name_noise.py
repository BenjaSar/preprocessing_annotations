"""How noisy are the room NAMES the SFT exporter would train on?

The SFT prompt asks for "exact label text as printed", so a target string is
trained verbatim. Spot-checked on track2_v2's Rally Palmhurst page the drawing
reads OFFICE 03 / 02 / 01 while the pipeline recorded OFFICE-08, OFFICE-02 and
OFFICE (PaddleOCR read "-FICE.0") -- correct boxes, wrong strings.

There is no ground truth for label strings (gt_segmentation/result.json carries
room polygons and categories, not the printed text), so this does not measure
accuracy. It reports the signal that is available -- `confidence_detail.ocr`,
PaddleOCR's own text-recognition confidence -- and what an
SFTDatasetConfig.min_name_ocr_confidence floor would cost in targets at each
candidate value. Choosing the floor is a judgment about that trade, made on
the numbers rather than on one spot-check.

Rooms are read exactly as the exporter reads them (qwen_sft_exporter.rooms_of),
so the counts are of targets, not of every room on the page.

Usage:
    python eval/sft_name_noise.py --run <run_dir> [--run <run_dir> ...]
"""

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

from preprocessing_annotations.config import SFTDatasetConfig
from preprocessing_annotations.export.qwen_sft_exporter import rooms_of

from report_io import emit_report

logger = logging.getLogger(__name__)

# Floors to price, spanning the observed range. Candidates to choose among,
# not thresholds this project has adopted.
CANDIDATE_FLOORS = (0.0, 0.80, 0.85, 0.90, 0.95)


def _ocr_confidence(room: Dict[str, Any]) -> float:
    return float((room.get("confidence_detail") or {}).get("ocr", 0.0))


def _target_rooms(annotation: Dict[str, Any], convention: str) -> List[Dict[str, Any]]:
    """The raw room records behind the exporter's targets, in the same order."""
    names = [room.name for room in rooms_of(annotation, convention)[0]]
    wanted = Counter(names)
    kept = []
    for room in annotation.get("roomsRecognized") or []:
        name = (room.get("name") or "").strip()
        if wanted[name]:
            wanted[name] -= 1
            kept.append(room)
    return kept


def measure(run_dirs: Sequence[Path], config: SFTDatasetConfig) -> Dict[str, Any]:
    per_run: Dict[str, Any] = {}
    for run_dir in run_dirs:
        confidences: List[float] = []
        lowest: List[Dict[str, Any]] = []
        for path in sorted((run_dir / "processed_annotations").glob("*.json")):
            annotation = json.loads(path.read_text())
            for room in _target_rooms(annotation, config.box_convention):
                confidence = _ocr_confidence(room)
                confidences.append(confidence)
                lowest.append({
                    "name": room.get("name"),
                    "type": room.get("type"),
                    "ocr_confidence": round(confidence, 3),
                    "page": path.stem,
                })
        lowest.sort(key=lambda row: row["ocr_confidence"])
        total = len(confidences)
        per_run[run_dir.name] = {
            "targets": total,
            "ocr_confidence": {
                "min": round(min(confidences), 3) if confidences else None,
                "median": round(statistics.median(confidences), 3) if confidences else None,
                "max": round(max(confidences), 3) if confidences else None,
            },
            "targets_dropped_by_floor": {
                str(floor): sum(1 for c in confidences if c < floor)
                for floor in CANDIDATE_FLOORS
            },
            "lowest_confidence_targets": lowest[:20],
        }
    return {
        "box_convention": config.box_convention,
        "current_floor": config.min_name_ocr_confidence,
        "candidate_floors": list(CANDIDATE_FLOORS),
        "per_run": per_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price an OCR-confidence floor on SFT target names")
    parser.add_argument("--run", type=Path, action="append", required=True, dest="runs",
                        help="Run dir with processed_annotations; repeat for several")
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    logging.getLogger("preprocessing_annotations").setLevel(logging.ERROR)
    emit_report(measure(args.runs, SFTDatasetConfig()), args.out)
