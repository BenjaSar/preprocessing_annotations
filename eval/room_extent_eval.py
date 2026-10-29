"""Score PDF-vector-wall room extents (orchestration/room_extent_step.py) on a
finished run, against room GT.

Applies the pipeline's own RoomExtentStep to each processed page's final
rooms -- the same code a run with --use-vector-room-extent executes in
Step 4 -- so a configuration can be judged without a fresh run. Box sets
scored with room_gt_eval.score_localization:

  label            the rooms' label boxes (what every run outputs today)
  extent_only      the extent box of rooms that got one
  extent_or_label  extent box where present, label box otherwise

`extent_precision` / `extent_recall` there are box IoU >= gt_iou_threshold
against the GT room's bounding box. `polygon_iou_matched` additionally counts
extents whose outline reaches that IoU with the GT polygon holding the label
centre. Coverage counts every page of the run, GT or not.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from preprocessing_annotations.config import RoomExtentConfig, RoomGtEvalConfig
from preprocessing_annotations.ingestion.pdf_extractor import page_source
from preprocessing_annotations.ingestion.vector_walls import VectorWallError, VectorWallReader
from preprocessing_annotations.orchestration.room_extent_step import RoomExtentStep

from report_io import emit_report
from room_gt_eval import (
    BoxXYXY,
    LocalizationScore,
    load_gt_polygons,
    page_key,
    room_gt_index,
    score_localization,
)
from room_replay import document_key, parse_overrides

BOX_SETS = ("label", "extent_only", "extent_or_label")


def _source_page(stem: str, pdf_dir: Path) -> Optional[Tuple[Path, int]]:
    source = page_source(stem)
    if source is None or not (pdf_dir / f"{source[0]}.pdf").is_file():
        return None
    return pdf_dir / f"{source[0]}.pdf", source[1]


def _box_sets(rooms: List[Dict[str, Any]]) -> Dict[str, List[BoxXYXY]]:
    labels = [tuple(r["coordinates"]["bbox"]) for r in rooms]
    extents = [tuple(r["coordinates"]["extent"]["bbox"]) if "extent" in r["coordinates"] else None for r in rooms]
    return {
        "label": labels,
        "extent_only": [e for e in extents if e is not None],
        "extent_or_label": [e if e is not None else l for e, l in zip(extents, labels)],
    }


def _polygon_iou_matched(rooms: List[Dict[str, Any]], gts: List[Polygon], threshold: float) -> int:
    labels = [tuple(r["coordinates"]["bbox"]) for r in rooms]
    matched = 0
    for room, gt_index in zip(rooms, room_gt_index(labels, gts)):
        extent = room["coordinates"].get("extent")
        if extent is None or gt_index is None:
            continue
        outline = Polygon(extent["polygon"]).buffer(0)
        gt = gts[gt_index]
        union = outline.union(gt).area
        matched += union > 0 and outline.intersection(gt).area / union >= threshold
    return matched


def evaluate(
    run_dir: Path, pdf_dir: Path, extent_config: RoomExtentConfig, eval_config: RoomGtEvalConfig,
) -> Dict[str, Any]:
    gt = load_gt_polygons(eval_config.gt_coco_path)
    step = RoomExtentStep(VectorWallReader(extent_config))
    coverage: Counter = Counter()
    overall = {name: LocalizationScore() for name in BOX_SETS}
    per_document: Dict[str, Dict[str, LocalizationScore]] = {}
    polygon_matched = 0
    for path in sorted((run_dir / "processed_annotations").glob("*.json")):
        annotation = json.loads(path.read_text())
        rooms = annotation.get("roomsRecognized", [])
        coverage["pages"] += 1
        source = _source_page(path.stem, pdf_dir)
        if source is None:
            coverage["pages_without_source_pdf"] += 1
            continue
        size = annotation["image_size"]
        try:
            attached = step.attach(rooms, *source, (size["width"], size["height"]))
        except VectorWallError:
            coverage["pages_unreadable"] += 1
            continue
        coverage["rooms"] += len(rooms)
        coverage["rooms_with_extent"] += attached
        coverage["pages_with_extent"] += attached > 0
        key = page_key(path.name)
        if key not in gt:
            continue
        document = per_document.setdefault(
            document_key(key), {name: LocalizationScore() for name in BOX_SETS}
        )
        for name, boxes in _box_sets(rooms).items():
            score_localization(boxes, gt[key], eval_config.gt_iou_threshold, overall[name])
            score_localization(boxes, gt[key], eval_config.gt_iou_threshold, document[name])
        polygon_matched += _polygon_iou_matched(rooms, gt[key], eval_config.gt_iou_threshold)
    return {
        "run_dir": str(run_dir),
        "room_extent": extent_config.__dict__,
        "iou_threshold": eval_config.gt_iou_threshold,
        "coverage": dict(coverage),
        "polygon_iou_matched": polygon_matched,
        "localization": {name: score.to_dict() for name, score in overall.items()},
        "per_document": {
            doc: {name: score.to_dict() for name, score in scores.items()}
            for doc, scores in sorted(per_document.items())
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score PDF vector-wall room extents on a finished run")
    parser.add_argument("--run", type=Path, required=True, help="Run dir with processed_annotations")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory holding the run's source PDFs")
    parser.add_argument("--extent", nargs="*", default=[], metavar="KEY=VALUE", help="RoomExtentConfig overrides")
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    logging.getLogger("preprocessing_annotations").setLevel(logging.ERROR)
    emit_report(
        evaluate(args.run, args.pdf_dir, parse_overrides(RoomExtentConfig(), args.extent), RoomGtEvalConfig()),
        args.out,
    )
