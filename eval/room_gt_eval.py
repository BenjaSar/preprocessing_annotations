"""AV-8 (2026-10-10): score a pipeline run against room-polygon ground truth.

First room GT for this project's target domain (RoomGtEvalConfig,
gt_segmentation/result.json: 48 pages, 1497 room polygons). Read-only --
touches no pipeline code path, so it cannot regress pipeline behavior.

Four measurements per run:
  - localization: does each predicted room's center fall inside a GT room
    polygon (the label-location objective), and how many GT rooms get one
  - extent: bbox IoU of each prediction against its best GT room
  - extent_size (2026-10-14): box area over the area of the GT room it sits
    in. IoU is undirected -- a label-scale box and a box that swallows its
    neighbours both score 0 -- so oversize is invisible in `extent` alone
  - attachment: per AttachmentPolicy, is each door/window attached to a
    room whose GT polygon the detection actually touches

Detection source, in order of preference:
  - persisted: a run's raw annotations carry `objectDetections` (attached to
    a room) plus `unattachedObjectDetections` (N2, 2026-10-11) -- together
    the complete record. `objectDetections` alone is pre-filtered by the very
    attachment policy being scored (sprint1_verify51's 48 GT pages: 310
    persisted of 1496 detected).
  - log: for runs that predate N2, the "YOLO object ..." debug line
    (pipeline._detect_object_mappings) is the only complete record;
    setup_logging's file handler is always DEBUG.
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, box

from preprocessing_annotations.config import AttachmentConfig, RoomGtEvalConfig
from preprocessing_annotations.detection.attachment import (
    AttachmentPolicy,
    AttachmentPolicyFactory,
)
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
)

from report_io import emit_report

logger = logging.getLogger(__name__)

BoxXYXY = Tuple[float, float, float, float]

# Label Studio prefixes every uploaded file with an 8-hex-char id + "-".
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")
# Upload also rewrites spaces/punctuation; compare on alphanumerics only.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_LOG_PAGE_HEADER = re.compile(r"SAM3 exemplar: \d+ door seeds .* for (?P<image>.+)\s*$")
_LOG_DETECTION = re.compile(
    r"YOLO object (?P<type>\w+): bbox=\((?P<x1>[\d.]+), (?P<y1>[\d.]+), "
    r"(?P<x2>[\d.]+), (?P<y2>[\d.]+)\) conf=(?P<conf>[\d.]+)"
)


def page_key(file_name: str) -> str:
    """Normalize a GT/run/log file name to one comparable page key."""
    basename = Path(file_name.replace("\\", "/")).stem
    basename = _UPLOAD_PREFIX.sub("", basename)
    return _NON_ALNUM.sub("_", basename).strip("_").lower()


def load_gt_polygons(coco_path: Path) -> Dict[str, List[Polygon]]:
    """GT room polygons per page key."""
    data = json.loads(coco_path.read_text())
    key_by_image = {img["id"]: page_key(img["file_name"]) for img in data["images"]}
    polygons: Dict[str, List[Polygon]] = {key: [] for key in key_by_image.values()}
    for ann in data["annotations"]:
        flat = ann["segmentation"][0]
        polygon = Polygon(list(zip(flat[0::2], flat[1::2]))).buffer(0)
        polygons[key_by_image[ann["image_id"]]].append(polygon)
    return polygons


def load_gt_categories(coco_path: Path) -> Dict[str, List[str]]:
    """GT room category names per page key, aligned index-for-index with
    load_gt_polygons."""
    data = json.loads(coco_path.read_text())
    key_by_image = {img["id"]: page_key(img["file_name"]) for img in data["images"]}
    name_by_id = {cat["id"]: cat["name"] for cat in data["categories"]}
    categories: Dict[str, List[str]] = {key: [] for key in key_by_image.values()}
    for ann in data["annotations"]:
        categories[key_by_image[ann["image_id"]]].append(name_by_id[ann["category_id"]])
    return categories


def xywh_to_xyxy(bbox: Sequence[float]) -> BoxXYXY:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


class PredictedRoom(NamedTuple):
    """One predicted room: where it is, and what class it was given."""

    box: BoxXYXY
    room_type: Optional[str]


def boxes_of(rooms: Sequence[PredictedRoom]) -> List[BoxXYXY]:
    return [room.box for room in rooms]


def load_run_rooms(run_dir: Path) -> Dict[str, List[PredictedRoom]]:
    """Final (post-processed) rooms per page key; boxes xyxy."""
    rooms: Dict[str, List[PredictedRoom]] = {}
    for path in sorted((run_dir / "processed_annotations").glob("*.json")):
        annotation = json.loads(path.read_text())
        rooms[page_key(path.name)] = [
            PredictedRoom(tuple(room["coordinates"]["bbox"]), room.get("type"))
            for room in annotation.get("roomsRecognized", [])
        ]
    return rooms


def load_gt_category_map(path: Path) -> Dict[str, Any]:
    """Reviewed GT-category -> taxonomy-class map (see the drafting script)."""
    return json.loads(Path(path).read_text())


def load_logged_detections(log_path: Path) -> Dict[str, List[WindowDetection]]:
    """Every door/window detection per page key, from the run's debug log."""
    detections: Dict[str, List[WindowDetection]] = {}
    current: Optional[str] = None
    with open(log_path, errors="replace") as log:
        for line in log:
            header = _LOG_PAGE_HEADER.search(line)
            if header:
                current = page_key(header.group("image"))
                detections.setdefault(current, [])
                continue
            match = _LOG_DETECTION.search(line)
            if match and current is not None:
                detections[current].append(_detection_from_match(match))
    return detections


def load_persisted_detections(run_dir: Path) -> Dict[str, List[WindowDetection]]:
    """Every door/window detection per page key, from a run's raw
    annotations (attached + unattached). Raises if a GT-page annotation
    lacks `unattachedObjectDetections`, i.e. the run predates N2 and its
    persisted list is incomplete -- use load_logged_detections for those."""
    detections: Dict[str, List[WindowDetection]] = {}
    for path in sorted((run_dir / "annotations").glob("*.json")):
        annotation = json.loads(path.read_text())
        if "unattachedObjectDetections" not in annotation:
            raise ValueError(
                f"{path.name}: no unattachedObjectDetections -- persisted "
                f"detections are incomplete for this run, use the debug log"
            )
        detections[page_key(path.name)] = [
            _detection_from_record(record)
            for record in annotation.get("objectDetections", [])
            + annotation["unattachedObjectDetections"]
        ]
    return detections


def _detection_from_record(record: Dict[str, Any]) -> WindowDetection:
    return WindowDetection(
        bbox=tuple(record["bbox"]),
        confidence=record["confidence"],
        source_tier=WindowDetectionTier(record["source_tier"]),
        metadata={"type": record["category"]},
    )


def _detection_from_match(match: "re.Match[str]") -> WindowDetection:
    return WindowDetection(
        bbox=tuple(float(match.group(k)) for k in ("x1", "y1", "x2", "y2")),
        confidence=float(match.group("conf")),
        source_tier=WindowDetectionTier.YOLO,
        metadata={"type": match.group("type")},
    )


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def room_gt_index(rooms: List[BoxXYXY], gts: List[Polygon]) -> List[Optional[int]]:
    """Per room: smallest GT polygon containing the room's center, or None."""
    index: List[Optional[int]] = []
    for room in rooms:
        center = Point(_center(room))
        containing = [i for i, gt in enumerate(gts) if gt.contains(center)]
        index.append(min(containing, key=lambda i: gts[i].area) if containing else None)
    return index


def _touched_gt(detection: WindowDetection, gts: List[Polygon]) -> Set[int]:
    footprint = box(*detection.bbox)
    return {i for i, gt in enumerate(gts) if gt.intersects(footprint)}


@dataclass
class LocalizationScore:
    predictions: int = 0
    predictions_center_in_gt: int = 0
    predictions_iou_match: int = 0
    gt_rooms: int = 0
    gt_rooms_located: int = 0
    gt_rooms_iou_matched: int = 0
    prediction_best_iou_sum: float = 0.0
    gt_best_iou_sum: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        label_precision = _ratio(self.predictions_center_in_gt, self.predictions)
        label_recall = _ratio(self.gt_rooms_located, self.gt_rooms)
        extent_precision = _ratio(self.predictions_iou_match, self.predictions)
        extent_recall = _ratio(self.gt_rooms_iou_matched, self.gt_rooms)
        return {
            **self.__dict__,
            "label_precision": label_precision,
            "label_recall": label_recall,
            "label_f1": _f1(label_precision, label_recall),
            "label_fdr": 1.0 - label_precision if self.predictions else 0.0,
            "label_fnr": 1.0 - label_recall if self.gt_rooms else 0.0,
            "extent_precision": extent_precision,
            "extent_recall": extent_recall,
            "extent_f1": _f1(extent_precision, extent_recall),
            "mean_iou_per_prediction": self.prediction_best_iou_sum / self.predictions
            if self.predictions else 0.0,
            "mean_iou_per_gt_room": self.gt_best_iou_sum / self.gt_rooms
            if self.gt_rooms else 0.0,
        }


@dataclass
class AttachmentScore:
    correct: int = 0
    wrong: int = 0
    attachable: int = 0
    correctly_attached: int = 0

    def to_dict(self) -> Dict[str, Any]:
        decided = self.correct + self.wrong
        precision = _ratio(self.correct, decided)
        recall = _ratio(self.correctly_attached, self.attachable)
        return {
            **self.__dict__,
            "precision": precision,
            "recall_of_attachable": recall,
            "f1": _f1(precision, recall),
            "fdr": 1.0 - precision if decided else 0.0,
            "fnr": 1.0 - recall if self.attachable else 0.0,
        }


@dataclass
class CategoryScore:
    """Does a located room also carry the right class?

    Only rooms whose centre lies in a GT polygon are scored -- a room in the
    wrong place has no GT class to be judged against. Rooms whose GT category
    the map leaves unmapped are excluded and counted, never silently right.
    """

    correct: int = 0
    wrong: int = 0
    excluded_unmapped: int = 0
    excluded_untyped: int = 0
    confusions: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "wrong": self.wrong,
            "excluded_unmapped": self.excluded_unmapped,
            "excluded_untyped": self.excluded_untyped,
            "accuracy": _ratio(self.correct, self.correct + self.wrong),
            "top_confusions": [
                {"gt_class": gt, "predicted": predicted, "rooms": count}
                for (gt, predicted), count in self.confusions.most_common(15)
            ],
        }


@dataclass
class ExtentScore:
    """How big is each located room's box against the GT room it sits in?

    `LocalizationScore` already reports bbox IoU, but IoU is undirected: a
    label-scale box (far too small) and a box that swallows its neighbours
    both score 0, so neither an oversize regression nor an oversize fix is
    visible there. This separates them by area ratio, the only direction
    that matters for export: oversized boxes are not allowed.

    Only located rooms are scored -- a room in the wrong place has no GT
    room whose size it could be judged against.
    """

    located: int = 0
    oversize: int = 0
    area_ratios: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "located": self.located,
            "oversize": self.oversize,
            "oversize_rate": _ratio(self.oversize, self.located),
            **_ratio_percentiles(self.area_ratios),
        }


def _ratio_percentiles(ratios: Sequence[float]) -> Dict[str, float]:
    if not ratios:
        return {"median_area_ratio": 0.0, "p90_area_ratio": 0.0, "max_area_ratio": 0.0}
    return {
        "median_area_ratio": float(np.percentile(ratios, 50)),
        "p90_area_ratio": float(np.percentile(ratios, 90)),
        "max_area_ratio": float(max(ratios)),
    }


def score_extent(
    rooms: List[BoxXYXY], gts: List[Polygon], max_area_ratio: float, score: ExtentScore,
) -> None:
    """Accumulate one page's box-size counts into score."""
    for room, gt_index in zip(rooms, room_gt_index(rooms, gts)):
        if gt_index is None or not gts[gt_index].area:
            continue
        ratio = box(*room).area / gts[gt_index].area
        score.located += 1
        score.area_ratios.append(ratio)
        if ratio > max_area_ratio:
            score.oversize += 1


def score_category(
    rooms: Sequence[PredictedRoom], gts: List[Polygon], gt_categories: Sequence[str],
    class_of_category: Mapping[str, Optional[str]], score: CategoryScore,
) -> None:
    """Accumulate one page's category counts into score."""
    for room, gt_index in zip(rooms, room_gt_index(boxes_of(rooms), gts)):
        if gt_index is None:
            continue
        expected = class_of_category.get(gt_categories[gt_index])
        if expected is None:
            score.excluded_unmapped += 1
        elif room.room_type is None:
            score.excluded_untyped += 1
        elif room.room_type == expected:
            score.correct += 1
        else:
            score.wrong += 1
            score.confusions[(expected, room.room_type)] += 1


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bbox_iou(room: BoxXYXY, gt: Polygon) -> float:
    gt_box = box(*gt.bounds)
    union = box(*room).union(gt_box).area
    return box(*room).intersection(gt_box).area / union if union else 0.0


def score_localization(
    rooms: List[BoxXYXY], gts: List[Polygon], iou_threshold: float,
    score: LocalizationScore,
) -> None:
    """Accumulate one page's localization/extent counts into score."""
    gt_of_room = room_gt_index(rooms, gts)
    located = {i for i in gt_of_room if i is not None}
    iou_matched: Set[int] = set()
    gt_best = [0.0] * len(gts)
    for room in rooms:
        ious = [_bbox_iou(room, gt) for gt in gts]
        best = max(zip(ious, range(len(gts))), default=(0.0, None))
        score.prediction_best_iou_sum += best[0]
        gt_best = [max(a, b) for a, b in zip(gt_best, ious)]
        if best[0] >= iou_threshold:
            score.predictions_iou_match += 1
            iou_matched.add(best[1])
    score.gt_best_iou_sum += sum(gt_best)
    score.predictions += len(rooms)
    score.predictions_center_in_gt += sum(1 for i in gt_of_room if i is not None)
    score.gt_rooms += len(gts)
    score.gt_rooms_located += len(located)
    score.gt_rooms_iou_matched += len(iou_matched)


def score_attachment(
    policy: AttachmentPolicy, detections: List[WindowDetection],
    rooms: List[BoxXYXY], gts: List[Polygon], score: AttachmentScore,
) -> None:
    """Accumulate one page's attachment counts into score. A detection is
    correctly attached when its room's GT polygon is one the detection
    touches (a door on a shared wall is correct for either room)."""
    gt_of_room = room_gt_index(rooms, gts)
    reachable = {i for i in gt_of_room if i is not None}
    touched = {id(d): _touched_gt(d, gts) for d in detections}
    score.attachable += sum(1 for d in detections if touched[id(d)] & reachable)

    room_dicts = [{"id": i, "bbox": list(room)} for i, room in enumerate(rooms)]
    correct_ids: Set[int] = set()
    for mapping in policy.attach(detections, room_dicts):
        gt_index = gt_of_room[mapping.room_id]
        for detection in mapping.intersecting_windows:
            if gt_index is not None and gt_index in touched[id(detection)]:
                score.correct += 1
                correct_ids.add(id(detection))
            else:
                score.wrong += 1
    score.correctly_attached += len(correct_ids)


def correct_pair_distances(
    detections: List[WindowDetection], rooms: List[BoxXYXY], gts: List[Polygon],
) -> List[float]:
    """Detection-center -> room-label-center distance for every (detection,
    room) pair whose room's GT polygon the detection touches."""
    gt_of_room = room_gt_index(rooms, gts)
    distances: List[float] = []
    for detection in detections:
        touched = _touched_gt(detection, gts)
        dx, dy = _center(detection.bbox)
        for room, gt_index in zip(rooms, gt_of_room):
            if gt_index in touched:
                rx, ry = _center(room)
                distances.append(math.hypot(rx - dx, ry - dy))
    return distances


@dataclass
class _ScoredPage:
    rooms: List[PredictedRoom]
    gts: List[Polygon]
    gt_categories: List[str] = field(default_factory=list)
    detections: List[WindowDetection] = field(default_factory=list)


def _join_pages(
    gt: Dict[str, List[Polygon]], rooms: Dict[str, List[PredictedRoom]],
    detections: Dict[str, List[WindowDetection]],
    gt_categories: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[_ScoredPage], List[str]]:
    """Pages present in both GT and the run; GT pages the run lacks are
    returned separately (reported, never silently dropped)."""
    categories = gt_categories or {}
    pages = [
        _ScoredPage(rooms[key], gts, categories.get(key, []), detections.get(key, []))
        for key, gts in gt.items() if key in rooms
    ]
    missing = sorted(key for key in gt if key not in rooms)
    return pages, missing


def _attachment_configs(
    pages: List[_ScoredPage], percentiles: Sequence[int],
) -> Dict[str, AttachmentConfig]:
    """Default policy plus nearest_with_cutoff at data-derived cutoffs."""
    distances = [
        d for page in pages
        for d in correct_pair_distances(page.detections, boxes_of(page.rooms), page.gts)
    ]
    configs = {"intersection": AttachmentConfig()}
    if not distances:
        return configs
    for percentile in percentiles:
        cutoff = float(np.percentile(distances, percentile))
        configs[f"nearest_with_cutoff_p{percentile}_{round(cutoff)}px"] = AttachmentConfig(
            policy="nearest_with_cutoff", nearest_cutoff_px=cutoff,
        )
    return configs


def evaluate(
    config: RoomGtEvalConfig, run_dir: Path,
    detections: Dict[str, List[WindowDetection]],
) -> Dict[str, Any]:
    category_map = load_gt_category_map(config.gt_category_map_path)
    pages, missing = _join_pages(
        load_gt_polygons(config.gt_coco_path), load_run_rooms(run_dir), detections,
        load_gt_categories(config.gt_coco_path),
    )
    localization = LocalizationScore()
    extent = ExtentScore()
    category = CategoryScore()
    for page in pages:
        score_localization(
            boxes_of(page.rooms), page.gts, config.gt_iou_threshold, localization
        )
        score_extent(
            boxes_of(page.rooms), page.gts, config.max_box_to_gt_area_ratio, extent
        )
        score_category(
            page.rooms, page.gts, page.gt_categories, category_map["mapping"], category
        )

    attachment: Dict[str, Dict[str, Any]] = {}
    for name, attach_config in _attachment_configs(
        pages, config.attachment_cutoff_percentiles
    ).items():
        policy = AttachmentPolicyFactory.create(attach_config)
        score = AttachmentScore()
        for page in pages:
            score_attachment(
                policy, page.detections, boxes_of(page.rooms), page.gts, score
            )
        attachment[name] = score.to_dict()

    return {
        "run_dir": str(run_dir),
        "gt_coco_path": str(config.gt_coco_path),
        "pages_scored": len(pages),
        "gt_pages_missing_from_run": missing,
        "detections_scored": sum(len(p.detections) for p in pages),
        "iou_threshold": config.gt_iou_threshold,
        "localization": localization.to_dict(),
        "extent_size": {
            "max_box_to_gt_area_ratio": config.max_box_to_gt_area_ratio,
            **extent.to_dict(),
        },
        "category": {
            "gt_category_map_path": str(config.gt_category_map_path),
            "category_map_reviewed": category_map.get("reviewed", False),
            **category.to_dict(),
        },
        "attachment": attachment,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="AV-8: score a run against room-polygon GT")
    parser.add_argument("--run", type=Path, required=True, help="Pipeline output dir")
    parser.add_argument(
        "--log", type=Path, default=None,
        help="That run's pipeline_*.log; required only for runs without unattachedObjectDetections",
    )
    parser.add_argument("--coco", type=Path, default=None, help="Override GT COCO json path")
    parser.add_argument("--iou-threshold", type=float, default=None, help="Override IoU threshold")
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    eval_config = RoomGtEvalConfig()
    if args.coco is not None:
        eval_config.gt_coco_path = args.coco
    if args.iou_threshold is not None:
        eval_config.gt_iou_threshold = args.iou_threshold

    if args.log is not None:
        run_detections = load_logged_detections(args.log)
    else:
        run_detections = load_persisted_detections(args.run)
    emit_report(evaluate(eval_config, args.run, run_detections), args.out)
