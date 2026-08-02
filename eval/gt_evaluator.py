"""
T-0: Ground-truth evaluation harness against the test_pcs COCO dataset.

Scores CubiCasa5K icon detections (door/window/toilet/sink) against
human-labeled test_pcs annotations. Read-only: does not alter any
detection code path, so it cannot regress pipeline behavior.

Ground-truth scope (verified against test_pcs/coco/result.json):
  - door, window, toilet, sink: real annotations present, scored below.
  - wall: category exists with 0 annotations — reported, not scored.
  - stairs, elevators, rooms/spaces: no category in this dataset at all —
    not covered by this harness; a separate GT source is required for
    Track A (rooms) and the T-B3/T-C1 wall work.

Matching uses only the unambiguous primitives from bbox_metrics.py
(BBox dataclass + iou()) rather than BBox.from_dict()/evaluate_bboxes().
Those wrapper functions guess xyxy-vs-xywh from coordinate magnitude
(`bbox[2] < bbox[0]`), which misclassifies floorplan-scale boxes whose
x1 sits far from the origin (e.g. bbox [4973, 2325, 277, 66] -> guessed
xyxy, actually xywh). Ground truth here is converted from COCO xywh via
BBox.from_xywh() explicitly, sidestepping that guess entirely.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

ImagePathResolver = Callable[[str, Path], Optional[Path]]

import numpy as np
from PIL import Image

from preprocessing_annotations.bbox.bbox_metrics import BBox, iou

logger = logging.getLogger(__name__)


@dataclass
class GTAnnotation:
    bbox: BBox
    category: str


@dataclass
class GTImage:
    image_path: Path
    width: int
    height: int
    annotations: List[GTAnnotation] = field(default_factory=list)


@dataclass
class Detection:
    bbox: BBox
    category: str
    confidence: float = 1.0


@dataclass
class CategoryScore:
    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_ious: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "mean_iou": self.mean_iou,
        }


def _group_annotations_by_image(
    annotations: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group COCO annotations by their image id."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def _build_gt_annotations(
    annotations: List[Dict[str, Any]],
    categories: Dict[int, str],
    category_merge: Dict[str, str],
) -> List[GTAnnotation]:
    """Convert COCO annotations to merged-category GT annotations.

    Categories absent from ``category_merge`` are dropped (unscored).
    """
    result: List[GTAnnotation] = []
    for ann in annotations:
        merged = category_merge.get(categories.get(ann["category_id"]))
        if merged is None:
            continue
        x, y, w, h = ann["bbox"]
        result.append(
            GTAnnotation(bbox=BBox.from_xywh(x, y, w, h), category=merged)
        )
    return result


def resolve_by_basename(file_name: str, images_dir: Path) -> Optional[Path]:
    """Resolve a COCO file_name to `images_dir/basename` (default resolver).

    Only safe when every image's basename is unique within the dataset
    (true for test_pcs; NOT true for every COCO export — e.g. CubiCasa5K's
    export repeats "F1_original.png" for all images, which needs a
    different resolver that preserves the relative folder path instead).
    """
    basename = Path(file_name.replace("\\", "/")).name
    resolved = images_dir / basename
    return resolved if resolved.exists() else None


def _build_gt_image(
    image: Dict[str, Any],
    grouped: Dict[int, List[Dict[str, Any]]],
    categories: Dict[int, str],
    category_merge: Dict[str, str],
    images_dir: Path,
    resolve_path: ImagePathResolver,
) -> Optional[GTImage]:
    """Build one GTImage, or None if its file is missing on disk."""
    resolved_path = resolve_path(image["file_name"], images_dir)
    if resolved_path is None:
        logger.warning("GT image not found, skipping: %s", image["file_name"])
        return None
    annotations = _build_gt_annotations(
        grouped.get(image["id"], []), categories, category_merge
    )
    return GTImage(
        image_path=resolved_path,
        width=image["width"],
        height=image["height"],
        annotations=annotations,
    )


def load_ground_truth(
    coco_path: Path,
    images_dir: Path,
    category_merge: Dict[str, str],
    resolve_path: ImagePathResolver = resolve_by_basename,
) -> List[GTImage]:
    """Load a COCO export as GT images with merged categories.

    Categories absent from ``category_merge`` are dropped; images missing on
    disk are skipped and logged (GT integrity reported, never assumed).

    Args:
        coco_path: Path to the COCO ``result.json``.
        images_dir: Directory holding the referenced images.
        category_merge: Map from raw category name to scored category name.
        resolve_path: Maps a COCO file_name to a real path; override this
            for datasets whose file_name basenames are not unique.

    Returns:
        One GTImage per resolvable image.
    """
    data = json.loads(coco_path.read_text())
    categories = {c["id"]: c["name"] for c in data["categories"]}
    grouped = _group_annotations_by_image(data["annotations"])
    built = (
        _build_gt_image(
            img, grouped, categories, category_merge, images_dir, resolve_path
        )
        for img in data["images"]
    )
    return [gt_image for gt_image in built if gt_image is not None]


def report_unscored_categories(
    coco_path: Path, category_merge: Dict[str, str]
) -> Dict[str, int]:
    """Count GT annotations for categories no detector scores.

    Returns categories absent from ``category_merge`` (e.g. wall, rooms)
    with their annotation counts, so unscored GT is reported explicitly.
    """
    data = json.loads(coco_path.read_text())
    categories = {c["id"]: c["name"] for c in data["categories"]}
    counts: Dict[str, int] = {name: 0 for name in categories.values()}
    for ann in data["annotations"]:
        name = categories.get(ann["category_id"])
        if name is not None:
            counts[name] += 1
    return {
        name: n
        for name, n in counts.items()
        if name not in category_merge
    }


def detect_with_cubicasa(
    detector: Any, image_path: Path
) -> List[Detection]:
    """Run CubiCasa5K on one image, flattened to a Detection list.

    Returns [] if the model is unavailable or inference fails — CubiCasa's
    own methods already catch and log, so nothing is re-wrapped here.
    """
    with Image.open(image_path) as img:
        image_array = np.array(img.convert("RGB"))
    icon_masks = detector.detect_icons(image_array)
    detections: List[Detection] = []
    for category, icon_mask in icon_masks.items():
        for (x1, y1, x2, y2) in icon_mask.bboxes:
            box = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            detections.append(Detection(bbox=box, category=category))
    return detections


# A predictor bound to one already-loaded model: image path in,
# Detections out. score_against_gt_images depends only on this shape
# (Dependency Inversion), not on any specific detector implementation.
PredictFn = Callable[[Path], List[Detection]]


def cubicasa_predict(detector: Any) -> PredictFn:
    """Bind a loaded CubiCasa5KDetector into a PredictFn."""
    def predict(image_path: Path) -> List[Detection]:
        return detect_with_cubicasa(detector, image_path)
    return predict


def _best_gt_match(
    pred: BBox, gt_boxes: List[BBox], matched: Set[int]
) -> Tuple[Optional[int], float]:
    """Return (index, IoU) of the best unmatched GT box for a prediction."""
    best_idx: Optional[int] = None
    best_iou = 0.0
    for idx, gt in enumerate(gt_boxes):
        if idx in matched:
            continue
        candidate = iou(pred, gt)
        if candidate > best_iou:
            best_iou = candidate
            best_idx = idx
    return best_idx, best_iou


def _match_category(
    pred_boxes: List[BBox],
    gt_boxes: List[BBox],
    iou_threshold: float,
    score: CategoryScore,
) -> None:
    """Greedy one-to-one match a single category into ``score``."""
    matched: Set[int] = set()
    for pred in pred_boxes:
        best_idx, best_iou = _best_gt_match(pred, gt_boxes, matched)
        if best_idx is not None and best_iou >= iou_threshold:
            matched.add(best_idx)
            score.true_positives += 1
            score.matched_ious.append(best_iou)
        else:
            score.false_positives += 1
    score.false_negatives += len(gt_boxes) - len(matched)


def score_image(
    predictions: List[Detection],
    ground_truth: List[GTAnnotation],
    iou_threshold: float,
    scores: Dict[str, CategoryScore],
) -> None:
    """Accumulate per-category greedy IoU matches into ``scores``."""
    categories = (
        {gt.category for gt in ground_truth}
        | {p.category for p in predictions}
    )
    for category in categories:
        score = scores.setdefault(
            category, CategoryScore(category=category)
        )
        gt_boxes = [g.bbox for g in ground_truth if g.category == category]
        pred_boxes = [p.bbox for p in predictions if p.category == category]
        _match_category(pred_boxes, gt_boxes, iou_threshold, score)


def measure_cross_category_confusion(
    predictions: List[Detection],
    ground_truth: List[GTAnnotation],
    iou_threshold: float,
    confusion: Dict[str, Dict[str, int]],
) -> None:
    """Accumulate predicted-category -> GT-category IoU match counts.

    AF4. ``score_image`` filters predictions and GT to one category at a
    time, so a prediction that lands squarely on a real object but with
    the wrong label is counted twice as unrelated errors -- a false
    positive for the label it claimed, a false negative for the label it
    should have had -- and never as a class confusion. That makes
    "detector missed this object entirely" and "detector found it and
    mislabeled it" numerically indistinguishable, which is exactly the
    distinction needed to explain the door/window mix-ups observed on
    real MEP sheets.

    This is a sibling of ``score_image``, not a change to it: that
    function has two independent callers (this module's
    ``score_against_gt_images`` and ``cubicasa_room_eval``), so every
    recorded baseline in this project depends on its accumulation staying
    exactly as-is.

    Off-diagonal entries (``confusion[pred_category][other_category]``)
    are the class confusions. Diagonal entries are NOT the true positives
    ``score_image`` reports and must not be read as such: here each
    prediction competes against ground truth of every category at once,
    so a prediction can be won by a nearer box of a different class than
    it would have matched in the single-category pass.

    Matching is greedy one-to-one over the prediction list in input
    order, reusing ``_best_gt_match`` -- the same primitive and therefore
    the same order-dependence ``score_image`` already has (confidence is
    not consulted; callers that want confidence priority must sort
    ``predictions`` before calling).

    Nested plain dicts (not tuple keys) so the result serializes to JSON
    directly, matching how the eval CLIs already emit their reports.
    """
    gt_boxes = [annotation.bbox for annotation in ground_truth]
    gt_categories = [annotation.category for annotation in ground_truth]
    matched: Set[int] = set()
    for prediction in predictions:
        best_idx, best_iou = _best_gt_match(prediction.bbox, gt_boxes, matched)
        if best_idx is None or best_iou < iou_threshold:
            continue
        matched.add(best_idx)
        row = confusion.setdefault(prediction.category, {})
        gt_category = gt_categories[best_idx]
        row[gt_category] = row.get(gt_category, 0) + 1


def score_against_gt_images(
    gt_images: List[GTImage],
    iou_threshold: float,
    predict_fn: PredictFn,
    confusion: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, CategoryScore]:
    """Score any predictor against already-loaded GT images.

    Shared by any GT source (COCO test_pcs, YOLO Kaggle, ...) once its
    loader has produced ``GTImage`` objects, and any detector once it is
    bound to a ``PredictFn`` (e.g. ``cubicasa_predict``) -- loading,
    scoring, and the model under test are three independent concerns.

    AF5: pass a dict as ``confusion`` to also accumulate
    ``measure_cross_category_confusion`` over the same pass. Optional and
    ``None`` by default, so callers that do not ask for it are unaffected.
    Filling it here rather than in a second driver loop matters because
    ``predict_fn`` is the expensive part -- a separate confusion driver
    would re-run inference over every image to learn nothing new.
    """
    scores: Dict[str, CategoryScore] = {}
    for gt_image in gt_images:
        try:
            predictions = predict_fn(gt_image.image_path)
        except (OSError, ValueError) as error:
            logger.warning(
                "skipped %s: %s", gt_image.image_path.name, error
            )
            continue
        score_image(predictions, gt_image.annotations, iou_threshold, scores)
        if confusion is not None:
            measure_cross_category_confusion(
                predictions, gt_image.annotations, iou_threshold, confusion
            )
    return scores


def evaluate(
    gt_coco_path: Path,
    gt_images_dir: Path,
    category_merge: Dict[str, str],
    iou_threshold: float,
    detector: Any,
) -> Dict[str, Any]:
    """Run the full T-0 baseline: CubiCasa vs test_pcs GT, per category."""
    gt_images = load_ground_truth(gt_coco_path, gt_images_dir, category_merge)
    scores = score_against_gt_images(
        gt_images, iou_threshold, cubicasa_predict(detector)
    )

    unscored = report_unscored_categories(gt_coco_path, category_merge)
    result = {cat: score.to_dict() for cat, score in scores.items()}
    result["_unscored_categories_no_detector"] = unscored
    result["_images_evaluated"] = len(gt_images)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="T-0: score CubiCasa5K vs test_pcs GT"
    )
    parser.add_argument(
        "--coco", type=Path, default=None, help="Override GT COCO json path"
    )
    parser.add_argument(
        "--images", type=Path, default=None, help="Override GT images dir"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=None,
        help="Override IoU match threshold",
    )
    args = parser.parse_args()

    from preprocessing_annotations.config import EvalConfig

    eval_config = EvalConfig()
    coco_path = args.coco or eval_config.gt_coco_path
    images_dir = args.images or eval_config.gt_images_dir
    iou_threshold = eval_config.gt_iou_threshold
    if args.iou_threshold is not None:
        iou_threshold = args.iou_threshold

    try:
        from cubicasa5k_detector import CubiCasa5KDetector
    except ImportError:
        from preprocessing_annotations.detection.cubicasa5k_detector import (
            CubiCasa5KDetector,
        )

    cubicasa_detector = CubiCasa5KDetector(device=eval_config.device)

    results = evaluate(
        gt_coco_path=coco_path,
        gt_images_dir=images_dir,
        category_merge=eval_config.gt_category_merge,
        iou_threshold=iou_threshold,
        detector=cubicasa_detector,
    )

    print(json.dumps(results, indent=2))
