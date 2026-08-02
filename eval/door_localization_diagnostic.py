"""T-KG1b / R0b: diagnose WHY a detector tier scores 0 (or near-0) TP@0.5
for a category -- originally built for CubiCasa5K doors, now tier-generic
(R0b) so the same diagnostic covers the YOLO tiers too.

T-KG1 found door = 0 true positives at IoU>=0.5 on both GT sources, yet
the detector emitted 380 door boxes -- cause unestablished. This
read-only diagnostic classifies each GT box against its best-IoU
same-category prediction to separate the candidate causes:

  * pure_miss  (best IoU == 0)      -> no overlapping prediction.
  * near_miss  (0 < best IoU < 0.5) -> overlaps but under threshold
    (small-object / strict-IoU, or a systematic bbox-convention shift).
  * matched    (best IoU >= 0.5).

Reports the per-GT best-IoU bucket counts, the near-miss area ratio
(pred/gt, exposing a size-convention mismatch), and a threshold sweep
(does recall recover at a looser IoU?). Per-GT best-IoU is an
upper-bound proxy, not the greedy one-to-one score -- diagnostic only.

R0b: the tier is selected via --tier, reusing kaggle_door_window_eval.py's
own TIER_BUILDERS registry (cubicasa/vlm/yolo_pretrained/yolo_finetuned)
instead of a hardcoded CubiCasa5KDetector -- one registry, not two.
"""

import argparse
import json
import logging
import statistics
from typing import Callable, Dict, List, Optional, Tuple

from preprocessing_annotations.bbox.bbox_metrics import BBox, iou
from preprocessing_annotations.config import (
    EvalConfig,
    FloorplancadEvalConfig,
    KaggleFloorplanEvalConfig,
)
from floorplancad_gt import ensure_images, load_floorplancad_ground_truth
from gt_evaluator import Detection, GTImage, PredictFn, load_ground_truth
from kaggle_door_window_eval import DEFAULT_TIER, TIER_BUILDERS
from yolo_gt import load_kaggle_floorplan_ground_truth

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY = "door"
_IOU_SWEEP: Tuple[float, ...] = (0.3, 0.4, 0.5)
_ZERO_IOU = 0.0
_MATCHED = "matched"
_NEAR_MISS = "near_miss"
_PURE_MISS = "pure_miss"
_VERDICT_THRESHOLD = 0.5
_DATASET_TEST_PCS = "test_pcs"
_DATASET_KAGGLE = "kaggle_floorplans500"
_DATASET_FLOORPLANCAD = "floorplancad"
_ALL_DATASETS = "all"

GtLoader = Callable[[argparse.Namespace], List[GTImage]]


def _cap(
    gt_images: List[GTImage], sample_size: Optional[int]
) -> List[GTImage]:
    """Return at most `sample_size` images (all if None)."""
    if sample_size is None:
        return gt_images
    return gt_images[:sample_size]


def _load_test_pcs(_: argparse.Namespace) -> List[GTImage]:
    """Load the test_pcs COCO GT (commercial domain)."""
    config = EvalConfig()
    return load_ground_truth(
        config.gt_coco_path, config.gt_images_dir, config.gt_category_merge
    )


def _load_kaggle(args: argparse.Namespace) -> List[GTImage]:
    """Load the Kaggle floor-plans-500 YOLO GT for the chosen split."""
    return load_kaggle_floorplan_ground_truth(
        KaggleFloorplanEvalConfig(split=args.kaggle_split)
    )


def _load_floorplancad(args: argparse.Namespace) -> List[GTImage]:
    """Load a sampled FloorPlanCAD GT and fetch its images (commercial)."""
    config = FloorplancadEvalConfig()
    images = _cap(load_floorplancad_ground_truth(config), args.sample_size)
    ensure_images(images, config)
    return images


_GT_LOADERS: Dict[str, GtLoader] = {
    _DATASET_TEST_PCS: _load_test_pcs,
    _DATASET_KAGGLE: _load_kaggle,
    _DATASET_FLOORPLANCAD: _load_floorplancad,
}


def _category_pred_boxes(
    predictions: List[Detection], category: str
) -> List[BBox]:
    """Return prediction boxes for one category."""
    return [p.bbox for p in predictions if p.category == category]


def _best_iou(gt_box: BBox, pred_boxes: List[BBox]) -> float:
    """Return the largest IoU of any prediction against a GT box."""
    return max((iou(gt_box, p) for p in pred_boxes), default=_ZERO_IOU)


def _classify(best: float, threshold: float) -> str:
    """Bucket one GT box by its best prediction IoU (guard clauses)."""
    if best >= threshold:
        return _MATCHED
    if best > _ZERO_IOU:
        return _NEAR_MISS
    return _PURE_MISS


def _best_pred(gt_box: BBox, pred_boxes: List[BBox]) -> Optional[BBox]:
    """Return the single highest-IoU prediction box, if any overlaps."""
    if not pred_boxes:
        return None
    return max(pred_boxes, key=lambda p: iou(gt_box, p))


def _accumulate_gt_box(
    gt_box: BBox,
    pred_boxes: List[BBox],
    buckets: Dict[str, int],
    near_ratios: List[float],
    best_ious: List[float],
) -> None:
    """Classify one GT box and record its bucket, best IoU, area ratio."""
    best = _best_iou(gt_box, pred_boxes)
    best_ious.append(best)
    bucket = _classify(best, _VERDICT_THRESHOLD)
    buckets[bucket] = buckets.get(bucket, 0) + 1
    if bucket != _NEAR_MISS:
        return
    pred = _best_pred(gt_box, pred_boxes)
    if pred is not None and gt_box.area > 0:
        near_ratios.append(pred.area / gt_box.area)


def _sweep_recall(best_ious: List[float]) -> Dict[str, float]:
    """Per-GT best-IoU recall proxy at each swept threshold."""
    total = len(best_ious)
    if total == 0:
        return {str(t): 0.0 for t in _IOU_SWEEP}
    return {
        str(t): sum(1 for v in best_ious if v >= t) / total
        for t in _IOU_SWEEP
    }


def _diagnose(
    predict_fn: PredictFn,
    gt_images: List[GTImage],
    category: str,
) -> Dict[str, object]:
    """Run the per-GT best-IoU diagnostic for one category."""
    buckets: Dict[str, int] = {}
    near_ratios: List[float] = []
    best_ious: List[float] = []
    for gt_image in gt_images:
        preds = predict_fn(gt_image.image_path)
        pred_boxes = _category_pred_boxes(preds, category)
        for ann in gt_image.annotations:
            if ann.category != category:
                continue
            _accumulate_gt_box(
                ann.bbox, pred_boxes, buckets, near_ratios, best_ious
            )
    return _summarize(buckets, near_ratios, best_ious, len(gt_images))


def _summarize(
    buckets: Dict[str, int],
    near_ratios: List[float],
    best_ious: List[float],
    images: int,
) -> Dict[str, object]:
    """Assemble the diagnostic result payload."""
    median_ratio = (
        statistics.median(near_ratios) if near_ratios else None
    )
    return {
        "images_evaluated": images,
        "gt_boxes": len(best_ious),
        "buckets": buckets,
        "near_miss_area_ratio_median": median_ratio,
        "best_iou_recall_sweep": _sweep_recall(best_ious),
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI args: tier, category, dataset selection, split, sample cap."""
    parser = argparse.ArgumentParser(
        description="T-KG1b/R0b: detector-tier localization failure diagnostic"
    )
    parser.add_argument(
        "--tier", choices=tuple(TIER_BUILDERS), default=DEFAULT_TIER,
    )
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument(
        "--dataset", choices=(*_GT_LOADERS, _ALL_DATASETS),
        default=_ALL_DATASETS,
    )
    parser.add_argument("--kaggle-split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--yolo-checkpoint", default=None,
        help="Path to a yolo_train.py checkpoint, required for "
             "--tier yolo_finetuned",
    )
    return parser.parse_args()


def _selected_datasets(choice: str) -> List[str]:
    """Resolve the --dataset choice to concrete loader keys."""
    if choice == _ALL_DATASETS:
        return list(_GT_LOADERS)
    return [choice]


def main() -> None:
    """Diagnose one category's localization on selected GT sources."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    predict_fn = TIER_BUILDERS[args.tier](args)
    results = {
        name: _diagnose(predict_fn, _GT_LOADERS[name](args), args.category)
        for name in _selected_datasets(args.dataset)
    }
    print(json.dumps({"category": args.category, "results": results},
                     indent=2))


if __name__ == "__main__":
    main()
