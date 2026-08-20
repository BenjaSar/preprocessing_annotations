"""T-A follow-up: split the CNN room head's false negatives by cause.

run_room_baseline reports FN as one number. That number conflates two
different failures with different fixes:

  pure-miss     GT room has NO overlapping prediction at all (best IoU 0).
                Fix = better recall / more proposals.
  localize-fail GT room IS overlapped but below the IoU threshold.
                Fix = better boundaries / instance separation.

Mirrors the pure-miss vs localize-fail split already used for doors in
this project. Also reports the merge signal: how many GT rooms fall
inside a single prediction (undersegmentation), which is what a
boundary-aware architecture would target.
"""

import argparse
import json
import logging
from collections import Counter
from typing import Any, Dict, List

from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.config import CubicasaEvalConfig
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from cubicasa_gt import load_cubicasa_ground_truth
from cubicasa_cnn_room_eval import CNNRoomBackend
from cubicasa_room_eval import _limit_sample, _resolve_sample_size
from gt_evaluator import Detection, GTImage, iou

logger = logging.getLogger(__name__)

ROOM_CATEGORY = "room"
_LOCALIZE_FAIL_FLOOR = 0.0   # best IoU strictly above this = overlapped at all


def _containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner`'s area covered by `outer`."""
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = (inner.x2 - inner.x1) * (inner.y2 - inner.y1)
    return inter / area if area > 0 else 0.0


def _area(b: BBox) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def diagnose_image(
    predictions: List[Detection],
    gt_rooms: List[BBox],
    iou_threshold: float,
    acc: Dict[str, Any],
) -> None:
    """Accumulate FN-cause counts for one image."""
    pred_boxes = [p.bbox for p in predictions if p.category == ROOM_CATEGORY]

    # Greedy one-to-one match, same protocol as gt_evaluator.score_image.
    matched_gt = set()
    for pred in pred_boxes:
        best_idx, best_iou = None, 0.0
        for idx, gt in enumerate(gt_rooms):
            if idx in matched_gt:
                continue
            candidate = iou(pred, gt)
            if candidate > best_iou:
                best_idx, best_iou = idx, candidate
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)

    for idx, gt in enumerate(gt_rooms):
        if idx in matched_gt:
            acc["matched"] += 1
            continue
        # Unmatched GT: classify why. Use best IoU over ALL predictions
        # (not just unclaimed ones) -- we are diagnosing overlap, not
        # re-running assignment.
        best_iou = max((iou(p, gt) for p in pred_boxes), default=0.0)
        if best_iou <= _LOCALIZE_FAIL_FLOOR:
            acc["fn_pure_miss"] += 1
        else:
            acc["fn_localize_fail"] += 1
            acc["localize_fail_ious"].append(best_iou)
            # Oversize vs undersize: compare the best-overlapping pred's
            # area to GT area. >1 means prediction is too big (merge).
            best_pred = max(pred_boxes, key=lambda p: iou(p, gt))
            gt_area = _area(gt)
            if gt_area > 0:
                acc["localize_fail_area_ratios"].append(
                    _area(best_pred) / gt_area
                )

    # Merge signal: how many GT rooms are ≥80% contained in one prediction.
    for pred in pred_boxes:
        contained = sum(1 for gt in gt_rooms if _containment(gt, pred) >= 0.8)
        if contained >= 2:
            acc["preds_swallowing_multiple_gt"] += 1
            acc["gt_swallowed"] += contained
        acc["contained_hist"][min(contained, 5)] += 1


def _summarize(acc: Dict[str, Any]) -> Dict[str, Any]:
    fn_total = acc["fn_pure_miss"] + acc["fn_localize_fail"]
    ratios = acc["localize_fail_area_ratios"]
    ious = acc["localize_fail_ious"]
    return {
        "matched_tp": acc["matched"],
        "fn_total": fn_total,
        "fn_pure_miss": acc["fn_pure_miss"],
        "fn_localize_fail": acc["fn_localize_fail"],
        "fn_pure_miss_pct": round(
            100.0 * acc["fn_pure_miss"] / fn_total, 1) if fn_total else None,
        "fn_localize_fail_pct": round(
            100.0 * acc["fn_localize_fail"] / fn_total, 1) if fn_total else None,
        "localize_fail_mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "localize_fail_mean_area_ratio": round(
            sum(ratios) / len(ratios), 2) if ratios else None,
        "localize_fail_oversize_pct": round(
            100.0 * sum(1 for r in ratios if r > 1.0) / len(ratios), 1
        ) if ratios else None,
        "preds_swallowing_multiple_gt": acc["preds_swallowing_multiple_gt"],
        "gt_rooms_swallowed": acc["gt_swallowed"],
        "pred_containment_hist": dict(sorted(acc["contained_hist"].items())),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="T-A FN-cause diagnostic for the CubiCasa5K CNN room head"
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--min-area-px", type=int, default=0,
        help="Drop predictions below this area (see cubicasa_cnn_room_sweep)",
    )
    parser.add_argument(
        "--union", action="store_true",
        help="Union-mask extraction instead of per-class",
    )
    parser.add_argument(
        "--barrier-dilate-px", type=int, default=0,
        help="Wall+opening barrier dilation (see cubicasa_cnn_room_barrier_sweep)",
    )
    args = parser.parse_args()

    eval_config = CubicasaEvalConfig()
    sample_size = _resolve_sample_size(args.sample_size, eval_config)
    gt_images: List[GTImage] = load_cubicasa_ground_truth(
        eval_config, eval_config.coco_test_path
    )
    gt_images = _limit_sample(gt_images, sample_size)

    detector = CubiCasa5KDetector()
    if not detector.is_available:
        raise SystemExit(f"checkpoint not found: {detector.model_path}")
    backend = CNNRoomBackend(detector)
    backend.initialize()

    acc: Dict[str, Any] = {
        "matched": 0,
        "fn_pure_miss": 0,
        "fn_localize_fail": 0,
        "localize_fail_ious": [],
        "localize_fail_area_ratios": [],
        "preds_swallowing_multiple_gt": 0,
        "gt_swallowed": 0,
        "contained_hist": Counter(),
    }
    evaluated = 0
    for gt_image in gt_images:
        try:
            rooms = backend.detect_rooms(
                gt_image.image_path,
                per_class=not args.union,
                min_area_px=args.min_area_px,
                barrier_dilate_px=args.barrier_dilate_px,
            )
        except (RuntimeError, OSError, ValueError) as error:
            logger.warning("skipped %s: %s", gt_image.image_path.name, error)
            continue
        predictions = [
            Detection(bbox=BBox(*r["bbox"]), category=ROOM_CATEGORY)
            for r in rooms if r.get("bbox")
        ]
        gt_rooms = [
            a.bbox for a in gt_image.annotations if a.category == ROOM_CATEGORY
        ]
        diagnose_image(
            predictions, gt_rooms, eval_config.gt_iou_threshold, acc
        )
        evaluated += 1

    print(json.dumps({
        "domain": "cubicasa5k_residential",
        "backend": "cubicasa5k_cnn_dormant_head",
        "iou_threshold": eval_config.gt_iou_threshold,
        "images_evaluated": evaluated,
        "fn_split": _summarize(acc),
    }, indent=2))


if __name__ == "__main__":
    main()
