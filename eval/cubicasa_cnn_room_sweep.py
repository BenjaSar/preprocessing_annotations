"""T-A follow-up: min-area sweep for the CNN room head.

Per-class instance extraction roughly doubles recall over the union-mask
variant but multiplies small fragments into false positives. This sweeps
the min-area filter to locate the precision/recall knee, running inference
ONCE per image and re-filtering per threshold (inference dominates runtime).
"""

import argparse
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.config import CubicasaEvalConfig
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from cubicasa_gt import load_cubicasa_ground_truth
from cubicasa_room_eval import _limit_sample, _resolve_sample_size
from gt_evaluator import CategoryScore, Detection, GTAnnotation, score_image

logger = logging.getLogger(__name__)

ROOM_CATEGORY = "room"
DEFAULT_THRESHOLDS = [0, 1000, 2500, 5000, 10000, 20000, 40000]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="T-A min-area sweep for the CubiCasa5K CNN room head"
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--union", action="store_true",
        help="Use the union-mask variant instead of per-class extraction",
    )
    parser.add_argument(
        "--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--barrier-dilate-px", type=int, default=0,
        help="Fixed wall+opening barrier dilation (see cubicasa_cnn_room_barrier_sweep)",
    )
    args = parser.parse_args()

    eval_config = CubicasaEvalConfig()
    sample_size = _resolve_sample_size(args.sample_size, eval_config)
    gt_images = load_cubicasa_ground_truth(
        eval_config, eval_config.coco_test_path
    )
    gt_images = _limit_sample(gt_images, sample_size)

    detector = CubiCasa5KDetector()
    if not detector.is_available:
        raise SystemExit(f"checkpoint not found: {detector.model_path}")
    detector.load_model()

    # Inference once per image; reuse boxes across thresholds.
    cached: List[Tuple[List[Tuple[float, float, float, float]],
                       List[GTAnnotation]]] = []
    for gt_image in gt_images:
        try:
            with Image.open(gt_image.image_path) as img:
                array = np.array(img.convert("RGB"))
            room_mask = detector.detect_rooms(
                array,
                per_class=not args.union,
                barrier_dilate_px=args.barrier_dilate_px,
            )
        except (RuntimeError, OSError, ValueError) as error:
            logger.warning("skipped %s: %s", gt_image.image_path.name, error)
            continue
        gt_rooms = [
            a for a in gt_image.annotations if a.category == ROOM_CATEGORY
        ]
        cached.append((room_mask.bboxes, gt_rooms))

    rows = []
    for threshold in args.thresholds:
        scores: Dict[str, CategoryScore] = {}
        for bboxes, gt_rooms in cached:
            predictions = [
                Detection(bbox=BBox(x1, y1, x2, y2), category=ROOM_CATEGORY)
                for (x1, y1, x2, y2) in bboxes
                if (x2 - x1) * (y2 - y1) >= threshold
            ]
            score_image(
                predictions, gt_rooms, eval_config.gt_iou_threshold, scores
            )
        room = scores.get(ROOM_CATEGORY)
        rows.append({
            "min_area_px": threshold,
            **({k: v for k, v in room.to_dict().items()} if room else {}),
        })

    print(json.dumps({
        "domain": "cubicasa5k_residential",
        "extraction": "union" if args.union else "per_class",
        "images_evaluated": len(cached),
        "sweep": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
