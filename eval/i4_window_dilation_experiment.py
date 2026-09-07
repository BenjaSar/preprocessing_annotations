"""I-4 (plan/implementation-plan-2026-09-01.md H-3): read-only diagnostic
separating window "scoring" failure (GT box too thin to reach IoU>=0.5)
from window "detection" failure (detector doesn't fire) on the new
commercial GT (test_pcs_commercial, 252 window instances).

Measured convention gap (commercial-dataset-g12-assessment-2026-09-01.md
Sec 2.1): new set's window short axis is 1.08 permille of page diagonal
vs the old test_pcs fixture's 3.22 permille -- 3x thinner, because the
new set excludes wall thickness. Dilating the new set's window boxes on
their short axis only, toward the old fixture's convention, is a
mechanical transform requiring no re-labeling (GT boxes only, in
memory -- no production code path touched, no file written back).

Must run at conf 0.25, not production's 0.75: at 0.75 the detector
emits only 8 window predictions against 252 GT, so dilation would be
scored against an empty detection set and the result would be
vacuous either way.

Baseline to beat (already measured, I-2 sweep,
sweep/test_pcs_commercial_conf0.25.json): window TP=6, P=.0169, R=.0238.

Pass: dilated-GT window TP materially exceeds 6 -- convention/scoring
problem, fixable without re-labeling.
Failure: TP stays ~6 -- detection-side failure dominates; window is a
genuine detection gap, park it, do not fund more window annotation.

RESULT (2026-09-01): TP 6 -> 130, R .024 -> .516, P .017 -> .365 at
conf 0.25. PASSED -- convention/scoring problem confirmed, not a
detection gap. User confirmed wall-opening as the canonical convention
(I-6); the dilation this script measured is now applied at GT-load
time in production (kaggle_door_window_eval._build_test_pcs_commercial,
via config.COMMERCIAL_WINDOW_SHORT_AXIS_DILATION), not just here. This
script is kept as the standalone as-is-vs-dilated comparison the
production path no longer prints side by side.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

from preprocessing_annotations.config import (
    YoloObjectDetectorConfig,
    commercial_coco_eval_config,
    COMMERCIAL_WINDOW_SHORT_AXIS_DILATION,
)
from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.detection.yolo_detector import YoloObjectDetector

from gt_evaluator import (
    Detection,
    dilate_category_short_axis,
    load_ground_truth,
    score_against_gt_images,
)

logger = logging.getLogger(__name__)

_CATEGORY_WINDOW = "window"
_CATEGORY_DOOR = "door"


def _build_predict_fn(checkpoint: str, conf: float):
    detector_config = YoloObjectDetectorConfig(
        checkpoint_path=checkpoint,
        use_tiling=True,
        keep_categories=(_CATEGORY_DOOR, _CATEGORY_WINDOW),
        confidence_threshold=conf,
    )
    detector = YoloObjectDetector(config=detector_config)

    def predict(image_path: Path) -> List[Detection]:
        detections = detector.detect_objects_tiled(image_path)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        return [
            Detection(bbox=BBox(*d.bbox), category=d.metadata["type"])
            for d in detections
            if d.metadata.get("type") in (_CATEGORY_DOOR, _CATEGORY_WINDOW)
        ]

    return predict


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-checkpoint", required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = commercial_coco_eval_config()
    gt_images = load_ground_truth(
        config.gt_coco_path, config.gt_images_dir, config.gt_category_merge
    )
    dilated_images = dilate_category_short_axis(
        gt_images, _CATEGORY_WINDOW, COMMERCIAL_WINDOW_SHORT_AXIS_DILATION
    )

    predict_fn = _build_predict_fn(args.yolo_checkpoint, args.yolo_conf)

    as_is_scores = score_against_gt_images(gt_images, 0.5, predict_fn)
    dilated_scores = score_against_gt_images(dilated_images, 0.5, predict_fn)

    report = {
        "dataset": "test_pcs_commercial",
        "yolo_conf": args.yolo_conf,
        "dilation_factor": COMMERCIAL_WINDOW_SHORT_AXIS_DILATION,
        "as_is": {
            cat: as_is_scores[cat].to_dict()
            for cat in (_CATEGORY_DOOR, _CATEGORY_WINDOW)
            if cat in as_is_scores
        },
        "window_short_axis_dilated": {
            cat: dilated_scores[cat].to_dict()
            for cat in (_CATEGORY_DOOR, _CATEGORY_WINDOW)
            if cat in dilated_scores
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
        logger.info("wrote report to %s", args.output)


if __name__ == "__main__":
    main()
