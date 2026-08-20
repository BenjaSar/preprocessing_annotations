"""Technique-1 spike: sweep barrier_dilate_px for wall+opening-guided
room instance separation.

Unlike cubicasa_cnn_room_sweep.py (min_area is a post-hoc numeric filter
on cached boxes), barrier_dilate_px changes segmentation BEFORE connected-
components, so each value needs its own detect_rooms() call -- no caching
across values here. min_area_px is held fixed at 2500 (the established
knee from cubicasa_cnn_room_sweep.py) so this isolates the barrier effect.
"""

import argparse
import json
import logging

from preprocessing_annotations.config import CubicasaEvalConfig
from cubicasa_gt import load_cubicasa_ground_truth
from cubicasa_cnn_room_eval import CNNRoomBackend
from cubicasa_room_eval import _limit_sample, _resolve_sample_size, run_room_baseline
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector

logger = logging.getLogger(__name__)

DEFAULT_DILATIONS = [0, 1, 2, 3, 5, 8]
BASELINE_MIN_AREA_PX = 2500  # knee from cubicasa_cnn_room_sweep.py


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Technique-1 spike: barrier_dilate_px sweep"
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--min-area-px", type=int, default=BASELINE_MIN_AREA_PX)
    parser.add_argument("--dilations", type=int, nargs="+", default=DEFAULT_DILATIONS)
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
    backend = CNNRoomBackend(detector)
    backend.initialize()

    rows = []
    for dilate_px in args.dilations:
        result = run_room_baseline(
            _DilateBoundBackend(backend, dilate_px, args.min_area_px),
            gt_images,
            eval_config.gt_iou_threshold,
        )
        room = result.get("room", {})
        rows.append({"barrier_dilate_px": dilate_px, **room})
        logger.warning("dilate_px=%d -> %s", dilate_px, room)

    print(json.dumps({
        "domain": "cubicasa5k_residential",
        "min_area_px": args.min_area_px,
        "images_evaluated": len(gt_images),
        "sweep": rows,
    }, indent=2))


class _DilateBoundBackend:
    """Binds one barrier_dilate_px value so run_room_baseline's fixed
    `backend.detect_rooms(image_path)` call shape still works."""

    def __init__(self, inner: CNNRoomBackend, dilate_px: int, min_area_px: int):
        self.inner = inner
        self.dilate_px = dilate_px
        self.min_area_px = min_area_px

    def detect_rooms(self, image_path):
        return self.inner.detect_rooms(
            image_path,
            per_class=True,
            min_area_px=self.min_area_px,
            barrier_dilate_px=self.dilate_px,
        )

    def detect_hallucinations(self, rooms, check_stripes=True):
        return self.inner.detect_hallucinations(rooms, check_stripes)


if __name__ == "__main__":
    main()
