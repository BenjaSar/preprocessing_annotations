"""T-A: room-localization baseline for the dormant CubiCasa5K CNN room head.

Reuses cubicasa_room_eval.py's harness (run_room_baseline, GT loading, SAM
expansion, sample-size resolution) unmodified, so this scores on the exact
same protocol as the VLM/zero-shot baselines it's compared against. Only new
code is CNNRoomBackend, adapting CubiCasa5KDetector.detect_rooms (checkpoint
channels 21:33, dormant until this task) to the VLMBackend interface that
harness expects.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from preprocessing_annotations.config import CubicasaEvalConfig
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from preprocessing_annotations.vlm.vlm_backend import VLMBackend
from cubicasa_gt import load_cubicasa_ground_truth
from cubicasa_room_eval import (
    _build_room_expander,
    _limit_sample,
    _resolve_sample_size,
    _resolve_use_expansion,
    run_room_baseline,
)

logger = logging.getLogger(__name__)


class CNNRoomBackend(VLMBackend):
    """Adapts CubiCasa5KDetector's dormant room head to VLMBackend's shape.

    detect_hallucinations is overridden to identity: the base-class filter
    (hallucination_detector.py) targets autoregressive VLM loop artifacts
    (identical-bbox repetition, NxM grid pattern, stripe alignment). A CNN
    forward pass cannot produce an autoregressive loop, and real floor plans
    routinely have rooms that legitimately share x1/y1 (aligned walls) or
    fall near a regular grid (repeated apartment units) -- applying the VLM
    filter here would discard genuine detections, not hallucinations.
    """

    def __init__(self, detector: CubiCasa5KDetector):
        self.detector = detector

    def initialize(self) -> None:
        self.detector.load_model()

    def detect_rooms(
        self,
        image_path: Union[str, Path],
        per_class: bool = True,
        min_area_px: int = 0,
        barrier_dilate_px: int = 0,
    ) -> List[Dict[str, Any]]:
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))
        room_mask = self.detector.detect_rooms(
            image,
            per_class=per_class,
            min_area_px=min_area_px,
            barrier_dilate_px=barrier_dilate_px,
        )
        return [
            {
                "room_id": f"cnn_room_{i}",
                "polygon": None,
                "bbox": [x1, y1, x2, y2],
                "room_type": None,
                "room_name": None,
                "confidence": 1.0,
                "metadata": {},
            }
            for i, (x1, y1, x2, y2) in enumerate(room_mask.bboxes)
        ]

    def detect_hallucinations(
        self, rooms: List[Dict[str, Any]], check_stripes: bool = True
    ) -> List[Dict[str, Any]]:
        return rooms


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T-A: CubiCasa5K CNN room-localization baseline"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Cap images evaluated (default: CubicasaEvalConfig.sample_size)",
    )
    parser.add_argument(
        "--room-expansion", action="store_true",
        help="Grow room boxes to walls via SAM (production use_sam path)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    eval_config = CubicasaEvalConfig()
    sample_size = _resolve_sample_size(args.sample_size, eval_config)

    gt_images = load_cubicasa_ground_truth(
        eval_config, eval_config.coco_test_path
    )
    gt_images = _limit_sample(gt_images, sample_size)

    detector = CubiCasa5KDetector()
    if not detector.is_available:
        raise SystemExit(
            f"CubiCasa5K checkpoint not found: {detector.model_path}"
        )
    backend = CNNRoomBackend(detector)
    backend.initialize()

    use_expansion = _resolve_use_expansion(args.room_expansion, eval_config)
    expand = _build_room_expander(use_expansion)
    result = run_room_baseline(
        backend, gt_images, eval_config.gt_iou_threshold, expand
    )
    print(json.dumps({
        "domain": "cubicasa5k_residential",
        "scope": "room_localization_only_no_room_type",
        "backend": "cubicasa5k_cnn_dormant_head",
        "room_expansion": use_expansion,
        "sample_size": sample_size or len(gt_images),
        "vs_gt": result,
    }, indent=2))


if __name__ == "__main__":
    main()
