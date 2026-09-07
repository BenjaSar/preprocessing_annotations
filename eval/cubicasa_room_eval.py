"""T-CC1: room-localization baseline vs CubiCasa5K GT (residential).

Runs the configured live VLM room detector on CubiCasa5K test images and
scores class-agnostic room recall/IoU against CubiCasa's generic "room"
category. This is the project's first room-localization baseline — prior
sessions had no room ground truth anywhere.

Scope guardrail (see CubicasaEvalConfig): this validates LOCALIZATION on a
RESIDENTIAL dataset only. It does not validate room_type classification
(CubiCasa has no room-type GT) and does not validate the project's
commercial/MEP domain (test_pcs covers that domain for doors/windows only).
Do not gate a room-prompt change solely on this baseline.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.config import CubicasaEvalConfig, SAMConfig, VLMConfig
from cubicasa_gt import load_cubicasa_ground_truth
from gt_evaluator import (
    CategoryScore,
    Detection,
    GTAnnotation,
    GTImage,
    score_image,
)
from preprocessing_annotations.detection.sam_segmenter import RoomSegmenter
from preprocessing_annotations.vlm.vlm_backend import VLMBackend, VLMFactory

logger = logging.getLogger(__name__)

ROOM_CATEGORY = "room"
_BBOX_LEN = 4

RoomExpander = Callable[[List[Dict[str, Any]], Path], List[Dict[str, Any]]]


def _to_xywh_room(room: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a room dict with its bbox converted xyxy -> xywh."""
    x1, y1, x2, y2 = room["bbox"]
    return {**room, "bbox": [x1, y1, x2 - x1, y2 - y1]}


def _to_xyxy_room(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Copy an annotation with its bbox converted xywh -> xyxy."""
    x, y, w, h = annotation["bbox"]
    return {**annotation, "bbox": [x, y, x + w, y + h]}


def _expand_rooms(
    rooms: List[Dict[str, Any]],
    image_path: Path,
    segmenter: RoomSegmenter,
    max_expand_frac: float,
) -> List[Dict[str, Any]]:
    """Grow raw room boxes to room walls via SAM (production expansion)."""
    xywh = [
        _to_xywh_room(r)
        for r in rooms
        if len(r.get("bbox", [])) == _BBOX_LEN
    ]
    expanded = segmenter.refine_annotations(
        image_path, xywh, max_expand_frac=max_expand_frac
    )
    return [_to_xyxy_room(a) for a in expanded]


def _build_room_expander(use_expansion: bool) -> Optional[RoomExpander]:
    """Return a SAM room-expander closure, or None when disabled."""
    if not use_expansion:
        return None
    sam_config = SAMConfig()
    segmenter = RoomSegmenter(sam_config)

    def expand(
        rooms: List[Dict[str, Any]], image_path: Path
    ) -> List[Dict[str, Any]]:
        return _expand_rooms(
            rooms, image_path, segmenter, sam_config.max_expand_frac
        )

    return expand


def _room_predictions(
    backend: VLMBackend,
    image_path: Path,
    expand: Optional[RoomExpander],
) -> List[Detection]:
    """Detect rooms, apply the production stripe gate, optionally expand.

    ``detect_rooms`` runs the per-parse hallucination filter with
    ``check_stripes=False``; the pipeline applies the stripe gate once on the
    merged result, so this single-image eval applies it here. When ``expand``
    is given, raw boxes are grown to room walls (production ``use_sam`` path).
    """
    rooms = backend.detect_rooms(image_path)
    rooms = backend.detect_hallucinations(rooms, check_stripes=True)
    if expand is not None:
        rooms = expand(rooms, image_path)
    return [
        Detection(bbox=BBox(*room["bbox"]), category=ROOM_CATEGORY)
        for room in rooms
        if room.get("bbox") and len(room["bbox"]) == _BBOX_LEN
    ]


def _room_ground_truth(gt_image: GTImage) -> List[GTAnnotation]:
    """Return only the room-category annotations for one GT image."""
    return [a for a in gt_image.annotations if a.category == ROOM_CATEGORY]


def _limit_sample(
    gt_images: List[GTImage], sample_size: int
) -> List[GTImage]:
    """Return at most `sample_size` images (all, if sample_size <= 0)."""
    if sample_size <= 0:
        return gt_images
    return gt_images[:sample_size]


def run_room_baseline(
    backend: VLMBackend,
    gt_images: List[GTImage],
    iou_threshold: float,
    expand: Optional[RoomExpander] = None,
) -> Dict[str, Any]:
    """Score the live room detector against CubiCasa5K room GT.

    Images that raise during detection are logged and skipped so partial
    results survive. When ``expand`` is given, room boxes are grown to walls.
    """
    scores: Dict[str, CategoryScore] = {}
    evaluated = 0
    for gt_image in gt_images:
        try:
            predictions = _room_predictions(
                backend, gt_image.image_path, expand
            )
        except (RuntimeError, OSError, ValueError) as error:
            logger.warning("skipped %s: %s", gt_image.image_path.name, error)
            continue
        score_image(
            predictions, _room_ground_truth(gt_image), iou_threshold, scores
        )
        evaluated += 1
    result = {cat: score.to_dict() for cat, score in scores.items()}
    result["_images_evaluated"] = evaluated
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the room baseline."""
    parser = argparse.ArgumentParser(
        description="T-CC1: room-localization baseline vs CubiCasa5K"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Cap images evaluated (default: CubicasaEvalConfig.sample_size)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="VLM backend override (default: VLMConfig.backend)",
    )
    parser.add_argument(
        "--room-expansion", action="store_true",
        help="Grow room boxes to walls via SAM (production use_sam path)",
    )
    return parser


def _resolve_sample_size(cli_value: Any, config: CubicasaEvalConfig) -> int:
    """CLI override wins; otherwise the config default (0 = all)."""
    if cli_value is not None:
        return cli_value
    return config.sample_size or 0


def _resolve_vlm_config(backend_override: Any) -> VLMConfig:
    """Build a VLMConfig, applying a CLI backend override if given."""
    vlm_config = VLMConfig()
    if backend_override is not None:
        vlm_config.backend = backend_override
    return vlm_config


def _print_result(
    vlm_config: VLMConfig,
    room_expansion: bool,
    sample_size: int,
    gt_count: int,
    result: Dict[str, Any],
) -> None:
    """Print the baseline result as JSON."""
    print(json.dumps({
        "domain": "cubicasa5k_residential",
        "scope": "room_localization_only_no_room_type",
        "backend": vlm_config.backend,
        "room_expansion": room_expansion,
        "sample_size": sample_size or gt_count,
        "vs_gt": result,
    }, indent=2))


def _resolve_use_expansion(cli_flag: bool, config: CubicasaEvalConfig) -> bool:
    """Enable expansion from the CLI flag or the config default."""
    return cli_flag or config.use_room_expansion


def main() -> None:
    """CLI entry point for the CubiCasa5K room-localization baseline."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    eval_config = CubicasaEvalConfig()
    sample_size = _resolve_sample_size(args.sample_size, eval_config)

    gt_images = load_cubicasa_ground_truth(
        eval_config, eval_config.coco_test_path
    )
    gt_images = _limit_sample(gt_images, sample_size)

    vlm_config = _resolve_vlm_config(args.backend)
    backend = VLMFactory.create(vlm_config)
    use_expansion = _resolve_use_expansion(args.room_expansion, eval_config)
    expand = _build_room_expander(use_expansion)
    result = run_room_baseline(
        backend, gt_images, eval_config.gt_iou_threshold, expand
    )
    _print_result(
        vlm_config, use_expansion, sample_size, len(gt_images), result
    )


if __name__ == "__main__":
    main()
