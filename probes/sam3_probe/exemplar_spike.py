"""T-S4 spike: SAM3 image-exemplar concept detection vs held-out GT.

Feeds a few ground-truth boxes per class as positive exemplars, asks SAM3
to detect all matching instances, and scores the detections against the
HELD-OUT ground truth (exemplars excluded -> no leakage).

Supports three GT sources via --dataset (SpikeDataset, dependency-
injected):
* test_pcs (default): commercial door/window, EvalConfig.
* cubicasa5k: residential room/wall bbox-only GT, CubicasaEvalConfig.
* kaggle_floorplans500: residential-styled door/window bbox-only GT,
  KaggleFloorplanEvalConfig (CC BY 4.0; the one door/window x
  residential-domain combination test_pcs/cubicasa5k don't cover).

Runs only in sam3_venv (isolated). No detector is treated as a baseline.
Because exemplars are drawn from GT, results are an optimistic upper
bound, not a production estimate.
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# cubicasa_gt/gt_evaluator/yolo_gt are eval-only scripts (not part of the
# installed package) -- put eval/ on sys.path so their bare sibling imports
# resolve, same convention those scripts use when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from preprocessing_annotations.bbox.bbox_metrics import BBox, iou
from preprocessing_annotations.config import (
    CubicasaEvalConfig,
    EvalConfig,
    KaggleFloorplanEvalConfig,
)
from cubicasa_gt import load_cubicasa_ground_truth  # noqa: E402
from gt_evaluator import (  # noqa: E402
    CategoryScore,
    Detection,
    GTAnnotation,
    GTImage,
    load_ground_truth,
    score_image,
)
from .domain import Detection as RawDetection
from .sam3_adapter import Sam3Adapter
from yolo_gt import load_kaggle_floorplan_ground_truth  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EXEMPLARS_PER_CLASS = 3

# GT dataset identifiers accepted by --dataset.
DATASET_TEST_PCS = "test_pcs"
DATASET_CUBICASA = "cubicasa5k"
DATASET_KAGGLE = "kaggle_floorplans500"
DEFAULT_DATASET = DATASET_TEST_PCS

# Categories scored per dataset (only classes each GT source actually
# has). test_pcs and kaggle_floorplans500 both cover door+window -- one
# shared constant, not two copies of the same tuple.
SPIKE_CATEGORIES_DOOR_WINDOW: Tuple[str, ...] = ("door", "window")
SPIKE_CATEGORIES_CUBICASA: Tuple[str, ...] = ("room", "wall")

# Longest-edge cap for the model input. test_pcs plans are 7200-9600 px and
# OOM a 14.6 GB T4 at full resolution; exemplar boxes are normalized so the
# resize is coordinate-safe (outputs are rescaled back to original pixels).
DEFAULT_MAX_IMAGE_DIM = 2048

# Minimum detection confidence kept from the model.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# IoU above which two same-class detections are treated as duplicates and the
# lower-scoring one is dropped (non-maximum suppression). Mirrors the repo's
# TemplateConfig.nms_iou_threshold convention.
DEFAULT_NMS_IOU_THRESHOLD = 0.3

# Fraction of each category's per-image GT reserved as the held-out scoring
# set. FIXED regardless of --exemplars (K), so recall stays comparable
# across a K sweep (T-CS3): with the old "first K / rest" split, held_out
# shrank as K grew, so recall rose with K purely from a shrinking
# denominator, not from better detection. A fixed fraction keeps the
# denominator constant; only the exemplar pool SAM sees changes with K.
HELD_OUT_FRACTION = 0.5

_NO_RESCALE = 1.0


@dataclass(frozen=True)
class SpikeThresholds:
    """Thresholds controlling detection, deduplication, and scoring."""

    confidence: float
    nms_iou: float
    match_iou: float


@dataclass(frozen=True)
class SpikeDataset:
    """A named GT source: its scored categories, match IoU, and images."""

    name: str
    categories: Tuple[str, ...]
    match_iou: float
    gt_images: List[GTImage]


def build_test_pcs_dataset() -> SpikeDataset:
    """Build the test_pcs (commercial door/window) spike dataset."""
    config = EvalConfig()
    gt_images = load_ground_truth(
        config.gt_coco_path, config.gt_images_dir, config.gt_category_merge
    )
    return SpikeDataset(
        name=DATASET_TEST_PCS,
        categories=SPIKE_CATEGORIES_DOOR_WINDOW,
        match_iou=config.gt_iou_threshold,
        gt_images=gt_images,
    )


def build_cubicasa_dataset() -> SpikeDataset:
    """Build the CubiCasa5K (residential room/wall bbox) spike dataset."""
    config = CubicasaEvalConfig()
    gt_images = load_cubicasa_ground_truth(config, config.coco_test_path)
    return SpikeDataset(
        name=DATASET_CUBICASA,
        categories=SPIKE_CATEGORIES_CUBICASA,
        match_iou=config.gt_iou_threshold,
        gt_images=gt_images,
    )


def build_kaggle_dataset() -> SpikeDataset:
    """Build the Kaggle floor-plans-500 (residential-styled door/window
    bbox) spike dataset -- the one door/window x residential-domain
    combination test_pcs/cubicasa5k don't cover (CC BY 4.0).
    """
    config = KaggleFloorplanEvalConfig()
    gt_images = load_kaggle_floorplan_ground_truth(config)
    return SpikeDataset(
        name=DATASET_KAGGLE,
        categories=SPIKE_CATEGORIES_DOOR_WINDOW,
        match_iou=config.gt_iou_threshold,
        gt_images=gt_images,
    )


_DATASET_BUILDERS: Dict[str, Callable[[], SpikeDataset]] = {
    DATASET_TEST_PCS: build_test_pcs_dataset,
    DATASET_CUBICASA: build_cubicasa_dataset,
    DATASET_KAGGLE: build_kaggle_dataset,
}


def _limit_dataset(
    dataset: SpikeDataset, sample_size: Optional[int]
) -> SpikeDataset:
    """Return `dataset` capped to `sample_size` images (unchanged if None)."""
    if sample_size is None:
        return dataset
    return replace(dataset, gt_images=dataset.gt_images[:sample_size])


def _to_cxcywh_norm(
    box: BBox, width: int, height: int
) -> Tuple[float, float, float, float]:
    """Convert an xyxy-pixel box to cxcywh normalized to [0, 1]."""
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    return cx, cy, (box.x2 - box.x1) / width, (box.y2 - box.y1) / height


def _split_exemplars(
    annotations: List[GTAnnotation], count: int
) -> Tuple[List[GTAnnotation], List[GTAnnotation]]:
    """Return (exemplars, held_out) with a FIXED held-out size.

    held_out size = ceil(len(annotations) * HELD_OUT_FRACTION), independent
    of `count` — see HELD_OUT_FRACTION for why. Exemplars are the first
    `count` annotations from the remaining (non-held-out) pool.
    """
    held_out_size = math.ceil(len(annotations) * HELD_OUT_FRACTION)
    pool = annotations[: len(annotations) - held_out_size]
    held_out = annotations[len(annotations) - held_out_size:]
    return pool[:count], held_out


def _load_rgb(image_path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def _resize_rgb(
    image: np.ndarray, max_dim: int
) -> Tuple[np.ndarray, float]:
    """Downscale so the longest edge <= max_dim; return (image, scale)."""
    height, width = image.shape[:2]
    longest_edge = max(width, height)
    if longest_edge <= max_dim:
        return image, _NO_RESCALE
    scale = max_dim / longest_edge
    resized = Image.fromarray(image).resize(
        (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
    )
    return np.array(resized), scale


def _max_iou(box: BBox, others: List[BBox]) -> float:
    """Return the largest IoU of `box` against any box in `others`."""
    return max((iou(box, other) for other in others), default=0.0)


def _nms(
    detections: List[RawDetection], iou_threshold: float
) -> List[RawDetection]:
    """Greedy non-maximum suppression by score, dropping duplicates."""
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: List[RawDetection] = []
    kept_boxes: List[BBox] = []
    for det in ordered:
        box = BBox(*det.bbox_xyxy)
        if _max_iou(box, kept_boxes) >= iou_threshold:
            continue
        kept.append(det)
        kept_boxes.append(box)
    return kept


def _rescale_detection(
    det: RawDetection, factor_x: float, factor_y: float, category: str
) -> Detection:
    """Rescale a resized-space detection to original pixels for scoring."""
    x1, y1, x2, y2 = det.bbox_xyxy
    return Detection(
        bbox=BBox(
            x1 * factor_x, y1 * factor_y, x2 * factor_x, y2 * factor_y
        ),
        category=category,
    )


def _predictions(
    adapter: Sam3Adapter,
    image: np.ndarray,
    exemplars: List[GTAnnotation],
    gt_image: GTImage,
    category: str,
    thresholds: SpikeThresholds,
) -> List[Detection]:
    """Detect from exemplars, deduplicate, and rescale to original pixels."""
    boxes = [
        _to_cxcywh_norm(a.bbox, gt_image.width, gt_image.height)
        for a in exemplars
    ]
    raw = adapter.segment_by_exemplars(image, boxes, thresholds.confidence)
    kept = _nms(raw, thresholds.nms_iou)
    logger.info("%s: raw=%d nms=%d", category, len(raw), len(kept))
    factor_x = gt_image.width / image.shape[1]
    factor_y = gt_image.height / image.shape[0]
    return [
        _rescale_detection(det, factor_x, factor_y, category) for det in kept
    ]


def _drop_exemplar_hits(
    preds: List[Detection],
    exemplars: List[GTAnnotation],
    match_iou: float,
) -> List[Detection]:
    """Remove predictions that re-detect an exemplar instance.

    Exemplars are excluded from the scored (held-out) GT, so a detection on
    an exemplar is neither a true positive nor a real false positive.
    """
    exemplar_boxes = [a.bbox for a in exemplars]
    return [
        pred for pred in preds
        if _max_iou(pred.bbox, exemplar_boxes) < match_iou
    ]


def _score_category(
    adapter: Sam3Adapter,
    image: np.ndarray,
    gt_image: GTImage,
    category: str,
    exemplar_count: int,
    thresholds: SpikeThresholds,
    scores: dict,
) -> None:
    """Score one category on one image via held-out GT."""
    category_gt = [a for a in gt_image.annotations if a.category == category]
    exemplars, held_out = _split_exemplars(category_gt, exemplar_count)
    if not exemplars or not held_out:
        return
    preds = _predictions(
        adapter, image, exemplars, gt_image, category, thresholds
    )
    preds = _drop_exemplar_hits(preds, exemplars, thresholds.match_iou)
    score_image(preds, held_out, thresholds.match_iou, scores)


def _score_one_image(
    adapter: Sam3Adapter,
    gt_image: GTImage,
    categories: Tuple[str, ...],
    exemplar_count: int,
    thresholds: SpikeThresholds,
    scores: dict,
    max_dim: int,
) -> None:
    """Score every category in `categories` on a single (resized) image."""
    image, _ = _resize_rgb(_load_rgb(gt_image.image_path), max_dim)
    for category in categories:
        _score_category(
            adapter, image, gt_image, category,
            exemplar_count, thresholds, scores,
        )


def run_spike(
    adapter: Sam3Adapter,
    dataset: SpikeDataset,
    exemplar_count: int,
    thresholds: SpikeThresholds,
    max_dim: int,
) -> dict:
    """Run the exemplar spike across every image and category in `dataset`.

    Images that raise (e.g. GPU OOM on very large plans) are logged and
    skipped so partial results survive.
    """
    scores: dict = {}
    for gt_image in dataset.gt_images:
        try:
            _score_one_image(
                adapter, gt_image, dataset.categories, exemplar_count,
                thresholds, scores, max_dim,
            )
        except (RuntimeError, OSError, ValueError) as error:
            logger.warning("skipped %s: %s", gt_image.image_path.name, error)
    return {cat: score.to_dict() for cat, score in scores.items()}


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add detection/dedup threshold flags shared by both datasets."""
    parser.add_argument(
        "--max-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
        help="Longest-edge cap for the model input",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=DEFAULT_NMS_IOU_THRESHOLD,
        help="IoU above which duplicate detections are suppressed",
    )


def _add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --dataset flag (GT source selector)."""
    parser.add_argument(
        "--dataset", choices=tuple(_DATASET_BUILDERS),
        default=DEFAULT_DATASET,
        help=(
            "GT source: test_pcs (commercial door/window), cubicasa5k "
            "(residential room/wall), or kaggle_floorplans500 "
            "(residential-styled door/window)"
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the spike."""
    parser = argparse.ArgumentParser(description="SAM3 exemplar spike")
    _add_dataset_arg(parser)
    parser.add_argument(
        "--exemplars", type=int, default=DEFAULT_EXEMPLARS_PER_CLASS,
        help="Positive exemplar boxes per class",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Cap images scored (default: all images in the dataset)",
    )
    parser.add_argument("--device", default=None, help="Inference device")
    _add_threshold_args(parser)
    return parser


def _resolve_device(cli_device: Optional[str]) -> str:
    """CLI override wins; otherwise EvalConfig's auto-detected device.

    Device is compute infra, not a GT-dataset property, so both datasets
    share this same resolution regardless of --dataset.
    """
    return cli_device or EvalConfig().device


def _print_result(
    dataset: SpikeDataset,
    thresholds: SpikeThresholds,
    max_dim: int,
    result: dict,
) -> None:
    """Print the spike result as JSON."""
    print(json.dumps({
        "dataset": dataset.name,
        "confidence": thresholds.confidence,
        "nms_iou": thresholds.nms_iou,
        "match_iou": dataset.match_iou,
        "max_dim": max_dim,
        "vs_gt_heldout": result,
    }, indent=2))


def main() -> None:
    """CLI entry point for the SAM3 exemplar spike."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    dataset = _DATASET_BUILDERS[args.dataset]()
    dataset = _limit_dataset(dataset, args.sample_size)
    thresholds = SpikeThresholds(
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        match_iou=dataset.match_iou,
    )
    adapter = Sam3Adapter(device=_resolve_device(args.device))
    result = run_spike(
        adapter, dataset, args.exemplars, thresholds, args.max_dim
    )
    _print_result(dataset, thresholds, args.max_dim, result)


if __name__ == "__main__":
    main()
