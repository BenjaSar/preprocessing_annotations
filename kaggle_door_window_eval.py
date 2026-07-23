"""T-KG1 / T-FC3 / T-P2: baseline the dormant door/window detector tiers
against multiple GT sources via two registries, reusing the
score_against_gt_images bridge from gt_evaluator.

GT sources (--dataset, one registry entry each):
* test_pcs         -- commercial door/window, COCO (EvalConfig).
* kaggle_floorplans500 -- residential-styled door/window, YOLO.
* floorplancad     -- residential+commercial door/window/wall, FiftyOne
  (CC BY-NC: EVAL-ONLY, never route into shipped SFT data).

Detector tiers (--tier, default cubicasa -- unchanged prior behavior):
* cubicasa -- CubiCasa5KDetector.detect_icons (T-KG1/T-FC3 baseline).
* vlm      -- DoorDetector/WindowDetector's dormant VLM Tier 3, gated by
  the same stripe/grid hallucination filter the room pipeline uses.

Read-only: no production code path is touched.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from config import (
    EvalConfig,
    FloorplancadEvalConfig,
    KaggleFloorplanEvalConfig,
    VLMConfig,
    _detect_device,
)
from bbox_metrics import BBox
from cubicasa5k_detector import CubiCasa5KDetector
from door_detector import DoorDetection, DoorDetector
from floorplancad_gt import ensure_images, load_floorplancad_ground_truth
from gt_evaluator import (
    Detection,
    GTImage,
    PredictFn,
    cubicasa_predict,
    load_ground_truth,
    score_against_gt_images,
)
from hallucination_detector import detect_hallucinations
from vlm_backend import VLMFactory
from window_detector import WindowDetection, WindowDetector
from yolo_gt import load_kaggle_floorplan_ground_truth

logger = logging.getLogger(__name__)

_CATEGORY_DOOR = "door"
_CATEGORY_WINDOW = "window"
_REPORT_CATEGORIES: Tuple[str, ...] = (_CATEGORY_DOOR, _CATEGORY_WINDOW)

# VLM backend used for the VLM tier -- matches the room-detection
# backend choice used throughout this project's eval work.
_VLM_TIER_BACKEND = "unsloth"
_DATASET_TEST_PCS = "test_pcs"
_DATASET_KAGGLE = "kaggle_floorplans500"
_DATASET_FLOORPLANCAD = "floorplancad"
_ALL_DATASETS = "all"

_TIER_CUBICASA = "cubicasa"
_TIER_VLM = "vlm"
_DEFAULT_TIER = _TIER_CUBICASA


@dataclass(frozen=True)
class GtSource:
    """A named GT source ready to score: images + its match IoU."""

    name: str
    gt_images: List[GTImage]
    iou_threshold: float


def _cap(gt_images: List[GTImage], sample_size: Any) -> List[GTImage]:
    """Return at most `sample_size` images (all if None)."""
    if sample_size is None:
        return gt_images
    return gt_images[:sample_size]


def _build_test_pcs(args: argparse.Namespace) -> GtSource:
    """Build the test_pcs (commercial door/window, COCO) source."""
    config = EvalConfig()
    images = load_ground_truth(
        config.gt_coco_path, config.gt_images_dir, config.gt_category_merge
    )
    return GtSource(
        _DATASET_TEST_PCS, _cap(images, args.sample_size),
        config.gt_iou_threshold,
    )


def _build_kaggle(args: argparse.Namespace) -> GtSource:
    """Build the Kaggle floor-plans-500 (YOLO) source."""
    config = KaggleFloorplanEvalConfig(split=args.kaggle_split)
    images = load_kaggle_floorplan_ground_truth(config)
    return GtSource(
        _DATASET_KAGGLE, _cap(images, args.sample_size),
        config.gt_iou_threshold,
    )


def _build_floorplancad(args: argparse.Namespace) -> GtSource:
    """Build the FloorPlanCAD (FiftyOne) source, fetching sampled images."""
    config = FloorplancadEvalConfig()
    images = _cap(load_floorplancad_ground_truth(config), args.sample_size)
    ensure_images(images, config)
    return GtSource(
        _DATASET_FLOORPLANCAD, images, config.gt_iou_threshold
    )


_GT_SOURCE_BUILDERS: Dict[str, Callable[[argparse.Namespace], GtSource]] = {
    _DATASET_TEST_PCS: _build_test_pcs,
    _DATASET_KAGGLE: _build_kaggle,
    _DATASET_FLOORPLANCAD: _build_floorplancad,
}


def _build_detector() -> CubiCasa5KDetector:
    """Load the CubiCasa5K detector once, reused across all sources."""
    detector = CubiCasa5KDetector(device=_detect_device())
    if not detector.load_model():
        raise RuntimeError("CubiCasa5K model failed to load")
    return detector


def _door_to_detection(door: DoorDetection) -> Detection:
    """Convert one VLM-tier DoorDetection to a scorer Detection."""
    return Detection(bbox=BBox(*door.bbox), category=_CATEGORY_DOOR)


def _window_to_detection(window: WindowDetection) -> Detection:
    """Convert one VLM-tier WindowDetection to a scorer Detection."""
    return Detection(bbox=BBox(*window.bbox), category=_CATEGORY_WINDOW)


def _drop_hallucinated(detections: List[Any]) -> List[Any]:
    """Discard one category's detections if they match a known VLM
    hallucination pattern (stripe/grid/identical-bbox/uniform-size loop).

    Reuses hallucination_detector.detect_hallucinations (already proven
    on VLM room output) rather than a second implementation -- it only
    needs a "bbox" key per item, and both DoorDetection/WindowDetection
    carry that. detect_hallucinations only ever returns [] (discard
    all), a PREFIX of its input, or the input unchanged, so slicing the
    original typed list to the filtered length recovers the matching
    DoorDetection/WindowDetection objects.

    CAVEAT: its uniform-size pattern normally exempts genuinely uniform
    layouts via name diversity (e.g. repeated apartment units with
    distinct names); door/window detections carry no name field, so
    that exemption never applies here -- a uniform-size door/window set
    is always treated as suspicious. Untested whether this ever fires
    in practice for real door/window data; watch the logged warnings.
    """
    if not detections:
        return detections
    bbox_dicts = [{"bbox": list(d.bbox)} for d in detections]
    kept = detect_hallucinations(bbox_dicts, check_stripes=True)
    return detections[:len(kept)]


def vlm_predict(
    vlm_backend: Any,
    door_detector: DoorDetector,
    window_detector: WindowDetector,
) -> PredictFn:
    """Bind a VLM backend + door/window detectors into a PredictFn.

    Runs the VLM tier only (DoorDetector/WindowDetector's dormant Tier
    3), bypassing their CubiCasa5K tiers entirely -- this baselines the
    VLM path in isolation, mirroring cubicasa_predict's role for the
    CubiCasa5K tier. Each category's raw output is passed through the
    same stripe/grid hallucination guard the room pipeline uses before
    conversion to scorer Detections.
    """
    def predict(image_path: Path) -> List[Detection]:
        doors = door_detector.detect_doors_from_vlm_prompt(
            vlm_backend, img_path=image_path
        )
        doors = _drop_hallucinated(doors)
        windows = window_detector.detect_windows_from_vlm_prompt(
            vlm_backend=vlm_backend, img_path=image_path
        )
        windows = _drop_hallucinated(windows)
        return (
            [_door_to_detection(d) for d in doors]
            + [_window_to_detection(w) for w in windows]
        )
    return predict


def build_vlm_predict_fn() -> PredictFn:
    """Load the VLM backend once and wire the VLM door/window tier."""
    backend = VLMFactory.create(VLMConfig(backend=_VLM_TIER_BACKEND))
    backend.initialize()
    return vlm_predict(backend, DoorDetector(), WindowDetector())


def build_cubicasa_predict_fn() -> PredictFn:
    """Load the CubiCasa5K detector once and bind it into a PredictFn."""
    return cubicasa_predict(_build_detector())


_TIER_BUILDERS: Dict[str, Callable[[], PredictFn]] = {
    _TIER_CUBICASA: build_cubicasa_predict_fn,
    _TIER_VLM: build_vlm_predict_fn,
}


def _filter_report(
    scores: Dict[str, Any], categories: Tuple[str, ...]
) -> Dict[str, Any]:
    """Keep only the requested categories from a category->score map."""
    return {
        cat: scores[cat].to_dict() for cat in categories if cat in scores
    }


def _run_source(
    source: GtSource, predict_fn: PredictFn
) -> Dict[str, Any]:
    """Score one GT source and return its door/window report."""
    scores = score_against_gt_images(
        source.gt_images, source.iou_threshold, predict_fn
    )
    return {
        "dataset": source.name,
        "images_evaluated": len(source.gt_images),
        "iou_threshold": source.iou_threshold,
        "vs_gt": _filter_report(scores, _REPORT_CATEGORIES),
    }


def _selected_datasets(choice: str) -> List[str]:
    """Resolve the --dataset choice to concrete registry keys."""
    if choice == _ALL_DATASETS:
        return list(_GT_SOURCE_BUILDERS)
    return [choice]


def _parse_args() -> argparse.Namespace:
    """Parse CLI args: tier, dataset selection, Kaggle split, sample cap."""
    parser = argparse.ArgumentParser(
        description="Door/window detector tier baseline vs GT sources"
    )
    parser.add_argument(
        "--tier", choices=tuple(_TIER_BUILDERS), default=_DEFAULT_TIER,
    )
    parser.add_argument(
        "--dataset", choices=(*_GT_SOURCE_BUILDERS, _ALL_DATASETS),
        default=_ALL_DATASETS,
    )
    parser.add_argument("--kaggle-split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Score the selected detector tier against selected GT sources."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    predict_fn = _TIER_BUILDERS[args.tier]()
    results = [
        _run_source(_GT_SOURCE_BUILDERS[name](args), predict_fn)
        for name in _selected_datasets(args.dataset)
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
