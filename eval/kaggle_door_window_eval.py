"""T-KG1 / T-FC3 / T-P2: baseline the dormant door/window detector tiers
against multiple GT sources via two registries, reusing the
score_against_gt_images bridge from gt_evaluator.

GT sources (--dataset, one registry entry each):
* test_pcs         -- commercial door/window, COCO (EvalConfig).
* kaggle_floorplans500 -- residential-styled door/window, YOLO.
* floorplancad     -- residential+commercial door/window/wall, FiftyOne
  (CC BY-NC: EVAL-ONLY, never route into shipped SFT data).

Detector tiers (--tier, default cubicasa -- unchanged prior behavior):
* cubicasa       -- CubiCasa5KDetector.detect_icons (T-KG1/T-FC3 baseline).
* vlm            -- DoorDetector/WindowDetector's dormant VLM Tier 3, gated
  by the same stripe/grid hallucination filter the room pipeline uses.
* yolo_pretrained -- T-S1 (YOLO_FINETUNING_ASSESSMENT.md): stock
  COCO-pretrained YOLO26n, no fine-tuning. Expected ~0 (COCO has no
  door/window class) -- confirms fine-tuning (T-S2) is structurally
  required, not just beneficial.
* yolo_finetuned_tiled -- R3: yolo_finetuned run through yolo_detector.py's
  tiled path (R2) instead of full-image. Decisive same-image test found
  full-page downscale (not domain gap) as the dominant recall killer;
  this tier measures the real effect on full GT sets, not one image.

Read-only: no production code path is touched.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from preprocessing_annotations.config import (
    EvalConfig,
    FloorplancadEvalConfig,
    KaggleFloorplanEvalConfig,
    VLMConfig,
    YOLO_PRETRAINED_CHECKPOINT,
    YoloObjectDetectorConfig,
    _detect_device,
)
from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from preprocessing_annotations.detection.door_detector import DoorDetection, DoorDetector
from floorplancad_gt import ensure_images, load_floorplancad_ground_truth
from gt_evaluator import (
    Detection,
    GTImage,
    PredictFn,
    cubicasa_predict,
    load_ground_truth,
    score_against_gt_images,
)
from preprocessing_annotations.vlm.hallucination_detector import detect_hallucinations
from preprocessing_annotations.vlm.vlm_backend import VLMFactory
from preprocessing_annotations.detection.window_detector import WindowDetection, WindowDetector
from preprocessing_annotations.detection.yolo_detector import YoloObjectDetector
from yolo_gt import load_kaggle_floorplan_ground_truth
from preprocessing_annotations.detection.yolo_infer import detect_objects, load_model

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
_TIER_YOLO_PRETRAINED = "yolo_pretrained"
_TIER_YOLO_FINETUNED = "yolo_finetuned"
_TIER_YOLO_FINETUNED_TILED = "yolo_finetuned_tiled"
DEFAULT_TIER = _TIER_CUBICASA

# T-S1 baseline checkpoint (YOLO_FINETUNING_ASSESSMENT.md Phase 3): stock
# COCO-pretrained YOLO26n, no fine-tuning. COCO's 80-class list has no
# door/window category, so this tier is expected to score ~0 -- run to
# confirm rather than assume, per the pipeline's evidence discipline.
# Absolute path (config.py, shared with yolo_train.py) so a bare filename
# doesn't make Ultralytics download/write into whatever CWD this script
# happens to run from.
_YOLO_PRETRAINED_CHECKPOINT = YOLO_PRETRAINED_CHECKPOINT

# T-S2 fine-tuned checkpoint (yolo_train.py's output). Not a config
# default: this tier is only usable after a training run has produced a
# best.pt, so an explicit --yolo-checkpoint flag is required rather than
# defaulting to a path that may not exist yet.


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


def build_vlm_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Load the VLM backend once and wire the VLM door/window tier."""
    backend = VLMFactory.create(VLMConfig(backend=_VLM_TIER_BACKEND))
    backend.initialize()
    return vlm_predict(backend, DoorDetector(), WindowDetector())


def build_cubicasa_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Load the CubiCasa5K detector once and bind it into a PredictFn."""
    return cubicasa_predict(_build_detector())


def yolo_model_predict(model: Any) -> PredictFn:
    """Bind any loaded YOLO checkpoint (pretrained or fine-tuned) into a
    PredictFn -- shared by the T-S1 (stock) and T-S2 (fine-tuned) tiers,
    since box->Detection conversion doesn't depend on which weights are
    loaded. Delegates to yolo_infer.detect_objects (T-I1) -- the same
    decode used by yolo_detector.py's pipeline-side detector, one
    implementation instead of two. confidence intentionally dropped here
    (Detection.confidence defaults to 1.0, unread anywhere in
    gt_evaluator's scoring) -- matches this eval's pre-T-I1 behavior
    exactly.
    """
    def predict(image_path: Path) -> List[Detection]:
        boxes = detect_objects(model, image_path, _REPORT_CATEGORIES)
        return [
            Detection(bbox=BBox(*xyxy), category=class_name)
            for class_name, xyxy, _confidence in boxes
        ]
    return predict


def build_yolo_pretrained_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Load the pretrained YOLO26n checkpoint once, bind into a PredictFn."""
    return yolo_model_predict(load_model(_YOLO_PRETRAINED_CHECKPOINT))


def build_yolo_finetuned_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Load a T-S2 fine-tuned checkpoint (yolo_train.py's output), bind
    into a PredictFn. Reuses yolo_model_predict -- same box->Detection
    mapping applies to any YOLO checkpoint, fine-tuned or not."""
    if not args.yolo_checkpoint:
        raise ValueError(
            "--yolo-checkpoint is required for --tier yolo_finetuned "
            "(no default: only usable after yolo_train.py has produced one)"
        )
    return yolo_model_predict(load_model(args.yolo_checkpoint))


def build_yolo_finetuned_tiled_predict_fn(args: argparse.Namespace) -> PredictFn:
    """R3: same fine-tuned checkpoint as yolo_finetuned, run through
    yolo_detector.py's tiled path (R2, detect_objects_tiled) instead of
    full-image -- measures whether tiling actually moves precision/
    recall on real GT, not just on the single image checked while R2
    was built.

    score_against_gt_images matches in INPUT LIST ORDER and never reads
    Detection.confidence (verified, gt_evaluator.py) -- sort by
    confidence descending here, before conversion, so the
    highest-confidence prediction claims a GT box first. Matches
    yolo_model_predict's existing convention of not carrying confidence
    into Detection (unread downstream either way).
    """
    if not args.yolo_checkpoint:
        raise ValueError(
            "--yolo-checkpoint is required for --tier yolo_finetuned_tiled "
            "(no default: only usable after yolo_train.py has produced one)"
        )
    detector_config = YoloObjectDetectorConfig(
        checkpoint_path=args.yolo_checkpoint,
        use_tiling=True,
        keep_categories=_REPORT_CATEGORIES,
    )
    # CD0: CLI override wins, config default otherwise -- same resolution
    # rule cubicasa_room_eval._resolve_sample_size already uses. Left
    # unset the dataclass default applies, so every recorded baseline for
    # this tier reproduces byte-for-byte.
    if args.yolo_conf is not None:
        detector_config.confidence_threshold = args.yolo_conf
    detector = YoloObjectDetector(config=detector_config)

    def predict(image_path: Path) -> List[Detection]:
        detections = detector.detect_objects_tiled(image_path)
        detections = sorted(
            detections, key=lambda d: d.confidence, reverse=True
        )
        return [
            Detection(bbox=BBox(*d.bbox), category=d.metadata["type"])
            for d in detections
            if d.metadata.get("type") in _REPORT_CATEGORIES
        ]
    return predict


TIER_BUILDERS: Dict[str, Callable[[argparse.Namespace], PredictFn]] = {
    _TIER_CUBICASA: build_cubicasa_predict_fn,
    _TIER_VLM: build_vlm_predict_fn,
    _TIER_YOLO_PRETRAINED: build_yolo_pretrained_predict_fn,
    _TIER_YOLO_FINETUNED: build_yolo_finetuned_predict_fn,
    _TIER_YOLO_FINETUNED_TILED: build_yolo_finetuned_tiled_predict_fn,
}


def _filter_report(
    scores: Dict[str, Any], categories: Tuple[str, ...]
) -> Dict[str, Any]:
    """Keep only the requested categories from a category->score map."""
    return {
        cat: scores[cat].to_dict() for cat in categories if cat in scores
    }


def _run_source(
    source: GtSource, predict_fn: PredictFn, with_confusion: bool = False
) -> Dict[str, Any]:
    """Score one GT source and return its door/window report.

    AF5: when ``with_confusion`` is set, the report gains a
    ``cross_category`` section counting predicted-category -> GT-category
    IoU matches. The per-category precision/recall block cannot express
    this: a correctly-located box carrying the wrong label is a false
    positive for one category and a false negative for another, never a
    confusion, so "missed" and "mislabeled" are indistinguishable there.
    Off by default -- the existing report shape is what every recorded
    baseline was measured against.
    """
    confusion: Optional[Dict[str, Dict[str, int]]] = (
        {} if with_confusion else None
    )
    scores = score_against_gt_images(
        source.gt_images, source.iou_threshold, predict_fn, confusion
    )
    report = {
        "dataset": source.name,
        "images_evaluated": len(source.gt_images),
        "iou_threshold": source.iou_threshold,
        "vs_gt": _filter_report(scores, _REPORT_CATEGORIES),
    }
    if confusion is not None:
        report["cross_category"] = confusion
    return report


def _selected_datasets(choice: str) -> List[str]:
    """Resolve the --dataset choice to concrete registry keys."""
    if choice == _ALL_DATASETS:
        return list(_GT_SOURCE_BUILDERS)
    return [choice]


def _emit(results: List[Dict[str, Any]], output_path: Optional[Path]) -> None:
    """Print the report, and additionally persist it when a path is given.

    AF3: this CLI printed to stdout only, so every measurement it ever
    produced survived just as long as the terminal scrollback -- two of
    this project's recorded baselines had to be recovered from a session
    transcript, and one config docstring drifted stale because the newer
    numbers were never written anywhere. Persisting is opt-in: with no
    --output the stdout text is byte-identical to before.

    One serialization feeds both sinks, so a persisted file can never
    disagree with what was printed -- including the trailing newline
    print() adds, so the file diffs clean against captured stdout.
    """
    payload = json.dumps(results, indent=2) + "\n"
    print(payload, end="")
    if output_path is not None:
        output_path.write_text(payload)
        logger.info("wrote report to %s", output_path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI args: tier, dataset selection, Kaggle split, sample cap."""
    parser = argparse.ArgumentParser(
        description="Door/window detector tier baseline vs GT sources"
    )
    parser.add_argument(
        "--tier", choices=tuple(TIER_BUILDERS), default=DEFAULT_TIER,
    )
    parser.add_argument(
        "--dataset", choices=(*_GT_SOURCE_BUILDERS, _ALL_DATASETS),
        default=_ALL_DATASETS,
    )
    parser.add_argument("--kaggle-split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--yolo-checkpoint", default=None,
        help="Path to a yolo_train.py checkpoint, required for "
             "--tier yolo_finetuned",
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=None,
        help="Confidence floor override for --tier yolo_finetuned_tiled "
             "(default: YoloObjectDetectorConfig.confidence_threshold). "
             "Ignored by the yolo_pretrained/yolo_finetuned tiers, which "
             "call the decoder without a threshold so their recorded "
             "baselines stay fixed",
    )
    parser.add_argument(
        "--confusion", action="store_true",
        help="Also report predicted-category -> GT-category IoU match "
             "counts (default: off, report shape unchanged)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write the JSON report to this path in addition to stdout "
             "(default: stdout only, unchanged behavior)",
    )
    return parser.parse_args()


def main() -> None:
    """Score the selected detector tier against selected GT sources."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.yolo_conf is not None and args.tier != _TIER_YOLO_FINETUNED_TILED:
        logger.warning(
            "--yolo-conf ignored: tier %s applies no confidence floor; "
            "its numbers are NOT a measurement of that threshold",
            args.tier,
        )
    predict_fn = TIER_BUILDERS[args.tier](args)
    results = [
        _run_source(_GT_SOURCE_BUILDERS[name](args), predict_fn, args.confusion)
        for name in _selected_datasets(args.dataset)
    ]
    _emit(results, args.output)


if __name__ == "__main__":
    main()
