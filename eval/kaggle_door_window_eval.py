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
* sam3_exemplar_tiled -- seeds the PRODUCTION Sam3ExemplarDetector
  (detection/sam3_exemplar_detector.py, P5) with yolo_finetuned_tiled's
  own high-confidence door detections, union-merges the result back
  with the seeds. Door only -- the validating spike found window seed
  supply near-empty and no recall gain there. Native-resolution tiling
  is required, not optional: a full-page resize puts a ~44px door under
  one 14px ViT patch token (imgsz 644 -> 0.45 tokens, measured max
  confidence .32); a 1008px native tile centered on each seed puts it
  at ~3 tokens (measured max confidence .898 on the single-image check
  that motivated this tier).
  Numbers cited for this tier have moved three times (P6 withdrawal,
  P9 grid fix, P13 dedup fix) -- do not restate any of them here, they
  go stale fast and this file has done that twice already. See
  test/test_sam3_exemplar_gt_regression.py for the current pinned
  numbers and the full repin history, and Sam3ExemplarDetectorConfig's
  docstring for the incident writeup and the open real-page question
  (C2/C6: the P13 GT win is measured at yolo_conf=0.5, is a no-op at
  production's actual 0.75 default, and P9's tile_cols/tile_rows bump
  that GT numbers credit was found losing real doors on real pages --
  unresolved as of 2026-08-18).

Read-only: no production code path is touched.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from preprocessing_annotations.config import (
    EvalConfig,
    FloorplancadEvalConfig,
    KaggleFloorplanEvalConfig,
    Sam3ExemplarDetectorConfig,
    VLMConfig,
    YOLO_PRETRAINED_CHECKPOINT,
    YoloObjectDetectorConfig,
    _detect_device,
)
from preprocessing_annotations.bbox.bbox_metrics import BBox, iou
from preprocessing_annotations.detection.cubicasa5k_detector import CubiCasa5KDetector
from preprocessing_annotations.detection.door_detector import DoorDetection, DoorDetector
from preprocessing_annotations.detection.sam3_exemplar_detector import Sam3ExemplarDetector
from floorplancad_gt import ensure_images, load_floorplancad_ground_truth
from report_io import emit_report
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
_TIER_SAM3_EXEMPLAR_TILED = "sam3_exemplar_tiled"
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


def build_sam3_exemplar_tiled_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Seed the PRODUCTION Sam3ExemplarDetector (detection/sam3_exemplar_
    detector.py) with yolo_finetuned_tiled's own door detections, union-
    merge the result back with the seeds.

    P5 (audit finding A4): this tier used to carry its OWN duplicate
    implementation -- seed-centred crops, no scale-consistency gate --
    that diverged silently from production after the P0 (native-res
    ceil() tiling) and P2 (scale-consistency gate, min_seed_area_ratio)
    fixes landed in Sam3ExemplarDetector. Running that duplicate today
    would re-measure an algorithm production no longer runs and report
    "no change" on both fixes. Deleted; this tier now calls the same
    class pipeline.py's _detect_object_mappings calls, so a dataset
    score here is a score of what ships, not a fork of it.

    Door only: keep_categories restricts the seed detector to door
    (Sam3ExemplarDetector.detect_additional also filters to door
    internally -- redundant intentionally, matches production's own
    double-filter, not a new constraint added here).

    Two distinct confidence tiers, not one -- ``--seed-conf`` (0.75)
    ONLY gates which detections are handed to SAM3 as exemplar boxes
    (Sam3ExemplarDetectorConfig.seed_confidence_threshold). The seed
    detector itself runs at its own baseline threshold (unchanged
    default, matching yolo_finetuned_tiled with no --yolo-conf override)
    so its FULL detection set forms the union's YOLO half -- collapsing
    both into one threshold was tried and measured wrong (P0/P1,
    pre-dates this file's P5 rewrite): it silently reproduces "SAM3
    seeded on the high-conf subset, unioned with only that subset"
    instead of "seeded on the subset, unioned with every YOLO detection".
    """
    seed_config = YoloObjectDetectorConfig(
        use_tiling=True, keep_categories=(_CATEGORY_DOOR,)
    )
    if args.yolo_checkpoint:
        seed_config.checkpoint_path = args.yolo_checkpoint
    # CD0 override rule (matches yolo_finetuned_tiled) -- explicit, not
    # inherited: YoloObjectDetectorConfig.confidence_threshold's class
    # default has since moved to 0.75 (config.py, "explicit instruction
    # 2026-08-09"; that docstring itself measures F1 falling on both
    # door and window at 0.75 and says to revert). The prior (now
    # deleted) duplicate's union result was measured at 0.5 -- pin it
    # explicitly so this tier's baseline half doesn't silently drift
    # with that default in either direction.
    seed_config.confidence_threshold = (
        args.yolo_conf if args.yolo_conf is not None else 0.5
    )
    seed_detector = YoloObjectDetector(config=seed_config)

    exemplar_config = Sam3ExemplarDetectorConfig(
        seed_confidence_threshold=args.seed_conf,
        sam3_confidence_threshold=args.sam3_conf,
        tile_px=args.sam3_tile_px,
        tile_target_px=args.sam3_tile_px,
    )
    exemplar_detector = Sam3ExemplarDetector(config=exemplar_config)

    def _to_bbox(xyxy: Tuple[float, float, float, float]) -> BBox:
        return BBox(*xyxy)

    def predict(image_path: Path) -> List[Detection]:
        all_doors = [
            d
            for d in seed_detector.detect_objects_tiled(image_path)
            if d.metadata.get("type") == _CATEGORY_DOOR
        ]
        baseline_dets = [
            Detection(
                bbox=_to_bbox(d.bbox), category=_CATEGORY_DOOR, confidence=d.confidence
            )
            for d in all_doors
        ]
        if not all_doors:
            return baseline_dets

        additions = exemplar_detector.detect_additional(image_path, all_doors)
        sam3_dets = [
            Detection(
                bbox=_to_bbox(a.bbox), category=_CATEGORY_DOOR, confidence=a.confidence
            )
            for a in additions
        ]
        return sorted(baseline_dets + sam3_dets, key=lambda d: d.confidence, reverse=True)

    return predict


TIER_BUILDERS: Dict[str, Callable[[argparse.Namespace], PredictFn]] = {
    _TIER_CUBICASA: build_cubicasa_predict_fn,
    _TIER_VLM: build_vlm_predict_fn,
    _TIER_YOLO_PRETRAINED: build_yolo_pretrained_predict_fn,
    _TIER_YOLO_FINETUNED: build_yolo_finetuned_predict_fn,
    _TIER_YOLO_FINETUNED_TILED: build_yolo_finetuned_tiled_predict_fn,
    _TIER_SAM3_EXEMPLAR_TILED: build_sam3_exemplar_tiled_predict_fn,
}


def _filter_report(
    scores: Dict[str, Any], categories: Tuple[str, ...]
) -> Dict[str, Any]:
    """Keep only the requested categories from a category->score map."""
    return {
        cat: scores[cat].to_dict() for cat in categories if cat in scores
    }


def _run_source(
    source: GtSource, predict_fn: PredictFn, with_confusion: bool = False,
    report_categories: Tuple[str, ...] = _REPORT_CATEGORIES,
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

    ``report_categories`` is an OUTPUT filter only -- score_against_gt_images
    (gt_evaluator.py) scores every category present in GT/predictions
    regardless of this argument; changing it can only reveal categories
    already being detected/scored, never alter detection behavior. Defaults
    to the module constant so every existing call site and recorded
    baseline is byte-identical unless a caller opts in.
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
        "vs_gt": _filter_report(scores, report_categories),
    }
    if confusion is not None:
        report["cross_category"] = confusion
    return report


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
        "--tier", choices=tuple(TIER_BUILDERS), default=DEFAULT_TIER,
    )
    parser.add_argument(
        "--dataset", choices=(*_GT_SOURCE_BUILDERS, _ALL_DATASETS),
        default=_ALL_DATASETS,
    )
    parser.add_argument("--kaggle-split", default="test")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Override which categories appear in the report (default: "
             "door,window -- every recorded baseline). OUTPUT filter only: "
             "score_against_gt_images already scores every category "
             "present in GT/predictions regardless of this flag, so this "
             "can only reveal existing scoring, never change detection. "
             "e.g. --categories door window toilet sink",
    )
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
        "--seed-conf", type=float, default=0.75,
        help="Confidence floor for the YOLO seed detections fed to "
             "--tier sam3_exemplar_tiled as exemplar boxes (default "
             "0.75 -- the threshold the validating spike used)",
    )
    parser.add_argument(
        "--sam3-conf", type=float, default=0.70,
        help="SAM3 concept-match confidence floor for --tier "
             "sam3_exemplar_tiled (default 0.70 -- the only threshold "
             "measured to beat yolo_finetuned_tiled on BOTH precision "
             "and recall on test_pcs; fit on one dataset, validate "
             "independently on --dataset floorplancad before trusting it)",
    )
    parser.add_argument(
        "--sam3-tile-px", type=int, default=1008,
        help="Native-resolution tile size (px) centered on each seed "
             "box for --tier sam3_exemplar_tiled -- must stay near "
             "SAM3's own pretrain resolution (1008, ViT patch=14) or "
             "door-sized symbols fall below one patch token and "
             "confidence collapses (measured: .32 at imgsz 1288 full-page "
             "vs .898 at 1008 native tile, same model/image/exemplar)",
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
    if args.yolo_conf is not None and args.tier not in (
        _TIER_YOLO_FINETUNED_TILED, _TIER_SAM3_EXEMPLAR_TILED,
    ):
        logger.warning(
            "--yolo-conf ignored: tier %s applies no confidence floor; "
            "its numbers are NOT a measurement of that threshold",
            args.tier,
        )
    report_categories = tuple(args.categories) if args.categories else _REPORT_CATEGORIES
    predict_fn = TIER_BUILDERS[args.tier](args)
    results = [
        _run_source(
            _GT_SOURCE_BUILDERS[name](args), predict_fn, args.confusion,
            report_categories=report_categories,
        )
        for name in _selected_datasets(args.dataset)
    ]
    emit_report(results, args.output)


if __name__ == "__main__":
    main()
